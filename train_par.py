import wandb
import os
import sys
import shutil
import argparse
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.distributed as distrib
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

from model.utils.vc import get_version, has_changes
from model.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from model.utils.data import PaddingCollate
from model.utils.train import ScalarMetricAccumulator, count_parameters, get_optimizer, get_scheduler, log_losses, recursive_to, sum_weighted_losses

from model.models_con.pep_dataloader import PepDataset
from model.par import PAR
# from models_con.flow_model import FlowModel
from torch.serialization import add_safe_globals
import easydict
add_safe_globals([easydict.EasyDict])

from model.vqpae import VQPAE

from collections import defaultdict
import heapq
import numpy as np

class DynamicIndexStats:
    def __init__(self):
        self.counter = defaultdict(int)  # 索引计数器
        self.total = 0                  # 总索引数
        # self.history = []                # 历史批次轨迹（可选）

    def update(self, batch_indices):
        """
        
        参数:
            batch_indices: 形状为[B, L]的二维索引数组/张量
        """
        # 展平处理并转换类型
        if isinstance(batch_indices, (torch.Tensor, np.ndarray)):
            flat_indices = batch_indices.reshape(-1).tolist()
        else:
            flat_indices = [i for row in batch_indices for i in row]
        
        # 增量更新计数器
        for idx in flat_indices:
            self.counter[idx] += 1
        self.total += len(flat_indices)
        # self.history.append(flat_indices)  # 可选历史记录

    def get_freq_stats(self):
        """获取频率统计结果
        
        返回:
            {
                'top5': [(index, 归一化频率), ...], 
                'bottom5': [(index, 归一化频率), ...]
            }
        """
        if self.total == 0:
            return {'top5': [], 'bottom5': []}

        # 获取最高频率前5项
        top5 = heapq.nlargest(5, self.counter.items(), key=lambda x: x[1])
        # 获取最低频率后5项
        bottom5 = heapq.nsmallest(5, self.counter.items(), key=lambda x: x[1])
        
        # 计算归一化频率[6]
        normalized = lambda x: (x[0], round(x[1]/self.total, 4))
        return {
            'top5': [normalized(item) for item in top5],
            'bottom5': [normalized(item) for item in bottom5]
        }

    def reset(self):
        """重置统计器"""
        self.counter.clear()
        self.total = 0
        # self.history = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/remote-home/wangyu/VQ-PAR/configs/learn_all.yaml')
    parser.add_argument('--logdir', type=str, default="/remote-home/wangyu/VQ-PAR/log_par")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--from_pretrain', type=str, default="logs/learn_all[main-f067471]_2025_05_04__15_09_44/checkpoints/15000.pt")
    # /remote-home/wangyu/VQ-PAR/log_par/learn_all[main-d443eff]_2025_04_23__15_00_18/checkpoints/36000.pt
    parser.add_argument('--name', type=str, default='train_par')
    # parser.add_argument('--codebook_init', default=False, action='store_true')
    args = parser.parse_args()

    args.name = args.name
    # Version control
    branch, version = get_version()
    version_short = '%s-%s' % (branch, version[:7])
    if has_changes() and not args.debug:
        c = input('Start training anyway? (y/n) ')
        if c != 'y':
            exit()

    # Load configs

    
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)
    config['device'] = args.device

    # Logging
    ckpt_dir = None
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        run = wandb.init(project=args.name, config=config, name='%s[%s]' % (config_name, args.tag))
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix='%s[%s]' % (config_name, version_short), tag=args.tag)
        with open(os.path.join(log_dir, 'commit.txt'), 'w') as f:
            f.write(branch + '\n')
            f.write(version + '\n')
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        # tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    # train_dataset = get_dataset(config.dataset.train)
    # val_dataset = get_dataset(config.dataset.val)
    train_dataset = PepDataset(structure_dir = config.dataset.train.structure_dir, dataset_dir = config.dataset.train.dataset_dir,
                                            name = config.dataset.train.name, transform=None, reset=config.dataset.train.reset)
    val_dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
                                            name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size_pr, shuffle=True, collate_fn=PaddingCollate(), num_workers=args.num_workers, pin_memory=True)
    train_iterator = inf_iterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size_pr, shuffle=False, collate_fn=PaddingCollate(), num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))
    len_train_dataset = len(train_dataset)
    # Model
    logger.info('Building model...')
    # model = get_model(config.model).to(args.device)
    # model_vq = VQPAE(config.model).to(args.device)
    logger.info('Load pretrain model from checkpoint: %s' % args.from_pretrain)
    ckpt = torch.load(args.from_pretrain, map_location=args.device, weights_only=True)
    model_vq = VQPAE(ckpt['config'].model).to(args.device)  
    ckpt['model']['vqvae.quantizer.collected_samples'] = model_vq.vqvae.quantizer.collected_samples
    model_vq.load_state_dict(ckpt['model'])
    logger.info('Done!')
    
    model_par: PAR = PAR(model_vq, config).to(args.device)
    # wandb.watch(model,log='all',log_freq=1)
    logger.info(f'Number of parameters for model: {count_parameters(model_vq)*1e-7:.2f}M')
    logger.info(f'Number of parameters for model_par: {count_parameters(model_par)*1e-7:.2f}M')

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model_par)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model_par.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])
        
        
    model = model_par
    count = DynamicIndexStats()
    def train(it, mode):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), args.device)
        
        logits_BLV, gt_BL = model(batch) # get loss and metrics
        loss = F.cross_entropy(logits_BLV.view(-1, logits_BLV.size(-1)), gt_BL.view(-1), reduction='none')
        loss = loss.mean()
        
        count.update(gt_BL)
        
        # loss = loss / config.train.accum_grad
        time_forward_end = current_milli_time()

        if torch.isnan(loss):
            print('NAN Loss!')
            loss = torch.tensor(0.,requires_grad=True).to(loss.device)

        loss.backward()

        # rescue for nan grad
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    param.grad[torch.isnan(param.grad)] = 0

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        # Backward
        # if it % config.train.accum_grad ==0:
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        scalar_dict = {}
        # scalar_dict.update(metric_dict['scalar'])
        scalar_dict.update({
            'loss_tt': loss.item(),
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })
        if not args.debug:
            to_log = it % (len_train_dataset // config.train.batch_size // 5) == 0
            log_losses(loss, None, None, None, scalar_dict, it=it, tag='train', logger=logger, counter=None, to_log=to_log)

    def validate(it, mode):
        scalar_accum = ScalarMetricAccumulator()
        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)

                logits_BLV, gt_BL = model(batch) # get loss and metrics
                loss = F.cross_entropy(logits_BLV.view(-1, logits_BLV.size(-1)), gt_BL.view(-1), reduction='none')
                loss = loss.mean()

                scalar_accum.add(name='loss', value=loss, batchsize=len(batch['aa']), mode='mean')

                            
        avg_loss = scalar_accum.get_average('loss')
        summary = scalar_accum.log(it, 'val', logger=logger, writer=writer)
        if not args.debug:      
            for k,v in summary.items():
                wandb.log({f'val/{k}': v}, step=it)
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss
    

    
    print("Ckpt Path:", ckpt_dir)
    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it, mode='pep_given_poc')
            
            if it % config.train.val_freq == 0:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    # 'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        ckpt_path = os.path.join(ckpt_dir, '%d_last.pt' % it)
        torch.save({
            'config': config,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': it,
            # 'avg_val_loss': avg_val_loss,
        }, ckpt_path)
        logger.info('Terminating...')
        print('Current iteration: %d' % it)
        print("Log dir:", log_dir)
        print('Last checkpoint saved to %s' % ckpt_path)
        
        

        