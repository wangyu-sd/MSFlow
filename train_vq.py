import wandb
import os
import sys
import shutil
import argparse
import torch
import torch.cuda.amp as amp
import torch.distributed as distrib
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

USE_DDP = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from model.utils.vc import get_version, has_changes
from model.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from model.utils.data import PaddingCollate
from model.utils.train import ScalarMetricAccumulator, count_parameters, get_optimizer, get_scheduler, log_losses, recursive_to, sum_weighted_losses

from model.models_con.pep_dataloader import PepDataset
# from models_con.flow_model import FlowModel

from model.vqpae import VQPAE
from model.plot_results import plot_codebook_dist



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/remote-home/wangyu/VQ-PAR/configs/learn_all.yaml')
    parser.add_argument('--logdir', type=str, default="/remote-home/wangyu/VQ-PAR/logs")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--from_pretrain', type=str, default="/remote-home/wangyu/VQ-PAR/logs/learn_all[main-28553e6]_2025_05_02__16_48_06/checkpoints/145755_last.pt")
    # parser.add_argument('--from_pretrain', type=str, default=None)
    parser.add_argument('--name', type=str, default='vq_ft')
    parser.add_argument('--codebook_init', default=False, action='store_true')
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
    
    # if args.from_pretrain:
    #     path = os.path.dirname(os.path.dirname(args.from_pretrain))
    #     args.config = os.path.join(path, args.config.split('/')[-1])
    
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)
    config['device'] = args.device

    # Logging
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
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=args.num_workers, pin_memory=True)
    train_iterator = inf_iterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), num_workers=args.num_workers)
    len_train_dataset = len(train_dataset)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    # model = get_model(config.model).to(args.device)
    model = VQPAE(config.model).to(args.device)
    # wandb.watch(model,log='all',log_freq=1)
    logger.info(f'Number of parameters for model: {count_parameters(model)*1e-7:.2f}M')

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])
        
    elif args.from_pretrain is not None:
        logger.info('Load pretrain model from checkpoint: %s' % args.from_pretrain)
        ckpt = torch.load(args.from_pretrain, map_location=args.device)
        logger.info(f'Loading pretrain model states from {args.from_pretrain}')
        model.load_state_dict(ckpt['model'], strict=False)
        logger.info('Done!')
    
    if not args.debug:
        print("Current Loger Dir: %s" % log_dir)
    def train(it, mode):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), args.device)

        # Forward pass
        # loss_dict, metric_dict = model.get_loss(batch) # get loss and metrics
        all_loss_dict, poc_loss_dict, pep_loss_dict = model(batch, mode=mode) # get loss and metrics  
        all_loss = sum_weighted_losses(all_loss_dict, config.train.loss_weights)
        poc_loss = sum_weighted_losses(poc_loss_dict, config.train.loss_weights)
        pep_loss = sum_weighted_losses(pep_loss_dict, config.train.loss_weights)
        loss = all_loss + poc_loss + pep_loss
        
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
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })
        if not args.debug:
            to_log = it % (len_train_dataset // config.train.batch_size // 5) == 0
            log_losses(loss, all_loss_dict, poc_loss_dict, pep_loss_dict, scalar_dict, it=it, tag='train', logger=logger, to_log=to_log)
            if it % (len_train_dataset // config.train.batch_size * 10) == 0 or it ==1:
                coodbook_cnt = model.vqvae.quantizer.batch_counts.detach().cpu().numpy()
                plot_codebook_dist(coodbook_cnt, log_dir, it)
                model.vqvae.quantizer.reset_counts()
            

    def validate(it, mode):
        scalar_accum = ScalarMetricAccumulator()
        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)

                # Forward pass
                # loss_dict, metric_dict = model.get_loss(batch)
                all_loss_dict, poc_loss_dict, pep_loss_dict = model(batch, mode=mode) # get loss and metrics  
                all_loss = sum_weighted_losses(all_loss_dict, config.train.loss_weights)
                poc_loss = sum_weighted_losses(poc_loss_dict, config.train.loss_weights)
                pep_loss = sum_weighted_losses(pep_loss_dict, config.train.loss_weights)
                loss = all_loss + poc_loss + pep_loss

                scalar_accum.add(name='loss', value=loss, batchsize=len(batch['aa']), mode='mean')
                
                for loss_name, loss_dict in zip(['all', 'poc', 'pep'], [all_loss_dict, poc_loss_dict, pep_loss_dict]):
                    if loss_dict is None:
                        continue
                    for k, v in loss_dict.items():
                        k = loss_name + "_" + k
                        scalar_accum.add(name=k, value=v.mean().cpu().item(), batchsize=len(batch['aa']), mode='mean')
                            
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
    

    
    try:
        for it in range(it_first, config.train.max_iters + 1):
            
            if model.vqvae.quantizer.init_steps >= 0 and args.codebook_init:
                if it == 1:
                    logger.info('Starting Initialisation Phase 1...')
                if model.vqvae.quantizer.init_steps == 1:
                    logger.info('Starting Initialisation Phase 2...')
                    
                train(it, mode='codebook')
                
                if model.vqvae.quantizer.init_steps <= 0 and not model.vqvae.quantizer.collect_phase:
                    logger.info('Saving models...')
                    ckpt_path = os.path.join(ckpt_dir, '%d_coodbook.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        # 'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
                    
                    logger.info('Saving models to %s' % ckpt_path)
                    break
                    
            else:
                train(it, mode='pep_and_poc')
                    
                    
            # if it % config.train.val_freq == 0:
            #     avg_val_loss = validate(it)
                # if not args.debug:
                
            
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
        