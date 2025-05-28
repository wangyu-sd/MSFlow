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


from model.utils.vc import get_version, has_changes
from model.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from model.utils.data import PaddingCollate
from model.utils.train import ScalarMetricAccumulator, count_parameters, get_optimizer, get_scheduler, log_losses, recursive_to, sum_weighted_losses

from model.models_con.pep_dataloader import PepDataset
# from models_con.flow_model import FlowModel

from model.msfm import MSFlowMatching
from model.plot_results import plot_codebook_dist
from easydict import EasyDict
torch.serialization.add_safe_globals([EasyDict])
# x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
# batch = torch.tensor([0, 0, 0, 0])
# edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/remote-home/wangyu/VQ-PAR/configs/learn_all.yaml')
    parser.add_argument('--logdir', type=str, default="/remote-home/wangyu/VQ-PAR/logs")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    # parser.add_argument('--resume', type=str, default="/remote-home/wangyu/VQ-PAR/logs/learn_all[main-0338939]_2025_05_04__07_58_40/checkpoints/3971_last.pt")
    # parser.add_argument('--resume', type=str, default="/remote-home/wangyu/VQ-PAR/logs/learn_all[main-4ca134a]_2025_05_08__05_16_16/checkpoints/24119_last.pt")
    parser.add_argument('--resume', type=str, default=None)
    # parser.add_argument('--from_pretrain', type=str, default="/remote-home/wangyu/VQ-PAR/logs/learn_all[main-4ca134a]_2025_05_08__05_16_16/checkpoints/24119_last.pt")
    parser.add_argument('--from_pretrain', type=str, default=None)
    parser.add_argument('--name', type=str, default='vq_ft')
    parser.add_argument('--codebook_init', default=False, action='store_true')
    # parser.add_argument('--local-rank', type=int, help='Local rank. Necessary for using the torch.distributed.launch utility.')
    args = parser.parse_args()

    args.name = args.name


    
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)
    config['device'] = args.device


    # train_dataset = get_dataset(config.dataset.train)
    # val_dataset = get_dataset(config.dataset.val)
    train_dataset = PepDataset(structure_dir = config.dataset.train.structure_dir, dataset_dir = config.dataset.train.dataset_dir,
                                            name = config.dataset.train.name, transform=None, reset=config.dataset.train.reset)
    val_dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
                                            name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
    # train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(), num_workers=args.num_workers, pin_memory=True)
    train_iterator = inf_iterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), num_workers=args.num_workers)
    len_train_dataset = len(train_dataset)


    train_ids, val_ids = set(), set()
    
    print("train_dataset", len(train_dataset))
    print("val_dataset", len(val_dataset))
    
    for idx in tqdm(range(len(train_dataset))):
        train_ids.add(train_dataset[idx]['id'])
        
    for idx in tqdm(range(len(val_dataset))):
        val_ids.add(val_dataset[idx]['id'])
    
    print("train_ids", len(train_ids))
    print("val_ids", len(val_ids))
    
    
            