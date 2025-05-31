import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import math
from tqdm.auto import tqdm
import functools
import os
import argparse
import pandas as pd
from copy import deepcopy

from model.models_con.pep_dataloader import PepDataset
from model.utils.vc import get_version
from model.utils.misc import BlackHole, inf_iterator, load_config
from model.utils.train import recursive_to

from model.modules.common.geometry import reconstruct_backbone, reconstruct_backbone_partially, align, batch_align, local_to_global
from model.modules.protein.writers import save_pdb
from torch.utils.data.distributed import DistributedSampler
from model.utils.data import PaddingCollate

from model.models_con.torsion import full_atom_reconstruction, get_heavyatom_mask
# import wandb
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
from model.msfm import MSFlowMatching
from torch.serialization import add_safe_globals
import easydict
add_safe_globals([easydict.EasyDict])
collate_fn = PaddingCollate(eight=False)

import argparse


def mask_validate(data_saved, mask):
    for k, v in data_saved.items():
        if isinstance(v, torch.Tensor):
            data_saved[k] = v[mask]
        elif isinstance(v, list) or isinstance(v, tuple):
            data_saved[k] = [s for i, s in enumerate(v) if mask[i]]
        else:
            raise ValueError(f'Unknown type of {data_saved[k]}')
    return data_saved
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/remote-home/wangyu/VQ-PAR/configs/learn_all.yaml')
    parser.add_argument('--logdir', type=str, default="/remote-home/wangyu/VQ-PAR/log_sample")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default="/remote-home/wangyu/VQ-PAR/logs/learn_all[main-f456a86]_2025_05_22__17_06_13/checkpoints/286331_last.pt")
    # parser.add_argument('--from_pretrain', type=str, default="/remote-home/wangyu/VQ-PAR/logs/learn_all[main-4ca134a]_2025_05_08__05_16_16/checkpoints/33778_last.pt")
    # parser.add_argument("--from_pretrain", type=str, default="/remote-home/wangyu/VQ-PAR/logs/learn_all[main-71aac24]_2025_05_22__09_49_20/checkpoints/25000.pt")
    parser.add_argument('--name', type=str, default='train_par')
    parser.add_argument('--fix_seq', action='store_true', default=False, help='fix sequence')
    parser.add_argument('--fix_bb', action='store_true', default=False, help='fix backbone')
    parser.add_argument('--fix_crd', action='store_true', default=False, help='fix side-chain conf')
    parser.add_argument("--sample_num", type=int, default=64, help="number of samples")
    parser.add_argument("--sample_batch_size", type=int, default=32, help="batch size for sampling")

    args = parser.parse_args()
    os.makedirs(args.logdir,exist_ok=True)
    # Version control
    branch, version = get_version()
    version_short = '%s-%s' % (branch, version[:7])
    # if has_changes() and not args.debug:
    #     c = input('Start training anyway? (y/n) ')
    #     if c != 'y':
    #         exit()

    # Load configs

    
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


    # Logging
    if args.debug:
        logger = get_logger('sample', None)
        writer = BlackHole()
    else:
        # run = wandb.init(project=args.name, config=config, name='%s[%s]' % (config_name, args.tag))
        # if args.resume:
        #     log_dir = os.path.dirname(os.path.dirname(args.resume))
        # else:
        # log_dir = get_new_log_dir(args.logdir, prefix='%s[%s]' % (config_name, version_short), tag=args.tag)
        args.tag = args.tag + "_fix_seq" if args.fix_seq else args.tag
        args.tag = args.tag + "_fix_bb" if args.fix_bb else args.tag
        args.tag = args.tag + "_fix_crd" if args.fix_crd else args.tag
        log_dir = args.resume.split('/')
        log_dir[-2] = log_dir[-2] + "_" + log_dir[-1][:-3] + args.tag
        log_dir[-1] = "sample"
        log_dir = "/".join(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        logger = get_logger('sample', log_dir)
        res_dir = log_dir.replace("sample", "results")
        # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        # tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)

    logger.info(args)
    logger.info(config)
    config['resume'] = args.resume
    config['sample_batch_size'] = args.sample_batch_size
    
    if local_rank == 0:
        logger.info(f"可见GPU: {torch.cuda.device_count()}张")  # 应输出4
        logger.info(f"当前使用GPU: {torch.cuda.current_device()}")  
    
    if log_dir is not None:
        config['log_dir'] = log_dir
        config['res_dir'] = res_dir
    
    logger.info('Initializing DDP...')
    distrib.init_process_group(backend="nccl")

    # Data
    logger.info('Loading datasets...')
    # train_dataset = get_dataset(config.dataset.train)
    # val_dataset = get_dataset(config.dataset.val)
    # train_dataset = PepDataset(structure_dir = config.dataset.train.structure_dir, dataset_dir = config.dataset.train.dataset_dir,
    #                                         name = config.dataset.train.name, transform=None, reset=config.dataset.train.reset)
    
    
    evaluation_dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
                                            name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
    sampler = DistributedSampler(evaluation_dataset, shuffle=True)
    # train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=args.num_workers, pin_memory=True)
    # train_iterator = inf_iterator(train_loader)
    # val_loader = DataLoader(evaluation_dataset, batch_size=args.sample_batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=args.num_workers)
    eval_loader = DataLoader(evaluation_dataset, batch_size=args.sample_batch_size, collate_fn=PaddingCollate(), sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    logger.info('Test %d' % (len(evaluation_dataset)))

    # Model
    logger.info('Building model...')
    # model = get_model(config.model).to(args.device)
    # model_vq = VQPAE(config.model).to(args.device)
    logger.info('Load resume model from checkpoint: %s' % args.resume)
    ckpt = torch.load(args.resume, map_location=f"cuda:{local_rank}", weights_only=True)
    ckpt['model'] = {k.replace('module.', ''): v for k,v in ckpt['model'].items()}
    model = MSFlowMatching(ckpt['config'].model).to(device=local_rank)  
    model.load_state_dict(ckpt['model'])
    model = DDP(model, device_ids=[local_rank])
    logger.info('Done!')
    
    
    
    # wandb.watch(model,log='all',log_freq=1)
    logger.info('Load pretrain model from checkpoint: %s' % args.resume)
    logger.info(f'Number of parameters for model: {count_parameters(model)*1e-7:.2f}M')
    model.eval()


    logger.info('Start sampling for in res_dir: %s' % res_dir)
    for batch_idx, batch in enumerate(tqdm(eval_loader, desc='Test', dynamic_ncols=True)):
            # Prepare data
            
            
        batch_chain_id = [list(item) for item in zip(*batch['chain_id'])]
        
        icode = [' ' for _ in range(len(batch_chain_id[0]))]
        batch_size = batch['res_mask'].shape[0]
        
        for jdx in range(batch_size):
            data_saved = {
                            'chain_nb':batch['chain_nb'][jdx],'chain_id':batch_chain_id[jdx],'resseq':batch['resseq'][jdx],'icode':icode,
                            'aa':batch['aa'][jdx], 'mask_heavyatom':batch['mask_heavyatom'][jdx], 'pos_heavyatom':batch['pos_heavyatom'][jdx],
                            }
            data_saved = mask_validate(data_saved, batch['res_mask'][jdx])
            os.makedirs(os.path.join(res_dir, batch["id"][jdx]),exist_ok=True)
            save_pdb(data_saved, path=os.path.join(res_dir, batch["id"][jdx],f'{batch["id"][jdx]}_gt.pdb'))
        
        batch = recursive_to(batch, local_rank)    
        
        for sp_idx in tqdm(range(args.sample_num), desc="Generating Multiple Samples", dynamic_ncols=True):
            final = model.module.sample(batch, num_steps=100, sample_bb=not args.fix_bb, sample_crd=not args.fix_crd, sample_seq=not args.fix_seq)
            final = final[-1]
            # final = model.anchor_based_infer(batch, cfg=0.0, top_k=5, top_p=0.0)
            # pos_ha,_,_ = full_atom_reconstruction(R_bb=final['rotmats'],t_bb=final['trans'],angles=final['angles'],aa=final['seqs_gt'])
            pos_ha = local_to_global(final['rotmats'], final['trans'], final['crd'])
            pos_ha = F.pad(pos_ha, pad=(0,0,0,15-14), value=0.) # (B,L,A,3) pos14 A=14
            pos_new = pos_ha
            # pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
            mask_new = get_heavyatom_mask(final['seqs'])
            aa_new = final['seqs']
            
            
            for jdx in range(batch_size):
                data_saved = {
                            'chain_nb':batch['chain_nb'][jdx].cpu(),'chain_id':batch_chain_id[jdx],'resseq':batch['resseq'][jdx].cpu(),'icode':icode,
                            'aa':aa_new[jdx], 'mask_heavyatom':mask_new[jdx], 'pos_heavyatom':pos_new[jdx],
                            }
                data_saved = mask_validate(data_saved, batch['res_mask'][jdx].cpu())
                save_pdb(data_saved,path=os.path.join(res_dir, batch["id"][jdx],f'{batch["id"][jdx]}_{sp_idx}.pdb'))

    logger.info(f"Finished sampling for {args.sample_num} samples in {res_dir}")

            

