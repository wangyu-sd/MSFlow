# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import copy
# import math
# from tqdm.auto import tqdm
# import functools
# from torch.utils.data import DataLoader
# import os
# import argparse
from typing import Dict

# import pandas as pd

from model.models_con.edge import EdgeEmbedder
from model.models_con.node import NodeEmbedder
# from model.modules.common.layers import sample_from, clampped_one_hot
from model.vqpae_layer import VQPAEBlock
from model.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms
from model.modules.common.geometry import construct_3d_basis
from torch.nn.utils.rnn import pad_sequence
# from model.utils.data import mask_select_data, find_longest_true_segment, PaddingCollate
# from model.utils.misc import seed_all
# from model.utils.train import sum_weighted_losses
# from torch.nn.utils import clip_grad_norm_

# from model.modules.so3.dist import centered_gaussian,uniform_so3
# from model.modules.common.geometry import batch_align, align

# from tqdm import tqdm

# import wandb

from dm import so3_utils
from dm import all_atom

# from model.models_con.pep_dataloader import PepDataset

# from model.utils.misc import load_config
# from model.utils.train import recursive_to
# from easydict import EasyDict


from model.models_con.torsion import torsions_mask

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}

class VQPAE(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._model_cfg = cfg.encoder
        self.node_embedder = NodeEmbedder(cfg.encoder.node_embed_size,max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(cfg.encoder.edge_embed_size,max_num_heavyatoms)
        self.vqvae: VQPAEBlock = VQPAEBlock(cfg.encoder.ipa)
        # self.node_proj = nn.Linear(cfg.encoder.node_embed_size, cfg.encoder.ipa.c_s)
        # self.edge_proj = nn.Linear(cfg.encoder.edge_embed_size, cfg.encoder.ipa.c_z)
    
        
    
    def extract_fea(self, batch):
        rotmats_1 =  construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N])      
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        seqs_1 = batch['aa']

        angles_1 = batch['torsion_angle']

        context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
        structure_mask = context_mask 
        sequence_mask = context_mask 
        
        node_embed = self.node_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        edge_embed = self.edge_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        
        
        # num_batch, num_res = batch['aa'].shape
        # node_embed = self.node_proj(node_embed) # (B,L,C)
        # edge_embed = self.edge_proj(edge_embed) # (B,L,C)
        gen_mask,res_mask, angle_mask = batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()
        trans_1, _ = self.zero_center_part(trans_1, gen_mask, res_mask)
        
        batched_res = {
            "rotmats": rotmats_1,
            "trans": trans_1,
            "angles": angles_1, 
            "seqs": seqs_1, 
            "node_embed": node_embed, 
            "edge_embed": edge_embed,
            "generate_mask": gen_mask,
            "res_mask": res_mask,
        }
        
        batched_res['node_embed'] = self.vqvae.fea_fusion(batched_res)
        
        return batched_res
    
    def zero_center_part(self,pos,gen_mask,res_mask):
        """
        move pos by center of gen_mask
        pos: (B,N,3)
        gen_mask, res_mask: (B,N)
        """
        center = torch.sum(pos * gen_mask[...,None], dim=1) / (torch.sum(gen_mask,dim=-1,keepdim=True) + 1e-8) # (B,N,3)*(B,N,1)->(B,3)/(B,1)->(B,3)
        center = center.unsqueeze(1) # (B,1,3)
        # center = 0. it seems not center didnt influence the result, but its good for training stabilty
        pos = pos - center
        pos = pos * res_mask[...,None]
        return pos,center
    
    
    def fape_loss(self, pred_coords, true_coords, mask):
        # pred_coords: [B, L, 3]
        # true_coords: [B, L, 3]
        # mask: [B, L] (有效残基掩码)
        
        # 计算局部坐标系变换
        def local_transform(coords):
            # 取前三个残基定义局部坐标系
            v1 = coords[..., 1, :] - coords[..., 0, :]
            v2 = coords[..., 2, :] - coords[..., 1, :]
            
            v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)  # [B, L, 3]
            v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)  # [B, L, 3]
            normal = torch.cross(v1, v2, dim=-1)
            normal = normal / torch.norm(normal, dim=-1, keepdim=True)  # [B, L, 3]
            return torch.stack([v1, v2, normal], dim=-1)  # [B, 3, 3]
        
        pred_frames = local_transform(pred_coords)
        true_frames = local_transform(true_coords)
        
        # 计算变换后的坐标差异
        diff = torch.bmm(pred_coords, pred_frames) - torch.bmm(true_coords, true_frames)
        loss = diff.pow(2) * mask[..., None]  # [B, L, 3]
        loss = loss / (mask.sum(-1, keepdim=True) + 1e-8)[..., None]  # [B, L, 3]
        return loss.sum()
    
    
    def get_loss(self, res, fea_dict, mode, weigeht=1.):
        pred_trans, pred_rotmats, pred_angles, pred_seqs_prob = \
            res['pred_trans'], res['pred_rotmats'], res['pred_angles'], res['pred_seqs']
        
        trans, rotamats, angles, seqs = \
            fea_dict['trans'], fea_dict['rotmats'], fea_dict['angles'], fea_dict['seqs']
        gen_mask, res_mask = fea_dict['generate_mask'], fea_dict['res_mask']
        
        if mode == "codebook":
            gen_mask = res_mask
        elif mode == "poc":
            gen_mask = torch.logical_and(res_mask, 1-gen_mask)
        elif mode == 'pep':
            gen_mask = gen_mask
        else:
            raise ValueError(f"Unknown mode: {mode} in get_loss function")
        
        
        pred_trans_c, _ = self.zero_center_part(pred_trans, gen_mask, res_mask)
        trans_loss = torch.sum((pred_trans_c - trans)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        trans_loss = torch.mean(trans_loss)
        
        trans_pred_list = [pred_trans_c[i][gen_mask[i]] for i in range(gen_mask.size(0))]
        trans_true_list = [trans[i][gen_mask[i]] for i in range(gen_mask.size(0))]
        
        trans_pred_gen = pad_sequence(trans_pred_list, batch_first=True, padding_value=0.0)
        trans_true_gen = pad_sequence(trans_true_list, batch_first=True, padding_value=0.0)
        gen_mask_sm = pad_sequence([gen_mask[i][gen_mask[i]] for i in range(gen_mask.size(0))], batch_first=True, padding_value=0.0)
        
        dist_gen_pred = torch.cdist(trans_pred_gen, trans_pred_gen)
        dist_gen_pred = torch.cdist(trans_true_gen, trans_true_gen)
        
        dist_mask = gen_mask_sm[:, :, None] * gen_mask_sm[:, None, :] # (B,L,L)
        dist_loss = (dist_gen_pred - dist_gen_pred).pow(2) * dist_mask
        dist_loss = torch.sum(dist_loss, dim=(-1,-2)) / (torch.sum(dist_mask,dim=(-1,-2)) + 1e-8) # (B,)
        dist_loss = torch.mean(dist_loss)
        
        mask = (dist_gen_pred < 3.8) & (dist_gen_pred > 2.0)  # 排除相邻残基
        clash = torch.where(mask, (3.8 - dist_gen_pred).pow(2), 0.0)
        clash_loss = torch.sum(clash, dim=(-1,-2)) / (torch.sum(dist_mask,dim=(-1,-2)) + 1e-8) # (B,)
        clash_loss = torch.mean(clash_loss)
        fape_loss = self.fape_loss(trans_pred_gen, trans_true_gen, gen_mask_sm)
        # dist_loss = 0.
        # clash_loss = 0.
        # for i in range(gen_mask.size(0)):
        #     dist_i = torch.cdist(trans_pred_list[i], trans_pred_list[i])
        #     dist_j = torch.cdist(trans_true_list[i], trans_true_list[i])
        #     dist_loss += torch.mean((dist_i - dist_j).pow(2))
            
        #     mask = (dist_j < 3.8) & (dist_j > 2.0)  # 排除相邻残基
        #     clash = torch.where(mask, (3.8 - dist_i).pow(2), 0.0)
        #     clash_loss += torch.mean(clash)
        # dist_loss = dist_loss / gen_mask.size(0)
        # fape_loss = self.fape_loss(pred_trans_c, trans, gen_mask)
        
        rotamats_vec = so3_utils.rotmat_to_rotvec(rotamats)
        pred_rotmats_vec = so3_utils.rotmat_to_rotvec(pred_rotmats) 
        rot_loss = torch.sum(((rotamats_vec - pred_rotmats_vec))**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        rot_loss = torch.mean(rot_loss)
        
        
        # bb aux loss
        gt_bb_atoms = all_atom.to_atom37(trans, rotamats)[:, :, :3] 
        pred_bb_atoms = all_atom.to_atom37(pred_trans_c, pred_rotmats)[:, :, :3]
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        bb_atom_loss = torch.mean(bb_atom_loss)
        
        
        # seqs loss
        seqs_loss = F.cross_entropy(pred_seqs_prob.view(-1, pred_seqs_prob.shape[-1]), seqs.view(-1), reduction='none').view(pred_seqs_prob.shape[:-1]) # (N,L), not softmax
        seqs_loss = torch.sum(seqs_loss * gen_mask, dim=-1) / (torch.sum(gen_mask,dim=-1) + 1e-8)
        seqs_loss = torch.mean(seqs_loss)
        
        num_batch,num_res = seqs.shape
        pred_seqs = torch.argmax(pred_seqs_prob,dim=-1) # (B,L)
        # angle loss
        angle_mask_loss = torsions_mask.to(pred_rotmats.device)
        angle_mask_loss = angle_mask_loss[pred_seqs.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        angle_mask_loss = torch.cat([angle_mask_loss,angle_mask_loss],dim=-1) # (B,L,10)
        angle_mask_loss = torch.logical_and(gen_mask[...,None].bool(),angle_mask_loss)
        
        # angle aux loss
        gt_angle_vec = torch.cat([torch.sin(angles),torch.cos(angles)],dim=-1)
        pred_angle_vec = torch.cat([torch.sin(pred_angles),torch.cos(pred_angles)],dim=-1)
        # angle_loss = torch.sum(((gt_angle_vf_vec - pred_angle_vf_vec) * norm_scale)**2*gen_mask[...,None],dim=(-1,-2)) / ((torch.sum(gen_mask,dim=-1)) + 1e-8) # (B,)
        angle_loss = torch.sum(((gt_angle_vec - pred_angle_vec))**2*angle_mask_loss,dim=(-1,-2)) / (torch.sum(angle_mask_loss,dim=(-1,-2)) + 1e-8) # (B,)
        angle_loss = torch.mean(angle_loss)
        

        res_ =  {
            "trans_loss": trans_loss * weigeht,
            'rot_loss': rot_loss * weigeht,
            'bb_atom_loss': bb_atom_loss * weigeht,
            'seqs_loss': seqs_loss * weigeht,
            'angle_loss': angle_loss * weigeht,
            'dist_loss': dist_loss * weigeht,
            'fape_loss': fape_loss * weigeht,
            "clash_loss": clash_loss * weigeht,
        }
        
        for key in res.keys():
            if "loss" in key:
                res_[key] = res[key] * weigeht
        return res_
        
      
    
    def forward(self, batch, mode="codebook"):

        #encode
        fea_dict: Dict[str:torch.Tensor] = self.extract_fea(batch) # no generate mask
        res_mask, gen_mask = fea_dict['res_mask'], fea_dict['generate_mask']
        fea_dict['trans_raw'] = fea_dict['trans']
        
        fea_dict['trans'], _ = self.zero_center_part(fea_dict['trans_raw'], gen_mask, res_mask)
        all_loss, poc_loss, pep_loss = None, None, None
        
        if mode == "codebook":
            # fea_dict['trans'], _ = self.zero_center_part(fea_dict['trans_raw'], res_mask, res_mask)
            res = self.vqvae.forward_init(fea_dict, mode="pep_given_poc")
            pep_loss = self.get_loss(res, fea_dict, mode='pep')
        
        elif mode == "poc_and_pep":
            pass
            # fea_dict['trans'], _ = self.zero_center_part(fea_dict['trans_raw'], res_mask, res_mask)
            # res = self.vqvae(fea_dict, mode=mode)
            # all_loss = self.get_loss(res, fea_dict)
            
        # elif mode == "poc_or_pep":
        elif mode == "pep_given_poc":
            # poc_mask = torch.logical_and(res_mask, 1-gen_mask)
            # fea_dict['trans'], _ = self.zero_center_part(fea_dict['trans_raw'], poc_mask, poc_mask)
            # poc_res = self.vqvae(fea_dict, mode='poc_only')
            
            # poc_loss = self.get_loss(poc_res, fea_dict, mode='poc', weigeht=0.1)
            
            ####### For Peptide
            pep_res = self.vqvae(fea_dict, mode='pep_given_poc')
            pep_loss = self.get_loss(pep_res, fea_dict, mode='pep')
        

        return all_loss, poc_loss, pep_loss

# if __name__ == '__main__':
#     prefix_dir = './pepflowww'
#     # config,cfg_name = load_config("../configs/angle/learn_sc.yaml")
#     config,cfg_name = load_config(os.path.join(prefix_dir,"configs/angle/learn_sc.yaml"))
#     # print(config)
#     device = 'cuda:0'
#     dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
#                                             name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=4, pin_memory=True)
#     ckpt = torch.load("./checkpoints/600000.pt", map_location=device)
#     seed_all(114514)
#     model = FlowModel(config.model).to(device)
#     model.load_state_dict(process_dic(ckpt['model']))
#     model.eval()
    
#     # print(model)

#     # print(dataset[0]['chain_id'])
#     # print(dataset[0]['id']) 
#     # print(dataset[0]['resseq'])
#     # print(dataset[0]['res_nb'])
#     # print(dataset[0]['icode'])

#     dic = {'id':[],'len':[],'tran':[],'aar':[],'rot':[],'trans_loss':[],'rot_loss':[]}

#     # for batch in tqdm(dataloader):
#     #     batch = recursive_to(batch,device)
#     for i in tqdm(range(len(dataset))):
#         item = dataset[i]
#         data_list = [deepcopy(item) for _ in range(16)]
#         batch = recursive_to(collate_fn(data_list),device)
#         loss_dic = model(batch)
#         # traj_1 = model.sample(batch,num_steps=50,sample_bb=False,sample_ang=True,sample_seq=False)
#         traj_1 = model.sample(batch,num_steps=50,sample_bb=True,sample_ang=True,sample_seq=True)
#         ca_dist = torch.sqrt(torch.sum((traj_1[-1]['trans']-traj_1[-1]['trans_1'])**2*batch['generate_mask'][...,None].cpu().long()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu()) # rmsd
#         rot_dist = torch.sqrt(torch.sum((traj_1[-1]['rotmats']-traj_1[-1]['rotmats_1'])**2*batch['generate_mask'][...,None,None].long().cpu()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu()) # rmsd
#         aar = torch.sum((traj_1[-1]['seqs']==traj_1[-1]['seqs_1']) * batch['generate_mask'].long().cpu()) / (torch.sum(batch['generate_mask']).cpu() + 1e-8)
        

#         print(loss_dic)
#         print(f'tran:{ca_dist},rot:{rot_dist},aar:{aar},len:{batch["generate_mask"].sum().item()}')

#         # free
#         torch.cuda.empty_cache()
#         gc.collect()
        
#     #     dic['tran'].append(ca_dist.item())
#     #     dic['rot'].append(rot_dist.item())
#         dic['aar'].append(aar.item())
#         dic['trans_loss'].append(loss_dic['trans_loss'].item())
#         dic['rot_loss'].append(loss_dic['rot_loss'].item())
#         dic['id'].append(batch['id'][0])
#         dic['len'].append(batch['generate_mask'].sum().item())
#     #     # break

#     #     traj_1[-1]['batch'] = batch
#     #     torch.save(traj_1[-1],f'/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Models_new/Pack_new/outputs/{batch["id"][0]}.pt')

#         # print(dic)
#     # dic = pd.DataFrame(dic)
#     # dic.to_csv(f'/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Models_new/Pack/outputs.csv',index=None)

#     print(np.mean(dic['aar']))
#     print(np.mean(dic['trans_loss']))

# if __name__ == '__main__':
#     config,cfg_name = load_config("./configs/angle/learn_angle.yaml")
#     seed_all(114514)
#     device = 'cpu'
#     dataset = PepDataset(structure_dir = config.dataset.train.structure_dir, dataset_dir = config.dataset.train.dataset_dir,
#                                             name = config.dataset.train.name, transform=None, reset=config.dataset.train.reset)
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=PaddingCollate(), num_workers=4, pin_memory=True)
#     model = FlowModel(config.model).to(device)
#     optimizer = torch.optim.Adam(model.parameters(),lr=1.e-4)

#     # ckpt = torch.load('./checkpoints/90000.pt',map_location=device)
#     # model.load_state_dict(process_dic(ckpt['model']))
#     # optimizer.load_state_dict(ckpt['optimizer'])
    
    
#     # torch.autograd.set_detect_anomaly(True)
#     for i,batch in tqdm(enumerate(dataloader)):
#         batch = recursive_to(batch,device)
#         loss_dict = model(batch)
#         loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
#         # if torch.isnan(loss):
#         #     print(i)
#         #     print(batch['id'])

#         loss.backward()
#         orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

#         print(f'{loss_dict},{loss},{orig_grad_norm}')

#         optimizer.step()
#         optimizer.zero_grad()