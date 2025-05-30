# import numpy as np
import math
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
from typing import Dict, Tuple

# import pandas as pd

from model.models_con.edge import EdgeEmbedder
from model.models_con.node import NodeEmbedder
# from model.modules.common.layers import sample_from, clampped_one_hot
from model.encoder_layer import VQPAEBlock
from model.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms
from model.modules.common.geometry import construct_3d_basis, batch_align_with_r
from torch.nn.utils.rnn import pad_sequence
from model.models_con import torus
from openfold.utils import rigid_utils as ru
# from equiformer_pytorch import Equiformer
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
        self.node_embedder = NodeEmbedder(cfg.encoder.node_embed_size, 3)
        self.edge_embedder = EdgeEmbedder(cfg.encoder.edge_embed_size, 3)
        self.vqvae: VQPAEBlock = VQPAEBlock(cfg.encoder.ipa)
        self.strc_loss_fn = ProteinStructureLoss()
        # self.node_proj = nn.Linear(cfg.encoder.node_embed_size, cfg.encoder.ipa.c_s)
        # self.edge_proj = nn.Linear(cfg.encoder.edge_embed_size, cfg.encoder.ipa.c_z)
    
    def extract_fea(self, batch):
        with torch.no_grad():
            rotmats_1 = construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N])  
            trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
            seqs_1 = batch['aa']
            context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
            trans_1, rotmats_1 = align_to_principal_axis(trans_1, rotmats_1,context_mask)
            
            angles_1 = batch['torsion_angle']
            # poc_mask = torch.logical_and(batch['res_mask'], ~batch['generate_mask'])
        # context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
        # structure_mask = context_mask 
        # sequence_mask = context_mask
        trans_1 = trans_1 * batch['res_mask'][..., None]
        rotmats_1 = rotmats_1 * batch['res_mask'][..., None, None]
        # rotmats_avg = avg_rotation(rotmats_1, poc_mask)
        # rotmats_1 = rotmats_1.unsqueeze(1) @ rotmats_avg
        trans_1, _ = self.zero_center_part(trans_1,  batch['generate_mask'], batch['res_mask'])
        # trans_1 = (rotmats_1.unsqueeze(1) @ trans_1.transpose(-1, -2)).transpose(-1, -2)n
    
        # trans_1 = self.zero_center_part()
        node_embed = self.node_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=context_mask, sequence_mask=context_mask)
        edge_embed = self.edge_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=context_mask, sequence_mask=context_mask)
        
        
        # num_batch, num_res = batch['aa'].shape
        # node_embed = self.node_proj(node_embed) # (B,L,C)
        # edge_embed = self.edge_proj(edge_embed) # (B,L,C)
        gen_mask,res_mask, angle_mask = batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()
        
        # trans_1, _ = self.zero_center_part(trans_1, gen_mask, res_mask)
        
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
        return pos, center
    
    
    
    def get_loss(self, res, fea_dict, mode, weigeht=1.):
        pred_trans, pred_rotmats, pred_angles, pred_seqs_prob = \
            res['pred_trans'], res['pred_rotmats'], res['pred_angles'], res['pred_seqs']
        
        trans, rotamats, angles, seqs = \
            fea_dict['trans'], fea_dict['rotmats'], fea_dict['angles'], fea_dict['seqs']
        gen_mask, res_mask = fea_dict['generate_mask'], fea_dict['res_mask']
        
        if mode == "codebook" or mode == 'all':
            gen_mask = res_mask
        elif mode == "poc":
            gen_mask = torch.logical_and(res_mask, 1-gen_mask)
        elif mode == 'pep':
            gen_mask = gen_mask
        else:
            raise ValueError(f"Unknown mode: {mode} in get_loss function")
        
        
        pred_trans_c, _ = self.zero_center_part(pred_trans, gen_mask, res_mask)
        pred_trans_gen = self.strc_loss_fn.extract_fea_from_gen(pred_trans_c, gen_mask)
        trans_gen = self.strc_loss_fn.extract_fea_from_gen(trans, gen_mask)
        pred_rotamats_gen, rotamats_gen = self.strc_loss_fn.extract_fea_from_gen(pred_rotmats, gen_mask), self.strc_loss_fn.extract_fea_from_gen(rotamats, gen_mask)
        gen_mask_sm = self.strc_loss_fn.extract_fea_from_gen(gen_mask, gen_mask)
        
        # # Add global rotation ===========
        # trans_gen =  (rotamats_gen[:, 0:1].transpose(-1, -2) @ trans_gen.unsqueeze(-1)).squeeze(-1)
        # # ===============================
        
        
        # pred_trans_gen, _, rot = batch_align_with_r(pred_trans_gen, trans_gen, gen_mask_sm.bool())
        trans_loss = torch.sum((pred_trans_gen - trans_gen)**2*gen_mask_sm[...,None],dim=(-1,-2)) / (torch.sum(gen_mask_sm,dim=-1) + 1e-8) # (B,)
        trans_loss = torch.mean(trans_loss)
        
        
        strc_loss = self.strc_loss_fn(pred_trans_gen, trans_gen, gen_mask_sm)
        
        # # cleanning rotamats =================
        
        # global_rot = rotamats_gen[:, 0].clone()
        # rotamats_gen = rotamats_gen[:, 0:1].transpose(-1, -2) @ rotamats_gen
        # # ====================================
        rotamats_vec = so3_utils.rotmat_to_rotvec(rotamats_gen) 
        # pred_rotmats_vec = so3_utils.rotmat_to_rotvec(global_rot.unsqueeze(dim=1)@pred_rotamats_gen) 
        pred_rotmats_vec = so3_utils.rotmat_to_rotvec(pred_rotamats_gen)
        rot_loss = torch.sum(((rotamats_vec - pred_rotmats_vec))**2*gen_mask_sm[...,None],dim=(-1,-2)) / (torch.sum(gen_mask_sm,dim=-1) + 1e-8) # (B,)
        rot_loss = torch.mean(rot_loss)
        
        # Calculate Global Vec ==================================
        # if rotate:
        #     global_rot_vec = so3_utils.rotmat_to_rotvec(global_rot)
        #     global_rot_vec_pred = so3_utils.rotmat_to_rotvec(res['pred_rotmats'])
        #     poc_mask = torch.logical_and(res_mask, 1-gen_mask)
        #     global_rot_vec_pred = (global_rot_vec_pred * poc_mask[...,None]).sum(dim=1) / (poc_mask.sum(dim=1, keepdim=True) + 1e-6)
        #     global_rot_loss = (global_rot_vec - global_rot_vec_pred).pow(2).sum(dim=-1)
        #     global_rot_loss = global_rot_loss.mean()
        # # global_rotmats_loss = torch.mean(global_pred_rotmats_vec**2)
        # else:
        #     # ==========================================================
        #     global_rot_loss = 0.
        global_rot_loss = 0.
        
        # bb aux loss
        gt_bb_atoms = all_atom.to_atom37(trans_gen, rotamats_gen)[:, :, :3] 
        pred_bb_atoms = all_atom.to_atom37(pred_trans_gen, pred_rotamats_gen)[:, :, :3]
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask_sm[..., None, None],
            dim=(-1, -2, -3)
        ) / (torch.sum(gen_mask_sm,dim=-1) + 1e-8) # (B,)
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
        angle_mask_loss = torch.logical_and(gen_mask[...,None].bool(), angle_mask_loss)
        
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
            'dist_loss': strc_loss['dist_loss'] * weigeht,
            "clash_loss": strc_loss['clash_loss'] * weigeht,
            "bb_angle_loss": strc_loss['bb_angle_loss'] * weigeht,
            "bb_torsion_loss": strc_loss['bb_torsion_loss'] * weigeht,
            "global_rotmats_loss": global_rot_loss * weigeht,
        }
        
        for key in res.keys():
            if "loss" in key:
                res_[key] = res[key] * weigeht
        return res_
        
      
    
    def forward(self, batch, mode="codebook"):

        #encode
        fea_dict: Dict[str:torch.Tensor] = self.extract_fea(batch) # no generate mask
        # res_mask, gen_mask = fea_dict['res_mask'], fea_dict['generate_mask']
        # fea_dict['trans_raw'] = fea_dict['trans']
        
        # fea_dict['trans'], _ = self.zero_center_part(fea_dict['trans_raw'], gen_mask, res_mask)
        all_loss, poc_loss, pep_loss = None, None, None
        
        if mode == "codebook":
            # fea_dict['trans'], _ = self.zero_center_part(fea_dict['trans_raw'], res_mask, res_mask)
            res = self.vqvae.forward_init(fea_dict, mode="pep_and_poc")
            pep_loss = self.get_loss(res, fea_dict, mode='pep_and_poc')
        
        elif mode == "poc_and_pep" or mode == "pep_and_poc":
            # pass
            # fea_dict['trans'], _ = self.zero_center_part(fea_dict['trans_raw'], res_mask, res_mask)
            res = self.vqvae(fea_dict, mode="poc_and_pep")
            all_loss = self.get_loss(res, fea_dict, "all")
            
        elif mode == "pep_or_poc":
            # res = self.vqvae(fea_dict, mode="poc")
            # poc_loss = self.get_loss(res, fea_dict, "poc", weigeht=0.05)
            
            res_pep = self.vqvae(fea_dict, mode="pep_given_poc")
            pep_loss = self.get_loss(res_pep, fea_dict, "pep")
            
        elif mode == "pep_given_poc":
            # poc_mask = torch.logical_and(res_mask, 1-gen_mask)
            # fea_dict['trans'], _ = self.zero_center_part(fea_dict['trans_raw'], poc_mask, poc_mask)
            # poc_res = self.vqvae(fea_dict, mode='poc_only')
            
            # poc_loss = self.get_loss(poc_res, fea_dict, mode='poc', weigeht=0.1)
            
            ####### For Peptide
            pep_res = self.vqvae(fea_dict, mode='pep_given_poc')
            pep_loss = self.get_loss(pep_res, fea_dict, mode='all')
        

        return all_loss, poc_loss, pep_loss



def compute_principal_axis(trans: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    输入：
        trans: [B, N, 3] 位置矩阵
        node_mask: [B, N] 有效点掩码
    输出：
        rotation_matrix: [B, 3, 3] 将点云对齐到主轴的旋转矩阵
    """
    B, N, _ = trans.shape
    
    # 计算有效点的质心
    masked_trans = trans * node_mask.unsqueeze(-1)  # [B, N, 3]
    valid_count = node_mask.sum(dim=1, keepdim=True) + 1e-7  # [B, 1]
    centroid = masked_trans.sum(dim=1) / valid_count  # [B, 3]
    
    # 去中心化
    centered = trans - centroid.unsqueeze(1)  # [B, N, 3]
    
    # 计算协方差矩阵（考虑掩码）
    cov_matrix = torch.einsum('bni,bnj->bij', centered * node_mask.unsqueeze(-1), centered * node_mask.unsqueeze(-1)) / valid_count[..., None]  # [B, 3, 3]
    
    # 特征分解
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix, UPLO='U')  # [B,3], [B,3,3]
    
    # 按特征值降序排列（主轴为最大特征值对应向量）
    sorted_indices = torch.argsort(eigenvalues, descending=True, dim=1)
    eigenvectors = torch.stack([eigenvectors[b, :, sorted_indices[b]] for b in range(B)])
    
    # 确保右手坐标系（PCA配准原理）
    cross = torch.cross(eigenvectors[:, :, 0], eigenvectors[:, :, 1], dim=1)
    eigenvectors[:, :, 2] = cross / (torch.norm(cross, dim=1, keepdim=True) + 1e-7)
    
    return eigenvectors.transpose(1, 2)  # [B, 3, 3]

def align_to_principal_axis(
    trans: torch.Tensor, 
    rotation: torch.Tensor,
    node_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入：
        trans: [B, N, 3] 位置矩阵
        rotation: [B, N, 3, 3] 原始旋转矩阵
        node_mask: [B, N] 有效点掩码
    输出：
        aligned_trans: [B, N, 3] 标准化后的位置
        aligned_rotation: [B, N, 3, 3] 标准化后的旋转矩阵
    """
    B, N = trans.shape[:2]
    
    # 计算全局旋转矩阵
    global_rot = compute_principal_axis(trans, node_mask)  # [B, 3, 3]
    global_rot_inv = global_rot.transpose(1, 2)  # 逆矩阵
    
    # 应用全局旋转到每个点
    centroid = torch.mean(trans * node_mask.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, 3]
    aligned_trans = torch.einsum('bij,bnj->bni', global_rot_inv, trans - centroid)  # [B, N, 3]
    
    # 更新旋转矩阵（张量旋转方法）
    aligned_rotation = torch.einsum('bnij,bnjk->bnik', rotation, global_rot_inv.unsqueeze(1).expand(B, N, 3, 3))  # [B, N, 3, 3]
    
    return aligned_trans, aligned_rotation

    

class ProteinStructureLoss(nn.Module):
    def __init__(self, 
                 pos_weight: float = 1.0,
                 angle_weight: float = 0.5,
                 torsion_weight: float = 0.3,
                 fape_weight: float = 0.2):
        super().__init__()
        self.pos_weight = pos_weight
        self.angle_weight = angle_weight 
        self.torsion_weight = torsion_weight
        self.fape_weight = fape_weight
        
    def masked_mean(self, tensor, mask):
        """
        掩码加权平均计算
        tensor: [B, L, ...]
        mask:   [B, L]
        """
        valid = mask.sum(dim=-1).clamp(min=1e-6)
        loss = (tensor * mask[..., None]).sum(dim=[-1, -2]) / valid
        return loss.mean()

    def position_loss(self, pred, target, mask):
        """坐标位置损失 (FAPE Loss变体)"""
        delta = pred - target
        dist = torch.norm(delta, dim=-1)  # [B, L]
        return self.masked_mean(dist, mask)
    
    def angle_loss(self, pred, target, mask):
        """Cα间夹角损失 (连续三个残基)"""
        # 计算预测角度
        pred_angles = self.compute_ca_angles(pred)  # [B, L]
        # 计算真实角度
        true_angles = self.compute_ca_angles(target)
        # 周期损失计算
        pred_angles = torch.stack([torch.sin(pred_angles), torch.cos(pred_angles)], dim=-1)  # [B, L, 2]
        true_angles = torch.stack([torch.sin(true_angles), torch.cos(true_angles)], dim=-1)  # [B, L, 2]
        # 计算损失
        loss = (pred_angles - true_angles).pow(2)
        return self.masked_mean(loss, mask[:, 1:-1])
    
    def torsion_loss(self, pred, target, mask):
        """扭转角损失 (四个连续残基)"""
        # 预测/真实扭转角计算
        pred_torsion = self.compute_dihedrals(pred)  # [B, L-3, 2]
        true_torsion = self.compute_dihedrals(target)
        # 周期损失计算
        pred_torsion = torch.stack([torch.sin(pred_torsion), torch.cos(pred_torsion)], dim=-1)  # [B, L-3, 2]
        true_torsion = torch.stack([torch.sin(true_torsion), torch.cos(true_torsion)], dim=-1)
        loss = (pred_torsion - true_torsion).pow(2)  # [B, L-3, 2]
        return self.masked_mean(loss, mask[:, 2:-1])
    
    def compute_ca_angles(self, coords):
        """计算三个连续Cα的夹角"""
        # coords: [B, L, 3]
        v1 = coords[:, :-2] - coords[:, 1:-1]  # BA向量
        v2 = coords[:, 2:] - coords[:, 1:-1]   # BC向量
        cos_theta = torch.sum(v1 * v2, dim=-1) / (
            torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-6
        )
        return torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # [B, L-2]
    
    def compute_dihedrals(self, coords):
        """计算四个连续Cα的二面角 (phi/psi)"""
        # 参考AlphaFold几何编码
        p0, p1, p2, p3 = coords[:, :-3], coords[:, 1:-2], coords[:, 2:-1], coords[:, 3:]
        
        v1 = p1 - p0
        v2 = p2 - p1
        v3 = p3 - p2
        
        # 计算法向量
        n1 = torch.linalg.cross(v1, v2)  # 平面p0-p1-p2的法向量
        n2 = torch.linalg.cross(v2, v3)  # 平面p1-p2-p3的法向量
        
        # 计算夹角
        cos_phi = torch.sum(n1 * n2, dim=-1) / (
            torch.norm(n1, dim=-1) * torch.norm(n2, dim=-1) + 1e-6
        )
        phi = torch.acos(torch.clamp(cos_phi, -1.0, 1.0))
        
        # 判断方向
        sign = torch.sign(torch.sum(torch.linalg.cross(n1, n2) * v2, dim=-1))
        return sign * phi  # [B, L-3]
    
    def dist_and_clash_loss(self, pred, target, mask):
        dist_1 = torch.cdist(pred, pred)
        dist_2 = torch.cdist(target, target)
        dist_mask = mask[:, :, None] * mask[:, None, :]  # [B, L, L]
        dist_loss = (dist_1 - dist_2).pow(2)
        dist_loss = torch.sum(dist_loss * dist_mask, dim=(-1, -2)) / (torch.sum(dist_mask, dim=(-1, -2)) + 1e-8)  # [B]
        
        # clash loss
        mask_clash = (dist_1 < 3.8) & (dist_1 > 2.0)
        clash_loss = torch.where(mask_clash, (3.8 - dist_1).pow(2), 0.0)
        clash_loss = torch.sum(clash_loss * dist_mask, dim=(-1, -2)) / (torch.sum(dist_mask, dim=(-1, -2)) + 1e-8)
        
        return torch.mean(dist_loss), torch.mean(clash_loss)
        
    
    def extract_fea_from_gen(self, fea, gen_mask):
        gen_mask = gen_mask.bool()
        return pad_sequence(
            [fea[i][gen_mask[i]] for i in range(gen_mask.size(0))], 
            batch_first=True, 
            padding_value=0.0
            )
        
    def fape_loss(self, pos_pred, pos_target, mask):
        """
        FAPE损失函数实现
        参数:
            pos_pred:   (B, N, 3) 模型预测的原子坐标
            pos_target: (B, N, 3) 真实原子坐标
            mask:       (B, N)   有效坐标标记(1有效，0无效)
        返回:
            loss: 标量损失值
        """
        # 步骤1：Kabsch对齐预测坐标
        aligned_pred = kabsch_transform(pos_pred, pos_target, mask)  # (B,N,3)
        
        # 步骤2：计算点对点L2距离
        squared_diff = torch.sum((aligned_pred - pos_target), dim=-1)  # (B,N)
        l2_dist = torch.sqrt(squared_diff + 1e-6)  # 数值稳定
        
        # 步骤3：应用mask过滤无效位置[6,8]
        valid_loss = l2_dist * mask  # (B,N)
        
        # 步骤4：计算加权平均损失
        num_valid = torch.sum(mask, dim=(0,1)) + 1e-6
        loss = torch.sum(valid_loss) / num_valid
        
        return loss
        
    
    def forward(self, pred, target, gen_mask):
        """
        Args:
            pred:  预测坐标 [B, L, 3]
            target:真实坐标 [B, L, 3] 
            mask:  有效残基掩码 [B, L]
        Returns:
            total_loss: 综合损失值
        """
        # 主损失项
        pred = self.extract_fea_from_gen(pred, gen_mask)
        target = self.extract_fea_from_gen(target, gen_mask)
        mask = self.extract_fea_from_gen(gen_mask, gen_mask)
        
        
        # pos_loss = self.position_loss(pred, target, mask)
        angle_loss = self.angle_loss(pred, target, mask)
        torsion_loss = self.torsion_loss(pred, target, mask)
        dist_loss, clash_loss = self.dist_and_clash_loss(pred, target, mask)
        # fape_loss = self.fape_loss(pred, target, mask)
        
        # 动态权重设置(参考Distance-AF)
        
        return {
            # 'pos_loss': pos_loss,
            'bb_angle_loss': angle_loss,
            'bb_torsion_loss': torsion_loss,
            'dist_loss': dist_loss,
            'clash_loss': clash_loss,
            # 'fape_loss': fape_loss,
        }
        

def kabsch_transform(pred, target, mask):
    """
    可微分Kabsch对齐实现[8]
    输入: 
        pred:   (B, N, 3) 预测坐标 
        target: (B, N, 3) 真实坐标
        mask:   (B, N)   有效点标记
    返回: 对齐后的预测坐标 (B, N, 3)
    """
    # 用mask过滤无效点
    masked_pred = pred * mask.unsqueeze(-1)  # (B,N,3)
    masked_target = target * mask.unsqueeze(-1)  # (B,N,3)
    
    # 计算质心
    center_pred = masked_pred.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)  # (B,3)
    center_target = masked_target.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)
    
    # 去中心化
    pred_centered = masked_pred - center_pred.unsqueeze(1)  # (B,N,3)
    target_centered = masked_target - center_target.unsqueeze(1)
    
    # 协方差矩阵
    H = torch.einsum('bni,bnj->bij', pred_centered, target_centered)  # (B,3,3)
    
    # SVD分解求旋转矩阵
    U, S, Vh = torch.linalg.svd(H)
    det = torch.det(torch.bmm(U, Vh)).sign().unsqueeze(-1).unsqueeze(-1)
    R = torch.bmm(U * det, Vh)  # (B,3,3)
    
    # 应用旋转平移
    aligned_pred = torch.einsum('bij,bnj->bni', R, pred_centered) + center_target.unsqueeze(1)
    return aligned_pred, R


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