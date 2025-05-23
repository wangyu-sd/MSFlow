# import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


from model.models_con.edge import EdgeEmbedder
from model.models_con.node import NodeEmbedder
from model.encoder_layer import GAEncoder
from model.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms
from model.modules.common.geometry import construct_3d_basis, global_to_local, local_to_global
from torch.nn.utils.rnn import pad_sequence
from model.models_con import torus
from openfold.utils import rigid_utils as ru
from model.modules.so3.dist import centered_gaussian, uniform_so3
from dm import so3_utils
from dm import all_atom
from openfold.np import residue_constants
from model.models_con.torsion import get_torsion_angle_batched
IDEALIZED_POS = torch.tensor(residue_constants.restype_atom14_rigid_group_positions) # K=21, 14, 3
IDEALIZED_POS = F.pad(IDEALIZED_POS, (0, 0, 0, 1), mode='constant', value=0.0)[:-1] # K=21, 15, 3  -> K=20, 15, 3
from tqdm import tqdm
from torch_geometric.nn import radius_graph
# from model.models_con.pep_dataloader import PepDataset

# from model.utils.misc import load_config
# from model.utils.train import recursive_to
# from easydict import EasyDict


from model.models_con.torsion import torsions_mask

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}

class MSFlowMatching(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._model_cfg = cfg.encoder
        self._interpolant_cfg = cfg.interpolant
        self.node_embedder = NodeEmbedder(cfg.encoder.node_embed_size, 3)
        self.edge_embedder = EdgeEmbedder(cfg.encoder.edge_embed_size, 3)
        self.encoder: GAEncoder = GAEncoder(cfg.encoder.ipa)
        self.strc_loss_fn = ProteinStructureLoss()
        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
        # self.node_proj = nn.Linear(cfg.encoder.node_embed_size, cfg.encoder.ipa.c_s)
        # self.edge_proj = nn.Linear(cfg.encoder.edge_embed_size, cfg.encoder.ipa.c_z)
    
    def extract_fea(self, batch):
        with torch.no_grad():
            rotmats_1 = construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N])  
            trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
            seqs_1 = batch['aa']
            context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
            crd = global_to_local(rotmats_1, trans_1, batch['pos_heavyatom']) * batch['mask_heavyatom'][:, :, :, None]
 
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
            "crd": crd,
            "seqs": seqs_1, 
            "node_embed": node_embed, 
            "edge_embed": edge_embed,
            "generate_mask": gen_mask,
            "res_mask": res_mask,
        }
        
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
    
    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k # (B,L,K)
    
    
    def forward(self, batch):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask,angle_mask = batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()

        #encode
        batch_fea = self.extract_fea(batch) # no generate mask
        

        # prepare for denoise
        # trans_1_c,_ = self.zero_center_part(batch_fea["trans_1"],gen_mask,res_mask)
        trans_1_c = batch_fea["trans"] # already centered when constructing dataset
        seqs_1_simplex = self.seq_to_simplex(batch_fea["seqs"])
        seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)
        crd_1 = batch_fea['crd']
        
        
        with torch.no_grad():
            t = torch.rand((num_batch,1), device=batch['aa'].device) 
            t = t*(1-2 * self._interpolant_cfg.min_t) + self._interpolant_cfg.min_t # avoid 0
            if True:
                # corrupt trans
                trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device) * self._interpolant_cfg.trans.sigma # scale with sigma?
                trans_0_c,_ = self.zero_center_part(trans_0,gen_mask,res_mask)
                trans_t = (1-t[...,None])*trans_0_c + t[...,None]*trans_1_c
                trans_t_c = torch.where(batch['generate_mask'][...,None],trans_t,trans_1_c)
                # corrupt rotmats
                rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
                rotmats_t = so3_utils.geodesic_t(t[..., None], batch_fea["rotmats"], rotmats_0)
                rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_t,batch_fea["rotmats"])

            else:
                trans_t_c = trans_1_c.detach().clone()
                rotmats_t = rotmats_1.detach().clone()
                angles_t = angles_1.detach().clone()
                
            if True:
                # corrupt seqs
                seqs_0_simplex = self.k * torch.randn_like(seqs_1_simplex) # (B,L,K)
                seqs_0_prob = F.softmax(seqs_0_simplex,dim=-1) # (B,L,K)
                seqs_t_simplex = ((1 - t[..., None]) * seqs_0_simplex) + (t[..., None] * seqs_1_simplex) # (B,L,K)
                seqs_t_simplex = torch.where(batch['generate_mask'][...,None],seqs_t_simplex,seqs_1_simplex)
                seqs_t_prob = F.softmax(seqs_t_simplex,dim=-1) # (B,L,K)
                seqs_t = sample_from(seqs_t_prob) # (B,L)
                seqs_t = torch.where(batch['generate_mask'],seqs_t,batch_fea["seqs"])
            else:
                seqs_t = seqs_1.detach().clone()
                seqs_t_simplex = seqs_1_simplex.detach().clone()
                seqs_t_prob = seqs_1_prob.detach().clone()
                
            # corrup crd
            # B, L, K @ K, 15, 3 -> B, L, 15, 3
            crd_0 = torch.einsum('bik,kjc->bijc', seqs_0_prob, IDEALIZED_POS.to(device=seqs_0_prob.device)) # (B,L,15,3)
            crd_0 = crd_0 + torch.randn_like(crd_1) * 1e-3 
            crd_t = (1-t[...,None, None]) * crd_0 + t[...,None, None] * crd_1
            crd_t = torch.where(batch['generate_mask'][...,None, None],crd_t, crd_1)
            
        batch_fea_t = {
            "t": t,
            "rotmats_t": rotmats_t,
            "trans_t": trans_t,
            "crd_t": crd_t,
            "seqs_t": seqs_t
        }
        batch_fea.update(batch_fea_t)
        res = self.encoder(batch_fea)
        
        
        loss = self.get_loss(res, batch_fea)
        
        return None, None, loss
        
    
    def get_loss(self, res, fea_dict, weigeht=1.):
        pred_trans, pred_rotmats, pred_crd, pred_seqs_1_prob = \
            res['pred_trans'], res['pred_rotmats'], res['pred_crd'], res['pred_seqs']
        
        trans_t, rotmats_t, crd_t, seqs_t = \
            fea_dict['trans_t'], fea_dict['rotmats_t'], fea_dict['crd_t'], fea_dict['seqs_t']
        gen_mask, res_mask = fea_dict['generate_mask'], fea_dict['res_mask']
        
        trans_1, rotmats_1, crd_1, seqs_1 = \
            fea_dict['trans'], fea_dict['rotmats'], fea_dict['crd'], fea_dict['seqs']
        
        t = fea_dict['t']
        # denoise
        pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
        pred_seqs_1 = torch.where(fea_dict['generate_mask'].bool(), pred_seqs_1, torch.clamp(seqs_1,0,19))
        # pred_trans_1_c, _ = self.zero_center_part(pred_trans, gen_mask, res_mask)
        trans_1_c, _ =  self.zero_center_part(trans_1, gen_mask, res_mask)
        pred_trans_1_c = pred_trans # implicitly enforce zero center in gen_mask, in this way, we dont need to move receptor when sampling

        norm_scale = 1 / (1 - torch.min(t[...,None], torch.tensor(self._interpolant_cfg.t_normalization_clip))) # yim etal.trick, 1/1-t

        # trans vf loss
        trans_loss = torch.sum((pred_trans_1_c - trans_1_c)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        trans_loss = torch.mean(trans_loss)
        
        # crd_loss
        B, L = pred_crd.shape[:2]
        crd_loss = torch.sum((pred_crd.view(B, L, -1) - crd_1.reshape(B, L, -1))**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        crd_loss = torch.mean(crd_loss)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pred_crd_dist = torch.cdist(pred_crd.view(B*L, -1, 3), pred_crd.view(B*L, -1, 3), p=2) # (BL, A, A)
            gt_crd_dist = torch.cdist(crd_1.view(B*L, -1, 3), crd_1.view(B*L, -1, 3), p=2) # (BL,A, A)
            crd_dist_loss = (pred_crd_dist - gt_crd_dist).pow(2).mean(dim=[1, 2]).view(B, L) * 3
            crd_dist_loss = torch.sum(crd_dist_loss * gen_mask, dim=-1) / (torch.sum(gen_mask, dim=-1) + 1e-8) # (B,)
            crd_dist_loss = torch.mean(crd_dist_loss)
            
        crd_dist_loss = crd_dist_loss.float()

        # aux loss
        pred_trans_gen = self.strc_loss_fn.extract_fea_from_gen(pred_trans_1_c, gen_mask)
        trans_gen = self.strc_loss_fn.extract_fea_from_gen(trans_1, gen_mask)
        gen_mask_sm = self.strc_loss_fn.extract_fea_from_gen(gen_mask, gen_mask)
        
        poc_mask = torch.logical_and(~gen_mask.bool(), res_mask.bool())
        trans_pos = self.strc_loss_fn.extract_fea_from_gen(trans_1, poc_mask)
        poc_mask = self.strc_loss_fn.extract_fea_from_gen(poc_mask, poc_mask)
        strc_loss = self.strc_loss_fn(pred_trans_gen, trans_gen, gen_mask_sm, trans_pos, poc_mask)
        
        # seqs loss
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats)
        rot_loss = torch.sum(((gt_rot_vf - pred_rot_vf) * norm_scale)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        rot_loss = torch.mean(rot_loss)
        

        # bb aux loss
        # pred_rotamats_gen, rotamats_gen = self.strc_loss_fn.extract_fea_from_gen(pred_rotmats, gen_mask), self.strc_loss_fn.extract_fea_from_gen(rotmats_1, gen_mask)
        gt_bb_atoms = all_atom.to_atom37(trans_1_c, rotmats_1)[:, :, :3] 
        pred_bb_atoms = all_atom.to_atom37(pred_trans, pred_rotmats)[:, :, :3]
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        bb_atom_loss = torch.mean(bb_atom_loss)
        
        pred_trans = torch.where(gen_mask[..., None].bool(), pred_trans, trans_1_c)
        pred_rotmats = torch.where(gen_mask[..., None, None].bool(), pred_rotmats, rotmats_1)
        pred_crd_global = local_to_global(pred_rotmats, pred_trans, pred_crd) # (B, L, 15, 3)
        
        clear_loss = clearance_loss(pred_crd_global, gen_mask, safe_threshold=3.0, buffer=0.6, alpha=2.2)
        
        # crd_global = local_to_global(rotmats_1, trans_1_c, crd_1)
        
        
        
        # pred_idg = (pred_crd_global[:, :-1] - pred_crd_global[:, 1:]).pow(2).sum(dim=[-1, -2]) # (B, L-1)
        # gt_idg = (crd_global[:, :-1] - crd_global[:, 1:]).pow(2).sum(dim=[-1, -2]) # (B, L-1)
        # idg_loss = torch.sum((pred_idg - gt_idg) ** 2 * gen_mask[:, 1:, None], dim=(-1, -2)) / (torch.sum(gen_mask[:, 1:], dim=-1) + 1e-8) # (B,)
        # idg_loss = torch.mean(idg_loss)
        
        
        # Angle Loss
        angle_1_pred, _ = get_torsion_angle_batched(pred_crd.view(B, L, 15, 3), pred_seqs_1_prob.argmax(dim=-1))
        angle_1_gt, _ = get_torsion_angle_batched(crd_1.view(B, L, 15, 3), seqs_1)
        
        angle_mask_loss = gen_mask[...,None].bool()
        
        gt_angle_vec = torch.cat([torch.sin(angle_1_gt),torch.cos(angle_1_gt)],dim=-1)
        pred_angle_vec = torch.cat([torch.sin(angle_1_pred),torch.cos(angle_1_pred)],dim=-1)
        # angle_loss = torch.sum(((gt_angle_vf_vec - pred_angle_vf_vec) * norm_scale)**2*gen_mask[...,None],dim=(-1,-2)) / ((torch.sum(gen_mask,dim=-1)) + 1e-8) # (B,)
        angle_loss = torch.sum(((gt_angle_vec - pred_angle_vec))**2*angle_mask_loss,dim=(-1,-2)) / (torch.sum(angle_mask_loss,dim=(-1,-2)) + 1e-8) # (B,)
        angle_loss = torch.mean(angle_loss)
        
        
        # seqs vf loss
        seqs_loss = F.cross_entropy(pred_seqs_1_prob.view(-1,pred_seqs_1_prob.shape[-1]),torch.clamp(seqs_1,0,19).view(-1), reduction='none').view(pred_seqs_1_prob.shape[:-1]) # (N,L), not softmax
        seqs_loss = torch.sum(seqs_loss * gen_mask, dim=-1) / (torch.sum(gen_mask,dim=-1) + 1e-8)
        seqs_loss = torch.mean(seqs_loss)
        
        
        
        
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
            "inter_dist_loss": strc_loss['inter_dist_loss'] * weigeht,
            "crd_loss": crd_loss * weigeht,
            "clear_loss": clear_loss * weigeht,
            # "idg_loss": idg_loss * weigeht,
            "crd_dist_loss": crd_dist_loss * weigeht,
        }
        
        for key in res.keys():
            if "loss" in key:
                res_[key] = res[key] * weigeht
        return res_
    
    
    
    @torch.no_grad()
    def sample(self, batch, num_steps = 100, sample_bb=True, sample_crd=True, sample_seq=True):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask = batch['generate_mask'],batch['res_mask']
        K = self._interpolant_cfg.seqs.num_classes
        k = self._interpolant_cfg.seqs.simplex_value
        # angle_mask_loss = torsions_mask.to(batch['aa'].device)
        angle_mask_loss = batch['generate_mask'][..., None]

        #encode
        batch_fea = self.extract_fea(batch)
        rotmats_1, trans_1, crd_1, seqs_1, node_embed, edge_embed = \
            batch_fea["rotmats"], batch_fea['trans'], batch_fea["crd"], batch_fea["seqs"], \
            batch_fea["node_embed"], batch_fea["edge_embed"]
        
        trans_1_c = trans_1
        seqs_1_simplex = self.seq_to_simplex(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)


        #initial noise
        if sample_bb:
            rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
            rotmats_0 = torch.where(batch['generate_mask'][...,None,None],rotmats_0,rotmats_1)
            trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device) # scale with sigma?
            # move center and receptor
            trans_0_c,center = self.zero_center_part(trans_0,gen_mask,res_mask)
            trans_0_c = torch.where(batch['generate_mask'][...,None],trans_0_c,trans_1_c)
        else:
            rotmats_0 = rotmats_1.detach().clone()
            trans_0_c = trans_1_c.detach().clone()
        
        if sample_seq:
            seqs_0_simplex = k * torch.randn((num_batch,num_res,K), device=batch['aa'].device)
            seqs_0_prob = F.softmax(seqs_0_simplex,dim=-1)
            seqs_0 = sample_from(seqs_0_prob)
            seqs_0 = torch.where(batch['generate_mask'],seqs_0,seqs_1)
            seqs_0_simplex = torch.where(batch['generate_mask'][...,None],seqs_0_simplex,seqs_1_simplex)
        else:
            seqs_0 = seqs_1.detach().clone()
            seqs_0_prob = seqs_1_prob.detach().clone()
            seqs_0_simplex = seqs_1_simplex.detach().clone()
        # Sample Local Coordinates
        
        crd_0 = torch.einsum('bik,kjc->bijc', seqs_0_prob, IDEALIZED_POS.to(device=seqs_0_prob.device)) # (B,L,15,3)
        

        # Set-up time
        ts = torch.linspace(1.e-2, 1.0, num_steps)
        t_1 = ts[0]
        # prot_traj = [{'rotmats':rotmats_0,'trans':trans_0_c,'seqs':seqs_0,'seqs_simplex':seqs_0_simplex,'rotmats_1':rotmats_1,'trans_1':trans_1-center,'seqs_1':seqs_1}]
        clean_traj = []
        rotmats_t_1, trans_t_1_c, crd_t_1, seqs_t_1, seqs_t_1_simplex = rotmats_0, trans_0_c, crd_0, seqs_0, seqs_0_simplex

        # denoise loop
        for t_2 in tqdm(ts[1:], desc='Denoise', leave=False):
            t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
            # rots
            
            batch_fea_t = {
            "t": t,
            "rotmats_t": rotmats_t_1,
            "trans_t": trans_t_1_c,
            "crd_t": crd_t_1,
            "seqs_t": seqs_t_1
            }
            batch_fea.update(batch_fea_t)
            res = self.encoder(batch_fea)
            
            pred_rotmats_1, pred_trans_1, pred_crd_1, pred_seqs_1_prob = \
                    res['pred_rotmats'], res['pred_trans'], res['pred_crd'], res['pred_seqs']
                    
            pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
            # trans, move center
            # pred_trans_1_c,center = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
            pred_trans_1_c = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_1_c) # move receptor also
            # angles
            pred_crd_1 = torch.where(batch['generate_mask'][...,None, None],pred_crd_1, crd_1)
            # seqs
            pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
            pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
            pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
            # seq-angle
            # torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            # pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
            if not sample_bb:
                pred_trans_1_c = trans_1_c.detach().clone()
                # _,center = self.zero_center_part(trans_1,gen_mask,res_mask)
                pred_rotmats_1 = rotmats_1.detach().clone()
            if not sample_seq:
                pred_seqs_1 = seqs_1.detach().clone()
                pred_seqs_1_simplex = seqs_1_simplex.detach().clone()
                
            pred_crd_1 = crd_1.detach().clone()
            clean_traj.append({'rotmats':pred_rotmats_1.cpu(),'trans':pred_trans_1_c.cpu(),'crd':pred_crd_1.cpu(),'seqs':pred_seqs_1.cpu(),'seqs_simplex':pred_seqs_1_simplex.cpu(),
                                    'rotmats_1':rotmats_1.cpu(),'trans_1':trans_1_c.cpu(),'angles_1':crd_1.cpu(),'seqs_1':seqs_1.cpu()})
            # reverse step, also only for gen mask region
            d_t = (t_2-t_1) * torch.ones((num_batch, 1), device=batch['aa'].device)
            # Euler step
            trans_t_2 = trans_t_1_c + (pred_trans_1_c-trans_0_c)*d_t[...,None]
            # trans_t_2_c,center = self.zero_center_part(trans_t_2,gen_mask,res_mask)
            trans_t_2_c = torch.where(batch['generate_mask'][...,None],trans_t_2,trans_1_c) # move receptor also
            # rotmats_t_2 = so3_utils.geodesic_t(d_t[...,None] / (1-t[...,None]), pred_rotmats_1, rotmats_t_1)
            rotmats_t_2 = so3_utils.geodesic_t(d_t[...,None] * 10, pred_rotmats_1, rotmats_t_1)
            rotmats_t_2 = torch.where(batch['generate_mask'][...,None,None],rotmats_t_2,rotmats_1)
            # angles
            # angles_t_2 = torus.tor_geodesic_t(d_t[...,None],pred_angles_1, angles_t_1)
            # angles_t_2 = torch.where(batch['generate_mask'][...,None],angles_t_2,angles_1)
            crd_t_2 = crd_t_1 + (pred_crd_1 - crd_0) * d_t[..., None, None]
            crd_t_2 = torch.where(batch['generate_mask'][..., None, None],crd_t_2,crd_1)
            
            # seqs
            seqs_t_2_simplex = seqs_t_1_simplex + (pred_seqs_1_simplex - seqs_0_simplex) * d_t[...,None]
            seqs_t_2 = sample_from(F.softmax(seqs_t_2_simplex,dim=-1))
            seqs_t_2 = torch.where(batch['generate_mask'],seqs_t_2,seqs_1)

            
            if not sample_bb:
                trans_t_2_c = trans_1_c.detach().clone()
                rotmats_t_2 = rotmats_1.detach().clone()

            if not sample_seq:
                seqs_t_2 = seqs_1.detach().clone()
            rotmats_t_1, trans_t_1_c, crd_t_1, seqs_t_1, seqs_t_1_simplex = rotmats_t_2, trans_t_2_c, crd_t_2, seqs_t_2, seqs_t_2_simplex
            t_1 = t_2

        # final step
        batch_fea_t = {
            "t": t,
            "rotmats_t": rotmats_t_1,
            "trans_t": trans_t_1_c,
            "crd_t": crd_t_1,
            "seqs_t": seqs_t_1
            }
        batch_fea.update(batch_fea_t)
        res = self.encoder(batch_fea)
            
        pred_rotmats_1, pred_trans_1, pred_crd_1, pred_seqs_1_prob = \
                    res['pred_rotmats'], res['pred_trans'], res['pred_crd'], res['pred_seqs']
        pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
        # move center
        # pred_trans_1_c,center = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
        pred_trans_1_c = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_1_c) # move receptor also
        # angles
        pred_crd_1 = torch.where(batch['generate_mask'][...,None, None], pred_crd_1, crd_1)
        # seqs
        pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
        pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
        pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)

        if not sample_bb:
            pred_trans_1_c = trans_1_c.detach().clone()
            # _,center = self.zero_center_part(trans_1,gen_mask,res_mask)
            pred_rotmats_1 = rotmats_1.detach().clone()
        if not sample_crd:
            pred_crd_1 = crd_1.detach().clone()
        if not sample_seq:
            pred_seqs_1 = seqs_1.detach().clone()
            pred_seqs_1_simplex = seqs_1_simplex.detach().clone()
        clean_traj.append({'rotmats':pred_rotmats_1.cpu(),'trans':pred_trans_1_c.cpu(),'crd':pred_crd_1.cpu(),'seqs':pred_seqs_1.cpu(),'seqs_simplex':pred_seqs_1_simplex.cpu(),
                                'rotmats_1':rotmats_1.cpu(),'trans_1':trans_1_c.cpu(),'crd_1':crd_1.cpu(),'seqs_1':seqs_1.cpu()})
        
        return clean_traj
        


def clearance_loss(pred_crd, crd_mask, safe_threshold=3.0, buffer=0.6, alpha=2.2):
    """
    改进版原子距离约束损失函数
    pred_crd: 预测原子坐标 [B, L, 15, 3]
    crd_mask: 原子有效性掩码 [B, L]
    safe_threshold: 动态安全阈值
    buffer: 缓冲区宽度
    alpha: 渐进惩罚系数
    """
    B, L, N, _ = pred_crd.shape
    
    # 原子掩码处理（
    valid_mask = crd_mask.unsqueeze(-1).expand(-1, -1, N).reshape(B*L*N).bool()  # [B*L*N]
    flat_crd = pred_crd.reshape(B*L*N, 3)[valid_mask]  # 过滤无效原子 [V,3]
    batch_idx = torch.arange(B, device=pred_crd.device).repeat_interleave(L*N)[valid_mask]

    # 邻域搜索优化
    edge_index = radius_graph(flat_crd, 
                            r=safe_threshold + buffer*2,  # 扩展搜索半径
                            batch=batch_idx,
                            loop=False,
                            max_num_neighbors=32)
    
    # 有效距离计算
    src, dst = edge_index
    pred_dist = torch.sqrt(
        (flat_crd[src] - flat_crd[dst]).pow(2).sum(dim=1) + 1e-6)  
    
    # 动态阈值调整
    delta = (safe_threshold - 0.1 * torch.sigmoid(pred_dist.detach())) - pred_dist
    delta = (safe_threshold - pred_dist).clamp(min=-2.0, max=5.0) 
    
    # 分段惩罚机制
    violation_mask = delta > 0  # 实际违规区域
    buffer_zone = (pred_dist > (safe_threshold - buffer)) & violation_mask
    severe_zone = ~buffer_zone & violation_mask

    # 梯度稳定处理
    with torch.no_grad():
        buffer_coef = (safe_threshold - pred_dist) / buffer
    
    # 惩罚项计算
    buffer_penalty = torch.where(buffer_zone, 
                               buffer_coef * delta**2, 
                               torch.zeros_like(delta))
    severe_penalty = torch.where(severe_zone, 
        torch.where(delta > 1.0, delta, 0.5*delta**2), 0.0)
    
    # 对称性归一化
    total_loss = (buffer_penalty + severe_penalty).sum() * 0.5  # 消除双向边重复
    valid_pairs = edge_index.shape[1] // 2  # 有效原子对数
    
    return total_loss / (max(valid_pairs, 1))  # 防止除零

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



def clampped_one_hot(x, num_classes):
    mask = (x >= 0) & (x < num_classes) # (N, L)
    x = x.clamp(min=0, max=num_classes-1)
    y = F.one_hot(x, num_classes) * mask[...,None]  # (N, L, C)
    return y


def sample_from(c):
    """sample from c"""
    N,L,K = c.size()
    c = c.view(N*L,K) + 1e-8
    x = torch.multinomial(c,1).view(N,L)
    return x
    

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
        
    
    def forward(self, pred, target, gen_mask, poc=None, poc_mask=None):
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
        
        if poc is not None:
            pred_inter_dist = torch.cdist(pred, poc, p=2)  # [B, L_ligand, L_poc]
            gt_inter_dist = torch.cdist(target, poc, p=2)  # [B, L_ligand, L_poc]
            dist_mask = gen_mask[:, :, None] * poc_mask[:, None, :]  # [B, L_ligand, L_poc]
            dist_loss_inter = (pred_inter_dist - gt_inter_dist).pow(2)
            dist_loss_inter = torch.sum(dist_loss * dist_mask, dim=(-1, -2)) / (torch.sum(dist_mask, dim=(-1, -2)) + 1e-8)  # [B]
            dist_loss_inter = torch.mean(dist_loss)
            
        
        # 动态权重设置(参考Distance-AF)
        
        return {
            # 'pos_loss': pos_loss,
            'bb_angle_loss': angle_loss,
            'bb_torsion_loss': torsion_loss,
            'dist_loss': dist_loss,
            'inter_dist_loss': dist_loss_inter,
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