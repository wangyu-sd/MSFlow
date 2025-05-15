import torch, math
import torch.nn as nn
from functools import partial
from typing import Tuple, List
import torch.nn.functional as F

from model.vq import VectorQuantizer
from model.var_basics import AdaLNSelfAttn, AdaLNBeforeHead, SharedAdaLin, AdaLNCrossAttn
from model.var_helpers import sample_with_top_k_top_p
from model.vqpae_layer import VQPAEBlock
from model.vqpae import VQPAE


class PAR(nn.Module):
    def __init__(self,
        vqpae: VQPAE, config, 
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, cond_drop_rate=0.1,
        attn_l2_norm=False, norm_eps=1e-6,
    ):
        super().__init__()
        
        # Hyperparameters
        self.depth = config.par.depth
        self.num_heads = config.par.num_heads
        self.C = self.D = config.par.emb_dim

        self.vqpae = vqpae
        self.vqpae.eval()
        
        self.Cvae = vqpae.vqvae.quantizer.embedding_dim
        self.V = vqpae.vqvae.quantizer.codebook_size

        self.scales = vqpae.vqvae.quantizer.scales
        # self.scales = [2**i for i in range(self.scales+1)]
        self.L = sum(self.scales)                
        self.first_l = self.scales[0]   # Size of the first scale (will be replaced by a class label of the same size during training)
        self.num_scales_minus_1 = len(self.scales) - 1

        self.cond_drop_rate = cond_drop_rate

        # Input (word) embedding
        self.word_embed = nn.Linear(self.Cvae, self.C)
        quant: VectorQuantizer = vqpae.vqvae.quantizer
        quant.update_embedding()
        self.vae_quant_proxy: Tuple[VectorQuantizer] = (quant, )
        self.vae_proxy: Tuple[VQPAEBlock] = (vqpae.vqvae, )
        
        # Class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.poc_emb = nn.Linear(self.vqpae._model_cfg.node_embed_size, self.C)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # Absolute position embedding
        self.pos_1LC = nn.Parameter(torch.empty(1, self.L, self.C))
        nn.init.trunc_normal_(self.pos_1LC.data, mean=0, std=init_std)
        self.lvl_embed = nn.Embedding(len(self.scales), self.C) # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # Backbone
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C))
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)

        # Backbone
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNCrossAttn(
                cond_dim=self.D, block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, 
                num_heads=self.num_heads, mlp_ratio=config.par.mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[block_idx], last_drop_p=0 if block_idx==0 else dpr[block_idx-1], attn_l2_norm=attn_l2_norm,
            )
            for block_idx in range(self.depth)
        ])
        
        # Attention mask used in training
        d = torch.cat([torch.full((pn,), i) for i, pn in enumerate(self.scales)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        # lvl_1L = torch.cat([torch.zeros((1,), dtype=lvl_1L.dtype, device=lvl_1L.device), lvl_1L], dim=0)
        self.register_buffer('lvl_1L', lvl_1L)
        # attn_bias_for_masking = torch.zeros((self.L + 1, self.L + 1), device=d.device, dtype=torch.float)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        # attn_bias_for_masking = torch.cat([torch.ones_like(attn_bias_for_masking), attn_bias_for_masking], dim=0) # SOS token
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # Classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    @ torch.no_grad()
    def get_poc_cond(self, batch):
        poc_mask = torch.logical_and(1-batch['generate_mask'], batch['res_mask'])
        poc_cond = batch['node_embed'] * poc_mask[..., None].float()
        return poc_cond.sum(dim=1) # B, D
    
    @ torch.no_grad()
    def get_batched_fea(self, batch):
        return self.vqpae.extract_fea(batch)
    
    @ torch.no_grad()
    def gt_idx_Bl(self, batch, is_fea=False):
        if is_fea:
            batch_fea = batch
        else:
            batch_fea = self.get_batched_fea(batch)
        gt_idx_Bl, _ = self.vqpae.vqvae.pep_to_idxBl(batch_fea, mode='pep_given_poc')
        return gt_idx_Bl
        
    
    def forward(self, batch):
        
        with torch.no_grad():
            batch_fea = self.get_batched_fea(batch)
            gt_idx_Bl, _ = self.vqpae.vqvae.pep_to_idxBl(batch_fea, mode='pep_given_poc')
            poc_idx_Bl, _ = self.vqpae.vqvae.poc_to_idxBl(batch_fea)
            
            gt_Bl = torch.cat(gt_idx_Bl, dim=1)
            # poc_Bl = torch.cat(poc_idx_Bl, dim=1)
            
            # gt_fea = self.vae_quant_proxy[0].embedding[gt_Bl]   # B, l, Cvae
            # poc_context = self.vae_quant_proxy[0].embedding[poc_Bl] # TODO 
            poc_context = self.vae_quant_proxy[0].idxBl_to_var_context(poc_idx_Bl)
            
            x_BLCv_wo_first_l = self.vae_quant_proxy[0].idxBl_to_var_input(gt_idx_Bl)
            poc_cond = self.get_poc_cond(batch_fea)
            # gt_BL = self.vqpae.vqvae.pep_to_idxBl(batch_fea, mode='pep_given_poc')
            # gt_BL = torch.cat(gt_idx_Bl, dim=1) 
        
        B = x_BLCv_wo_first_l.shape[0]
            
        # SOS token
        poc_cond = self.poc_emb(poc_cond) # B, Cvae
        # poc_cond = poc_cond * (torch.rand((B, 1), device=poc_cond.device) > self.cond_drop_rate)
        sos = poc_cond # B, Cs
        cond_BD = poc_cond
        sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) # B, 1, C => SOS embedding (not token anymore)

        # Whole sequence (SOS + sequence without first l)
        x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1) # B, 1+L-1 (sos+l_wo_frst), C
        
        lvl_emb = self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC 
        
        x_BLC_temp = x_BLC + lvl_emb # Add positional embedding => B, L, C
        x_BLC = x_BLC_temp
        # poc_BLC = torch.cat((sos, self.word_embed(poc_context.float())), dim=1) # B, L, C
        poc_BLC = self.word_embed(poc_context.float()) # B, L, C
        poc_BLC_temp = poc_BLC + lvl_emb # Add positional embedding => B, L, C
        
        poc_BLC = poc_BLC_temp

        # Masking and SharedAdaLN
        attn_bias = self.attn_bias_for_masking # 1, 1, L, L
        cond_BD_or_gss = self.shared_ada_lin(cond_BD) # B, 1, 6, C

        # Backbone
        for block in self.blocks:
            x_BLC = block(x=x_BLC, context=poc_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.head(self.head_nm(x_BLC.float(), cond_BD).float()).float() # codeword for each L => B, L, V
        
        
        return x_BLC, gt_Bl
    
    @ torch.no_grad()
    def anchor_based_infer(self, batch, cfg, top_k, top_p):
        batch_fea = self.get_batched_fea(batch)
        loss = self.vqpae(batch, mode='pep_given_poc')
        print(loss)
        return self.postprocess(self.vae_proxy[0].forward(batch_fea, sampling=True), batch_fea)
    
    @ torch.no_grad()
    def autoregressive_infer_cfg(self, batch, cfg, top_k, top_p):
        
        batch_fea = self.get_batched_fea(batch)
        B = batch_fea['node_embed'].shape[0]
        poc_cond = self.get_poc_cond(batch_fea)
        poc_idx_Bl, _ = self.vqpae.vqvae.poc_to_idxBl(batch_fea)
        # gt_Bl = torch.cat(gt_idx_Bl, dim=1)
        # poc_Bl = torch.cat(poc_idx_Bl, dim=1)
        poc_context = self.vae_quant_proxy[0].idxBl_to_var_context(poc_idx_Bl)
        poc_context = self.word_embed(poc_context.float())
        poc_context =  poc_context + self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC 
        
        sos = self.poc_emb(poc_cond)
        cond_BD = sos

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.scales[-1])
        for block in self.blocks:
            block.cross_attn.q_caching(False)
        
        x_raw = next_token_map.new_zeros((0,))
        idx_Bl_list = []
        
        # x_BLC, gt_Bl = self.forward(batch)
        # loss = F.cross_entropy(x_BLC.view(-1, self.V), gt_Bl.view(-1), reduction='none')
        # loss = loss.mean()
        # print("PAR loss: ", loss.item())
        gt_BL, h_hat_gt1 = self.vqpae.vqvae.pep_to_idxBl(batch_fea, mode='pep_given_poc')
        # _, _, x_BLC_temp, poc_BLC_tem = self.forward(batch)
        
        for si, pn in enumerate(self.scales):
            ratio = si / self.num_scales_minus_1
            cur_L += pn

            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x_raw = torch.cat([x_raw, next_token_map], dim=1)
            x = x_raw
            for block in self.blocks:
                x = block(x=x, context=poc_context, cond_BD=cond_BD_or_gss, attn_bias=self.attn_bias_for_masking[:, :, :x.size(1), :])
            logits_BLV = self.head(self.head_nm(x.float(), cond_BD).float()).float() # codeword for each L => B, L, V
            logits_BLV  = logits_BLV[:, cur_L-pn:cur_L, :] # SOS + L-1 + l_wo_frst
            # t = cfg * ratio
            # logits_BLV = (1+t) * logits_BLV[:B] - t * logits_BLV[B:]

            # 
            if si >= 3:
                idx_Bl = sample_with_top_k_top_p(logits_BLV, rng=None, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
                # idx_Bl = logits_BLV.argmax(dim=-1)
            else:
                idx_Bl = gt_BL[si]
            idx_Bl_list.append(idx_Bl)
            h_BCn = self.vae_quant_proxy[0].embedding[idx_Bl]   # B, l, Cvae
            
            h_BCn = h_BCn.transpose_(1, 2).reshape(B, self.Cvae, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.scales), f_hat, h_BCn)
            if si != self.num_scales_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.scales[si+1]]
                # next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for block in self.blocks:
            block.cross_attn.q_caching(False)
        
        # gt_BL, h_hat_gt1 = self.vqpae.vqvae.pep_to_idxBl(batch_fea, mode='pep_given_poc')
        # gt_BL = torch.cat(gt_BL, dim=1)
        # h_gt_BL = self.vae_quant_proxy[0].embedding[gt_BL].permute(0, 2, 1)# B, l, Cvae
        # f_hat_gt = h_gt_BL.new_zeros(B, self.Cvae, self.scales[-1])
        # pn_start = 0
        # for i, pn in enumerate(self.scales):
        #     f_hat_gt.add_(F.interpolate(h_gt_BL[:, :, pn_start:pn_start+pn], size=(self.scales[-1]), mode='linear'))
        #     pn_start += pn
        

        results = self.vae_proxy[0].fhat_to_graph(f_hat.transpose(1, 2), batch_fea, mode='pep_given_poc')
        return self.postprocess(results, batch_fea)
    
    def postprocess(self, results, batch_fea):
        
        # loss = self.vqpae.get_loss(results, batch_fea, mode='pep')
        # print(loss)
        
        final =  {
            'rotmats': results["pred_rotmats"],
            'seqs': results["pred_seqs"].argmax(dim=-1),
            'angles': results["pred_angles"],
            'trans': results["pred_trans"]
        }
        
        # Fix poc feature
        poc_mask = torch.logical_and(1-batch_fea['generate_mask'], batch_fea['res_mask'])
        final['trans'], _ = self.vqpae.zero_center_part(final['trans'], batch_fea['generate_mask'], batch_fea['res_mask'])
        
        final_ = {}
        for k, v in final.items():
            final[k][poc_mask] = batch_fea[k][poc_mask]
            final_[k] = v
            final_[k+"_gt"] = batch_fea[k]
            
        final_['seqs_gt'] = torch.where(batch_fea['res_mask'].bool(), batch_fea['seqs'], 0)
        final_['seqs'] = torch.where(batch_fea['res_mask'].bool(), final_['seqs'], 0)
        final_['res_mask'] = batch_fea['res_mask']
        final_['generate_mask'] = batch_fea['generate_mask']
        
        return final_
    
    
    
    def forward_ar(self, batch):
        
        with torch.no_grad():
            batch_fea = self.get_batched_fea(batch)
            gt_idx_Bl, _ = self.vqpae.vqvae.pep_to_idxBl(batch_fea, mode='pep_given_poc')
            # poc_idx_Bl = self.vqpae.vqvae.poc_to_idxBl(batch_fea)
            x_BLCv_wo_first_l = self.vae_quant_proxy[0].idxBl_to_var_input(gt_idx_Bl)
            poc_cond = self.get_poc_cond(batch_fea)
            # gt_BL = self.vqpae.vqvae.pep_to_idxBl(batch_fea, mode='pep_given_poc')
            gt_BL = torch.cat(gt_idx_Bl, dim=1) 
        
        B = x_BLCv_wo_first_l.shape[0]
            
        # SOS token
        poc_cond = self.poc_emb(poc_cond) # B, Cvae
        poc_cond = poc_cond * (torch.rand((B, 1), device=poc_cond.device) > self.cond_drop_rate)
        sos = poc_cond # B, Cs
        cond_BD = poc_cond
        sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) # B, 1, C => SOS embedding (not token anymore)

        # Whole sequence (SOS + sequence without first l)
        x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1) # B, 1+L-1 (sos+l_wo_frst), C
        x_BLC += self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC # Add positional embedding => B, L, C

        # Masking and SharedAdaLN
        attn_bias = self.attn_bias_for_masking # 1, 1, L, L
        cond_BD_or_gss = self.shared_ada_lin(cond_BD) # B, 1, 6, C

        # Backbone
        for block in self.blocks:
            x_BLC = block(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.head(self.head_nm(x_BLC.float(), cond_BD).float()).float() # codeword for each L => B, L, V
        
        
        return x_BLC, gt_BL
    