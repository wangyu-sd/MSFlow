import math
import torch
from torch import nn
import torch.nn.functional as F

from model.models_con import ipa_pytorch as ipa_pytorch
from dm import utils as du


from typing import Tuple, List, Dict
from model.modules.common.layers import AngularEncoding
from model.vq import VectorQuantizer
from dm import so3_utils




class VQPAEBlock(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf
       
        self.angles_embedder = AngularEncoding(num_funcs=12)
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            # nn.Linear(self._ipa_conf.c_s, 21)
            nn.Linear(self._ipa_conf.c_s, 22)
        )
        
        self.str_fea_fusion = FeaFusionLayer(ipa_conf)
        
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(2 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5), self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )
        self.feat_dim = self._ipa_conf.c_s

        # 主干拆分
        self.encoder_trunk = nn.ModuleDict()
        self.decoder_trunk = nn.ModuleDict()
        self.num_encoder_blocks = self._ipa_conf.num_encoder_blocks
        self.num_decoder_blocks = self._ipa_conf.num_decoder_blocks
        self.scales = self._ipa_conf.scales
        # 编码器主干构建
        for b in range(self.num_encoder_blocks):
            self._build_block(b, is_encoder=True)
            
        # 向量量化层
        self.quantizer: VectorQuantizer = VectorQuantizer(
            codebook_size=ipa_conf.codebook_size,
            embedding_dim=self._ipa_conf.c_s,   
            commitment_cost=ipa_conf.commitment_cost,
            init_steps=ipa_conf.init_steps,
            collect_desired_size=ipa_conf.collect_desired_size,
            scales=ipa_conf.scales,
        )
        
        # 解码器主干构建
        for b in range(self.num_decoder_blocks):
            self._build_block(b, is_encoder=False)
            
        self.var_prj = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), 
            nn.LayerNorm(self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s)
        )
        self.mu_prj = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.LayerNorm(self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s)
        )
        # self.decoder_init_rigid = nn.Sequential(
        #     ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s),
        #     nn.Linear(self._ipa_conf.c_s, 3),)

    def _build_block(self, b, is_encoder):

        trunk = self.encoder_trunk if is_encoder else self.decoder_trunk
        
        trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
        trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
        tfmr_in = self._ipa_conf.c_s
        num_blocks = self.num_encoder_blocks if is_encoder else self.num_decoder_blocks
        tfmr_layer = torch.nn.TransformerEncoderLayer(
            d_model=tfmr_in,
            nhead=self._ipa_conf.seq_tfmr_num_heads,
            dim_feedforward=self._ipa_conf.c_s,
            batch_first=True,
            norm_first=False,
            dropout=0.0,
        )
        # trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
        #     tfmr_layer, self._ipa_conf.seq_tfmr_num_layers)
        
        trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
            tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
        trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
            tfmr_in, self._ipa_conf.c_s, init="final")
        trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
            c=self._ipa_conf.c_s)
        
        trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
            self._ipa_conf.c_s, use_rot_updates=True)
        trunk[f'fea_fusion_{b}'] = FeaFusionLayer(self._ipa_conf)


        if b < num_blocks-1:
            # No edge update on the last block.
            edge_in = self._ipa_conf.c_z
            trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                node_embed_size=self._ipa_conf.c_s,
                edge_embed_in=edge_in,
                edge_embed_out=self._ipa_conf.c_z,
            )
            

    def _process_trunk(self, trunk_type, node_embed, edge_embed, curr_rigids, node_mask, edge_mask):
        """通用主干处理流程"""
        x = node_embed
        
        trunk = self.encoder_trunk if trunk_type == 'encoder' else self.decoder_trunk
        
        # prefix = 'enc_' if trunk_type == 'encoder' else 'dec_'
        num_blocks = self.num_encoder_blocks if trunk_type == 'encoder' else self.num_decoder_blocks
        
        
        for _ in range(1):
            for b in range(num_blocks):
                ipa_embed = trunk[f'ipa_{b}'](
                    node_embed, edge_embed, curr_rigids, node_mask)
                ipa_embed *= node_mask[..., None]
                node_embed = trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
                seq_tfmr_out = trunk[f'seq_tfmr_{b}'](
                    node_embed, src_key_padding_mask=(1 - node_mask).bool())
                node_embed = node_embed + trunk[f'post_tfmr_{b}'](seq_tfmr_out)
                node_embed = trunk[f'node_transition_{b}'](node_embed)
                node_embed = node_embed * node_mask[..., None]
                
                # if trunk_type == 'decoder' or b < num_blocks-1:
                rigid_update = trunk[f'bb_update_{b}'](
                    node_embed * node_mask[..., None])
                curr_rigids = curr_rigids.compose_q_update_vec(
                    rigid_update, node_mask[..., None])
                
                rot = curr_rigids.get_rots().get_rot_mats()
                trans = curr_rigids.get_trans()
                node_embed = trunk[f'fea_fusion_{b}'](
                    node_embed, rot, trans, node_mask)
                

                if b < num_blocks-1:
                    edge_embed = trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed)
                    edge_embed *= edge_mask[..., None]

                
                node_embed = x + node_embed
                
        return node_embed, curr_rigids
    
    
    def encoder_step(self, batch, mode):
        
        if mode == "poc_and_pep" or mode == "pep_given_poc" or mode=='codebook':
            node_mask = batch["res_mask"]
        elif mode == "poc_only":
            node_mask = torch.logical_and(batch["res_mask"], 1-batch["generate_mask"])
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        
        rotmats = batch['rotmats'] * node_mask[..., None, None]
        trans = batch['trans'] * node_mask[..., None]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = batch['node_embed'] * node_mask[..., None]
        x = node_embed
        curr_rigids = du.create_rigid(rotmats, trans)
        
        node_embed, rigids = self._process_trunk(
            'encoder', node_embed, batch["edge_embed"], curr_rigids, node_mask, edge_mask)
        
        # rotmats = rigids.get_rots().get_rot_mats()
        # trans = rigids.get_trans()
        # rotmats = so3_utils.rotmat_to_rotvec(rotmats)
        
        # node_embed = torch.cat([node_embed, rotmats, trans], dim=-1)
        
        # rotmats = so3_utils.rotvec_to_rotmat(node_embed[..., -6:-3])
        # curr_rigids = du.create_rigid(rotmats, node_embed[..., -3:])
        
        mu = self.mu_prj(node_embed)
        mu = mu * node_mask[..., None]
        mu = x + mu
        
        logvar = self.var_prj(node_embed)
        logvar = logvar * node_mask[..., None]
        logvar = x + logvar
        
        return mu, node_mask, edge_mask, curr_rigids, logvar
    
    def decoder_step(self, node_embed, node_emb_raw, edge_embed, curr_rigids, node_mask, 
                     edge_mask, res_mask, generate_mask,  mode):
        
        need_poc = mode == "pep_given_poc"
        poc_mask = torch.logical_and(node_mask, 1-generate_mask)
        if need_poc:
            ## Fix the Pocket Features
            node_embed[poc_mask] += node_emb_raw[poc_mask]
        
        curr_rigids = self.contex_filter(curr_rigids, poc_mask, res_mask, generate_mask, need_poc=need_poc, hidden_str=node_embed)
        # node_embed = node_embed[..., :-6] * node_mask[..., None]
        node_embed = node_embed * node_mask[..., None]
        
        
        
        node_embed, curr_rigids = self._process_trunk(
            'decoder', node_embed, edge_embed, curr_rigids, node_mask, edge_mask)
        
        # 输出预测
        pred_seqs = self.seq_net(node_embed)
        pred_angles = self.angle_net(node_embed)
        pred_angles = pred_angles % (2*math.pi) 
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        
        # if mode == 'pep_given_poc':
        #     final_mask = generate_mask
        # else:
        #     final_mask = node_mask
            
        
        return {
            'pred_rotmats': pred_rotmats,
            'pred_seqs': pred_seqs,
            'pred_angles': pred_angles,
            'pred_trans': pred_trans,
        }
        
    def before_quntized(self, node_embed, gen_mask):
        gen_mask = gen_mask.bool()
        nodes_list = [node_embed[i][gen_mask[i]] for i in range(node_embed.shape[0])]
        size = 2 ** self.scales
        to_sizes = [size for _ in range(node_embed.shape[0])]
        
        quantized = self.interpolate_batch(nodes=nodes_list, to_sizes=to_sizes)
        
        return quantized
    
    def after_quntized(self, quantized, gen_mask):
        nodes = torch.split(quantized, 1)
        original_sizes = gen_mask.sum(1)
        quantized = self.interpolate_batch(
            nodes=nodes, to_sizes=original_sizes, padding_size=gen_mask.size(1), gen_mask=gen_mask.bool(),
        ) # B, max_nodes, C
        return quantized
    
    
    def fea_fusion(self, batch, node_mask=None):
        if node_mask is None:
            node_mask = batch['res_mask']
        
        num_batch, num_res = batch["seqs"].shape
        angles = batch["angles"] * node_mask[..., None]
        seqs = torch.where(node_mask.bool(), batch["seqs"], 21)
        
        node_embed = self.res_feat_mixer(torch.cat([
            batch['node_embed'], 
            self.current_seq_embedder(seqs),
            self.angles_embedder(angles).reshape(num_batch,num_res,-1)
            ],dim=-1))
        
        node_embed = self.str_fea_fusion(
            node_embed, 
            batch['rotmats'], batch['trans'], 
            node_mask,
            )
        return node_embed
        
    def forward(self, batch:Dict[str, torch.Tensor], mode="poc_and_pep"):
        """
            batch contains: rotmats, trans, angles, seqs, node_embed, 
                edge_embed, generate_mask, res_mask
        Returns:
            _type_: _description_
        """
        node_emb_raw = batch['node_embed']
        node_embed, node_mask, edge_mask, curr_rigids, _ = self.encoder_step(batch, mode)
        quantized = self.before_quntized(node_embed, gen_mask=batch['generate_mask']) # TODO Add more choices for gen_mask
        
        quantized, commitment_loss, q_latent_loss = self.quantizer(quantized)
        
        quantized = self.after_quntized(quantized, gen_mask=batch['generate_mask'])
        
        res = self.decoder_step(
            node_embed=quantized, node_emb_raw=node_emb_raw, edge_embed=batch['edge_embed'],
            curr_rigids=curr_rigids, node_mask=node_mask, edge_mask=edge_mask,
            res_mask=batch['res_mask'], generate_mask=batch['generate_mask'],
            mode=mode,
        )
        res['commitment_loss'] = commitment_loss
        res['q_latent_loss'] = q_latent_loss

        return res

    
    
    def forward_init(self, batch:Dict[str, torch.Tensor], mode):
        
        node_embed, node_mask, edge_mask, curr_rigids, logvar = self.encoder_step(
            batch, mode=mode
            )
        interpolated_nodes = self.before_quntized(node_embed, batch['generate_mask'])
        # First stage: VAE-only latent training, no quantization
        if self.quantizer.init_steps > 0:
            self.quantizer.init_steps -= 1        
        
        # Secons stage: collect latent to initialise codebook words with k++ means, no quantization
        elif self.quantizer.collect_phase:
            self.quantizer.collect_samples(interpolated_nodes.detach())
        
        node, _, _ = self.quantizer(interpolated_nodes, vae_stage=True)
        origin_node = self.after_quntized(node, batch['generate_mask'])
        logvar = self.after_quntized(self.before_quntized(logvar, batch['generate_mask']), batch['generate_mask'])
        mu = origin_node
        node_embed = reparameterize(mu, logvar) * batch['generate_mask'][..., None]

        res = self.decoder_step(
            node_embed, batch['node_embed'], edge_embed=batch['edge_embed'],
            curr_rigids=curr_rigids, node_mask=node_mask, edge_mask=edge_mask,
            res_mask=batch['res_mask'], generate_mask=batch['generate_mask'],
            mode=mode
        )
        res['vae_loss'] = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        res['vae_loss'] = res['vae_loss'] * batch['generate_mask'][..., None]
        res['vae_loss'] = res['vae_loss'].sum(dim=[1, 2]) /  batch['generate_mask'].sum(dim=[1])
        res['vae_loss'] = res['vae_loss'].mean() * 0.01
        
        return res
    
    def vae_sample(self, batch:Dict[str, torch.Tensor], mode):
        node_embed, node_mask, edge_mask, curr_rigids, logvar = self.encoder_step(
            batch, mode=mode
        )
        
        # logvar = self.var_prj(node_embed) * node_mask[..., None]
        mu = node_embed
        node_embed = reparameterize(mu, logvar) * node_mask[..., None]
        
        res = self.decoder_step(
            node_embed, batch['node_embed'], edge_embed=batch['edge_embed'],
            curr_rigids=curr_rigids, node_mask=node_mask, edge_mask=edge_mask,
            res_mask=batch['res_mask'], generate_mask=batch['generate_mask'],
            mode=mode
        )
        res['vae_loss'] = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        res['vae_loss'] = res['vae_loss'] * node_mask[..., None]
        res['vae_loss'] = res['vae_loss'].sum(dim=[1, 2]) / node_mask.sum(dim=[1])
        res['vae_loss'] = res['vae_loss'].mean()
        return res
        
    
    def graph_to_idxBl(self, batch, mode) -> List:
        node_embed, node_mask, edge_mask, curr_rigids, logvar = self.encoder_step(
            batch, mode=mode
        )
        interpolated_nodes = self.before_quntized(node_embed, gen_mask=batch['generate_mask'])

        return self.quantizer.f_to_idxBl(interpolated_nodes)
    
    def fhat_to_graph(self, f_hat, batch, mode):
        node_mask = batch['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        curr_rigids = du.create_rigid(batch["rotmats"], batch["trans"])
        
        quantized = self.after_quntized(f_hat, gen_mask=batch['generate_mask'])
        
        res = self.decoder_step(
            node_embed=quantized, node_emb_raw=batch['node_embed'], edge_embed=batch['edge_embed'],
            curr_rigids=curr_rigids, node_mask=node_mask, edge_mask=edge_mask,
            res_mask=batch['res_mask'], generate_mask=batch['generate_mask'],
            mode=mode,
        )

        return res


    
    def interpolate_batch(self, nodes: Tuple[torch.Tensor] | torch.Tensor, to_sizes: List[int] | int, padding_size: int = None, gen_mask: torch.Tensor = None):
        '''
        Interpolate all graphs to desired sizes.
        If sizes are different, add padding.
        '''
        # Interpolate all graphs to desired sizes
        
        if isinstance(nodes, torch.Tensor):
            assert isinstance(to_sizes, int)
            return F.interpolate(nodes.permute(0, 2, 1), size=to_sizes, mode='linear').permute(0, 2, 1)
        
        
        interpolated_nodes = []
        for idx, (node, size) in enumerate(zip(nodes, to_sizes)):
            if len(node.shape) < 3: node = node.unsqueeze(0)
            node = node.transpose(1, 2)
            node = F.interpolate(node, size=(size if isinstance(size, int) else size.item()), mode='linear')
            node = node.transpose(1, 2).squeeze(0)

            # # Add padding if necessary 
            if padding_size is not None:
                padded = torch.zeros(padding_size, node.size(1), device=node.device)
                padded[gen_mask[idx]] = node
                node = padded
                
            interpolated_nodes.append(node)
            
        return torch.stack(interpolated_nodes, dim=0)
    
    def contex_filter(self, curr_rigids, poc_mask, res_mask, gen_mask, need_poc=False, hidden_str=None):
        """获取空刚体"""
        B, L = poc_mask.shape
        res_mask = res_mask.bool()
        gen_mask = gen_mask.bool()
        
        rotmats = torch.eye(3, device=poc_mask.device).unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1)
        trans = torch.zeros(B, L, 3, device=poc_mask.device, dtype=torch.float)
        
        
        
        if hidden_str is not None:
            # torch.autograd.set_detect_anomaly(True)
            # ridids = du.create_rigid(rotmats, trans)
            # str_vec = self.decoder_init_rigid(hidden_str * gen_mask[..., None]) * gen_mask[..., None]
            # ridids = ridids.compose_q_update_vec(str_vec, gen_mask[..., None])
            # rotmats[gen_mask] = ridids.get_rots().get_rot_mats()[gen_mask]
            # trans[gen_mask] = str_vec[gen_mask]
            # rotmats[res_mask] = so3_utils.rotvec_to_rotmat(hidden_str[:, :, :-3][res_mask])
            # trans[res_mask] = hidden_str[:, :, -3:][res_mask]
            pass
        
        if need_poc:
            rotmats[poc_mask] = curr_rigids.get_rots().get_rot_mats()[poc_mask]
            trans[poc_mask] = curr_rigids.get_trans()[poc_mask]
        
        return du.create_rigid(rotmats, trans)
    
    
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
    
    
def sizes_to_mask(original_sizes, max_size, device):
    ''' Convert list of graph sizes to a binary mask. '''
    B = len(original_sizes)
    mask = torch.zeros(B, max_size, device=device)
    for i, size in enumerate(original_sizes):
        mask[i, :size] = 1
    return mask.bool()



class FeaFusionLayer(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf
        self.rot_net = nn.Sequential(
            nn.Linear(3, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s // 2)
        )
        self.dist_net = nn.Sequential(
            nn.Linear(3, self._ipa_conf.c_s),nn.ReLU(),
            nn.LayerNorm(self._ipa_conf.c_s),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.LayerNorm(self._ipa_conf.c_s),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s // 2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s*2, self._ipa_conf.c_s*2),
            nn.LayerNorm(self._ipa_conf.c_s*2),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s*2, self._ipa_conf.c_s),
            nn.LayerNorm(self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )
        
    def forward(self, node_emb, rot, trans, node_mask):
        # rot = rigid.get_rots().get_rot_mats()
        rot = so3_utils.rotmat_to_rotvec(rot)
        rot = self.rot_net(rot)
        # trans = rigid.get_trans() # B, L, 3
        # dist = (trans[:, None, :, :] - trans[:, :, None, :]).norm(dim=-1, p=2)
        # dist = self.dist_net(dist[..., None]/10.0) * edge_mask[..., None]
        trans = local_transform(trans)
        node_emb_ = self.fusion(torch.cat(
            [node_emb, rot, self.dist_net(trans/10.)], dim=-1
        ))
        node_emb = node_emb_ + node_emb
        
        # edge_emb = edge_emb + dist
        
        return node_emb * node_mask[..., None]
        
def local_transform(coords):
    # 取前三个残基定义局部坐标系
    v1 = coords[..., 1, :] - coords[..., 0, :]
    v2 = coords[..., 2, :] - coords[..., 1, :]
    normal = torch.cross(v1, v2, dim=-1)
    frame = torch.stack([v1, v2, normal], dim=-1)
    return torch.bmm(coords, frame)

# class GAEncoder(nn.Module):
#     def __init__(self, ipa_conf):
#         super().__init__()
#         self._ipa_conf = ipa_conf 

#         # angles
#         self.angles_embedder = AngularEncoding(num_funcs=12) # 25*5=120, for competitive embedding size
#         self.angle_net = nn.Sequential(
#             nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
#             nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
#             nn.Linear(self._ipa_conf.c_s, 5)
#             # nn.Linear(self._ipa_conf.c_s, 22)
#         )

#         # for condition on current seq
#         self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
#         self.seq_net = nn.Sequential(
#             nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
#             nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
#             nn.Linear(self._ipa_conf.c_s, 20)
#             # nn.Linear(self._ipa_conf.c_s, 22)
#         )

#         # mixer
#         self.res_feat_mixer = nn.Sequential(
#             nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5), self._ipa_conf.c_s),
#             nn.ReLU(),
#             nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
#         )

#         self.feat_dim = self._ipa_conf.c_s

#         # Attention trunk
#         self.trunk = nn.ModuleDict()
#         for b in range(self._ipa_conf.num_blocks):
#             self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
#             self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
#             tfmr_in = self._ipa_conf.c_s
#             tfmr_layer = torch.nn.TransformerEncoderLayer(
#                 d_model=tfmr_in,
#                 nhead=self._ipa_conf.seq_tfmr_num_heads,
#                 dim_feedforward=tfmr_in,
#                 batch_first=True,
#                 dropout=0.0,
#                 norm_first=False
#             )
#             self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
#                 tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
#             self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
#                 tfmr_in, self._ipa_conf.c_s, init="final")
#             self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
#                 c=self._ipa_conf.c_s)
#             self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
#                 self._ipa_conf.c_s, use_rot_updates=True)

#             if b < self._ipa_conf.num_blocks-1:
#                 # No edge update on the last block.
#                 edge_in = self._ipa_conf.c_z
#                 self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
#                     node_embed_size=self._ipa_conf.c_s,
#                     edge_embed_in=edge_in,
#                     edge_embed_out=self._ipa_conf.c_z,
#                 )
    
#     def embed_t(self, timesteps, mask):
#         timestep_emb = get_time_embedding(
#             timesteps[:, 0],
#             self.feat_dim,
#             max_positions=2056
#         )[:, None, :].repeat(1, mask.shape[1], 1)
#         return timestep_emb

#     def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
#         num_batch, num_res = seqs_t.shape

#         # incorperate current seq and timesteps
#         node_mask = res_mask
#         edge_mask = node_mask[:, None] * node_mask[:, :, None]

#         node_embed = self.res_feat_mixer(torch.cat([node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t,node_mask), self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)],dim=-1))
#         node_embed = node_embed * node_mask[..., None]
#         curr_rigids = du.create_rigid(rotmats_t, trans_t)
#         for b in range(self._ipa_conf.num_blocks):
#             ipa_embed = self.trunk[f'ipa_{b}'](
#                 node_embed,
#                 edge_embed,
#                 curr_rigids,
#                 node_mask)
#             ipa_embed *= node_mask[..., None]
#             node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
#             seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
#                 node_embed, src_key_padding_mask=(1 - node_mask).bool())
#             node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
#             node_embed = self.trunk[f'node_transition_{b}'](node_embed)
#             node_embed = node_embed * node_mask[..., None]
#             rigid_update = self.trunk[f'bb_update_{b}'](
#                 node_embed * node_mask[..., None])
#             curr_rigids = curr_rigids.compose_q_update_vec(
#                 rigid_update, node_mask[..., None])

#             if b < self._ipa_conf.num_blocks-1:
#                 edge_embed = self.trunk[f'edge_transition_{b}'](
#                     node_embed, edge_embed)
#                 edge_embed *= edge_mask[..., None]
        
#         # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
#         pred_trans1 = curr_rigids.get_trans()
#         pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
#         pred_seqs1_prob = self.seq_net(node_embed)
#         pred_angles1 = self.angle_net(node_embed)
#         pred_angles1 = pred_angles1 % (2*math.pi) # inductive bias to bound between (0,2pi)

#         return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob

    