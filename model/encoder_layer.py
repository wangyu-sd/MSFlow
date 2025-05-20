import torch
from torch import nn

from models_con import ipa_pytorch as ipa_pytorch
from dm import utils as du

from models_con.utils import get_index_embedding, get_time_embedding

from model.modules.protein.constants import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE
from model.modules.common.layers import AngularEncoding

import math


class GAEncoder(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf 

        # rcd
        self.rcd_embedder = nn.Linear(14*3, self._ipa_conf.c_s)
        self.rcd_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 14*3)
        )
        # self.rcd_embedder = AngularEncoding(num_funcs=12) # 25*5=120, for competitive embedding size
        # self.angle_net = nn.Sequential(
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, 5)
        #     # nn.Linear(self._ipa_conf.c_s, 22)
        # )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(4 * self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )
    
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, batch):
        """
        Args:
            batch (Dict):
                "t": t, 
                "rotmats": rotmats_t,
                "trans_t": trans_t,
                "rcd_t": rcd_t, [B, L, 14, 3]
                "seqs_t": seqs_t,
                "generate_mask": gen_mask,
                "res_mask": res_mask,
                "node_embed": node_embed,
                "edge_embed": edge_embed,

        Returns:
            dict: 
        """
        t,  rotmats_t, trans_t, rcd_t, seqs_t, generate_mask, res_mask, node_embed, edge_embed = \
            batch['t'], batch['rotmats_t'], batch['trans_t'], batch['rcd_t'], batch['seqs_t'], \
            batch['generate_mask'], batch['res_mask'], batch['node_embed'], batch['edge_embed']
        
        
        num_batch, num_res = seqs_t.shape
        rcd_t = rcd_t.view(num_batch, num_res, -1)
        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(
            torch.cat([
                node_embed, 
                self.current_seq_embedder(seqs_t), 
                self.embed_t(t,node_mask), 
                self.rcd_embedder(rcd_t).reshape(num_batch,num_res,-1)],dim=-1)
            )
        
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        
        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_rcd = self.rcd_net(node_embed).reshape(num_batch, num_res, 14, 3)

        return {
            "pred_trans": pred_trans1,
            "pred_rotmats": pred_rotmats1,
            "pred_seqs": pred_seqs1_prob,
            "pred_rcd": pred_rcd
        }


    