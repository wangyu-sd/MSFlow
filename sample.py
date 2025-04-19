from model.vqpar import VQPAR
from model.gpt import MultiScaleGPT
import torch
@torch.no_grad()
def sample(batch, vqpar: VQPAR, gpt: MultiScaleGPT):
    
    gen_mask,res_mask, angle_mask = batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()

    #encode
    rotmats, trans, angles, seqs, node_embed, edge_embed = vqpar.extract_fea(batch) # no generate mask

    context_mask = torch.logical_and(res_mask, ~gen_mask)
    trans_c, _ = vqpar.zero_center_part(trans, context_mask, context_mask)
    pred_rotmats, pred_trans, pred_angles, pred_seqs_prob, vq_loss, pocket_indices = vqpar.vqvae(
        rotmats, trans_c, angles, seqs, node_embed, edge_embed, gen_mask, res_mask, context_only=True,
    )
    
    pred_peptide_indices = gpt.gen_seq(pocket_indices)
    pred_peptide_indices = torch.argmax(pred_peptide_indices, dim=-1)
    
    
    trans_c, _ = vqpar.zero_center_part(trans, gen_mask, res_mask)
    pred_rotmats, pred_trans, pred_angles, pred_seqs_prob = vqpar.vqvae.indices2res(
        pred_peptide_indices, rotmats, trans, node_embed, edge_embed, gen_mask, res_mask)
    
    return {
        "pred_rotmats": pred_rotmats.detach().cpu(),
        "pred_trans": pred_trans.detach().cpu(),
        "pred_angles": pred_angles.detach().cpu(),
        "pred_seqs_prob": pred_seqs_prob.detach().cpu(),
        "pred_peptide_indices": pred_peptide_indices.detach().cpu(),
        "trans": trans.detach().cpu(),
        "rotmats": rotmats.detach().cpu(),
        "angles": angles.detach().cpu(),
        "seqs": seqs.detach().cpu(),
        "gen_mask": gen_mask.detach().cpu(),
        "res_mask": res_mask.detach().cpu(),
        "context_mask": context_mask.detach().cpu(),
        "pocket_indices": pocket_indices.detach().cpu(),
        # "peptide_indices": peptide_indices.detach().cpu(),
    }