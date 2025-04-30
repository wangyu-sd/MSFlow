import argparse
from eval.geometry import *
import os
from model.utils.train import ScalarMetricAccumulator
from tqdm.auto import tqdm
from model.utils.misc import get_logger

class PepComp:
  def __init__(self, file_name, chain_id):
    self.file_name = file_name
    self.chain_id = chain_id
    self.seq = get_seq(file_name, chain_id)
    self.chain = get_chain_from_pdb(file_name, chain_id)
    self.traj = get_traj_chain(file_name, chain_id)

  def cmpare(self, other):
    
    # Calculate AAR
    aar = diff_ratio(self.seq, other.seq)
    
    # Calculate RMSD
    rmsd = get_rmsd(self.chain, other.chain)[0]

    # Calculate SSR
    ssr = get_ss(self.traj, other.traj)
    
    # Calculate BSR
    bsr = get_bind_ratio(self.file_name, other.file_name, self.chain_id, other.chain_id)
    return {
        'aar': aar,
        'rmsd': rmsd,
        'ssr': ssr,
        'bsr': bsr
    }
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--eval_dir', type=str, default='/remote-home/wangyu/VQ-PAR/log_sample/learn_all[main-cdf5f7d]_2025_04_27__18_30_16/results')
    # parser.add_argument('--eval_dir', type=str, default='/remote-home/wangyu/VQ-PAR/log_sample/learn_all[main-98e0b67]_2025_04_27__21_06_12/results')
    parser.add_argument('--eval_dir', type=str, default='/remote-home/wangyu/VQ-PAR/log_sample/learn_all[main-87bad41]_2025_04_30__16_57_56/results')
    args = parser.parse_args()
    
    logger_dir = args.eval_dir.replace('results', 'eval_res')
    os.makedirs(logger_dir, exist_ok=True)
    logger = get_logger('eval', log_dir=logger_dir)
    logger.info(args)
    logger.info("Start evaluating")
    
    scalar = ScalarMetricAccumulator()
    for pdb_idx, pdb_ids in enumerate(tqdm(os.listdir(args.eval_dir))):
      path_current = os.path.join(args.eval_dir, pdb_ids)
      batch_size = len(os.listdir(path_current)) - 1
      
      pep_chain_id = pdb_ids.split('_')[1]
      try:
        pdb_gt = os.path.join(path_current, pdb_ids+"_gt.pdb")
        pdb_gt = PepComp(pdb_gt, pep_chain_id)
      except Exception as e:
        logger.info(f"Error in {pdb_ids}: {e}")
        continue
      for sp_idx in range(batch_size):
        pdb_curr = os.path.join(path_current, f"{pdb_ids}_{sp_idx}.pdb")
        pdb_curr = PepComp(pdb_curr, pep_chain_id)
        
        res_dict = pdb_gt.cmpare(pdb_curr)
        
        for k, v in res_dict.items():
          scalar.add(k, v, batchsize=1, mode='mean')
          
      scalar.log(pdb_idx, 'eval'+f'_{pdb_ids}', logger=logger)
      
    scalar.log(pdb_idx, 'eval_all', logger=logger)
    
        
        
        