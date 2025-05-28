import argparse
from eval.geometry import *
from eval.energy import get_rosetta_score_base
import os
from model.utils.train import ScalarMetricAccumulator
from tqdm.auto import tqdm
from model.utils.misc import get_logger
import pandas as pd
import multiprocessing
from multiprocessing import Pool
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
    rmsd = get_rmsd(self.chain, other.chain)[1]

    # Calculate SSR
    ssr = get_ss(self.traj, other.traj)
    
    # Calculate BSR
    bsr = get_bind_ratio(other.file_name, self.file_name, other.chain_id, self.chain_id)
    return {
        'aar': aar,
        'rmsd': rmsd,
        'ssr': ssr,
        'bsr': bsr
    }
    
    
def single_process(pdb_ids, sp_idx, pep_chain_id, gt_dict):
          pdb_curr = os.path.join(path_current, f"{pdb_ids}_{sp_idx}.pdb")
          energy_dict = get_rosetta_score_base(pdb_curr, pep_chain_id)
          cur_list = [pdb_ids, energy_dict['stab'], energy_dict['bind'],
                          float(energy_dict['stab'] < gt_dict['stab']),
                          float(energy_dict['bind'] < gt_dict['bind']), f'Sample']
          return cur_list
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--eval_dir', type=str, default='/remote-home/wangyu/VQ-PAR/log_sample/learn_all[main-cdf5f7d]_2025_04_27__18_30_16/results')
    # parser.add_argument('--eval_dir', type=str, default='/remote-home/wangyu/VQ-PAR/log_sample/learn_all[main-98e0b67]_2025_04_27__21_06_12/results')
    # parser.add_argument('--eval_dir', type=str, default='/remote-home/wangyu/VQ-PAR/log_sample/learn_all[main-36612ab]_2025_05_05__02_31_46/results')
    parser.add_argument('--eval_dir',  type=str,  default="/remote-home/wangyu/VQ-PAR/logs/learn_all[main-f456a86]_2025_05_22__17_06_13/checkpoints_286331_last_v1/results")
    parser.add_argument("--mode", type=str, default='geo', choices=['geo', 'eng'], help="Mode of evaluation")
    parser.add_argument('--tag',  type=str,  default="")
    parser.add_argument('--num_workers', type=int, default=64, help='Number of workers for parallel processing')
    args = parser.parse_args()
    
    logger_dir = args.eval_dir.replace('results', 'eval_res')
    os.makedirs(logger_dir, exist_ok=True)
    logger = get_logger(f'eval_{args.mode}_{args.tag}', log_dir=logger_dir)
    logger.info(args)

    scalar = ScalarMetricAccumulator()
    if args.mode == 'geo':
      df = pd.DataFrame(columns=['pdb_ids', 'aar', 'rmsd', 'ssr', 'bsr'])
      logger.info("Evaluating geometry metrics")
      for pdb_idx, pdb_ids in enumerate(tqdm(os.listdir(args.eval_dir))):
        path_current = os.path.join(args.eval_dir, pdb_ids)
        total_size = len(os.listdir(path_current)) - 1
        
        pep_chain_id = pdb_ids.split('_')[1]
        try:
          pdb_gt = os.path.join(path_current, pdb_ids+"_gt.pdb")
          pdb_gt = PepComp(pdb_gt, pep_chain_id)
        except Exception as e:
          logger.info(f"Error in {pdb_ids}: {e}")
          continue
        for sp_idx in range(total_size):
          pdb_curr = os.path.join(path_current, f"{pdb_ids}_{sp_idx}.pdb")
          pdb_curr = PepComp(pdb_curr, pep_chain_id)
          
          res_dict = pdb_gt.cmpare(pdb_curr)
          
          df = df.append({
              'pdb_ids': pdb_ids,
              'aar': res_dict['aar'],
              'rmsd': res_dict['rmsd'],
              'ssr': res_dict['ssr'],
              'bsr': res_dict['bsr']
          }, ignore_index=True)
          
          for k, v in res_dict.items():
            scalar.add(k, v, batchsize=1, mode='mean')
          df.to_csv(os.path.join(logger_dir, 'geo.csv'), index=False)
            
        scalar.log(pdb_idx, 'eval'+f'_{pdb_ids}', logger=logger)
        
      scalar.log(pdb_idx, 'eval_all', logger=logger)
      df.to_csv(os.path.join(logger_dir, 'geo.csv'), index=False)
    
    

    elif args.mode == 'eng':      
      df_names = ['pdb_ids', 'stab', 'bind', 'stab_improve', 'bind_improve', 'group']
      df_list = []
      
      pdb_list = sorted(os.listdir(args.eval_dir))
      for pdb_idx, pdb_ids in enumerate(tqdm(pdb_list)):
        path_current = os.path.join(args.eval_dir, pdb_ids)
        total_size = 63
        
        pep_chain_id = pdb_ids.split('_')[1]
        pdb_gt = os.path.join(path_current, pdb_ids+"_gt.pdb")
        energy_dict = get_rosetta_score_base(pdb_gt, pep_chain_id)
        logger.info(f"Processing {pdb_ids}: {energy_dict}")
        gt_dict = energy_dict
        df_list.append([pdb_ids, energy_dict['stab'], energy_dict['bind'], 0.0, 0.0, 'Ground Truth'])
        
        p = Pool(processes=args.num_workers)
        futures = [p.apply_async(single_process, args=(pdb_ids, sp_idx, pep_chain_id, gt_dict)) for sp_idx in tqdm(range(total_size))]
        cur_lists = [f.get() for f in futures]
        p.close()
        p.join()
        df_list.extend(cur_lists)
        cur_dict = {
            'pdb_ids': [cur_list[0] for cur_list in cur_lists],
            'stab': [cur_list[1] for cur_list in cur_lists],
            'bind': [cur_list[2] for cur_list in cur_lists],
            'stab_improve': [cur_list[3] for cur_list in cur_lists],
            'bind_improve': [cur_list[4] for cur_list in cur_lists],
        }
        
        for k, v in cur_dict.items():
          if k == 'pdb_ids':
            continue
          # print(f"Adding {k} with values: {v}")
          v = np.array(v, dtype=np.float32)
          v = np.mean(v)
          scalar.add(k, v, batchsize=len(cur_lists), mode='mean')
        scalar.log(pdb_idx, 'eval'+f'_{pdb_ids}', logger=logger)
        
        df = pd.DataFrame(df_list, columns=df_names)
        df.to_csv(os.path.join(logger_dir, 'eng.csv'), index=False)
      
      df = pd.DataFrame(df_list, columns=df_names)
      df.to_csv(os.path.join(logger_dir, 'engy.csv'), index=False)
      
      
    elif args.mode == 'pac':
      pass
      
        
    
        
        
        