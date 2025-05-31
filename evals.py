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
    tm = get_tm(other.file_name, other.chain_id, self.file_name, self.chain_id)
    return {
        'aar': aar,
        'rmsd': rmsd,
        'ssr': ssr,
        'bsr': bsr,
        'tm': tm
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
    parser.add_argument('--resume_iter', type=int, default=0, help='Whether to resume from the logger directory')
    parser.add_argument('--add_sample', type=int, default=0, help='Whether to resume from the logger directory')
    parser.add_argument('--total_size', type=int, default=64, help='Whether to resume from the logger directory')
    parser.add_argument("--mode", type=str, default='geo', choices=['geo', 'eng', 'pac'], help="Mode of evaluation")
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
        total_size = args.total_size
        
        pep_chain_id = pdb_ids.split('_')[-1]
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
          
          df = df._append({
              'pdb_ids': pdb_ids,
              'aar': res_dict['aar'],
              'rmsd': res_dict['rmsd'],
              'ssr': res_dict['ssr'],
              'bsr': res_dict['bsr'],
              'tm': res_dict['tm']
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
      total_size = 64
      pdb_list = sorted(os.listdir(args.eval_dir))
      if args.resume_iter > 0:
        df = pd.read_csv(os.path.join(logger_dir, 'eng.csv'))
        df_list = df.values.tolist()
        logger.info(f"Resuming from {args.resume_iter} entries")
        logger.info(f"Resuming from {args.resume_iter}:{pdb_list[args.resume_iter]} entries")
        cur_pdb_id, stab_gt, bind_gt = None, None, None
        stabs, binds = [], []
        irer_count = 0
        for pdb_id, stab, bind, stab_improve, bind_improve, group in df_list:

          if group == 'Ground Truth':
            if stabs:
              scalar.add('stab', np.mean(stabs), batchsize=len(stabs), mode='mean')
              scalar.add('bind', np.mean(binds), batchsize=len(binds), mode='mean')
              scalar.add('stab_improve', np.mean(np.array(stabs) < float(stab_gt)), batchsize=len(stabs), mode='mean')
              scalar.add('bind_improve', np.mean(np.array(binds) < float(bind_gt)), batchsize=len(binds), mode='mean')
              scalar.log(irer_count, 'eval'+f'_{cur_pdb_id}', logger=logger)
              irer_count += 1
            cur_pdb_id, stab_gt, bind_gt = pdb_id, stab, bind
            stabs, binds = [], []
            
          else:
            assert cur_pdb_id == pdb_id, f"Mismatch in pdb_id: {cur_pdb_id} != {pdb_id}"
            stabs.append(float(stab))
            binds.append(float(bind))
          
        scalar.add('stab', np.mean(stabs), batchsize=len(stabs), mode='mean')
        scalar.add('bind', np.mean(binds), batchsize=len(binds), mode='mean')
        scalar.add('stab_improve', np.mean(np.array(stabs) < float(stab_gt)), batchsize=len(stabs), mode='mean')
        scalar.add('bind_improve', np.mean(np.array(binds) < float(bind_gt)), batchsize=len(binds), mode='mean')
        scalar.log(irer_count, 'eval'+f'_{cur_pdb_id}', logger=logger)
        irer_count += 1
          
      
      for pdb_idx, pdb_ids in enumerate(tqdm(pdb_list[args.resume_iter:])):
        print(pdb_ids)
        path_current = os.path.join(args.eval_dir, pdb_ids)
        
        pep_chain_id = pdb_ids.split('_')[-1]
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
      df.to_csv(os.path.join(logger_dir, 'eng.csv'), index=False)
      
      
    elif args.mode == 'pac':
      df = pd.DataFrame(columns=['pdb_ids', 
                                 'psi_mse', 'chi1_mse', 'chi2_mse', 'chi3_mse', 'chi4_mse',
                                 'psi_correct', 'chi1_correct', 'chi2_correct', 'chi3_correct', 'chi4_correct',
                                 'group'])
      
      logger.info("Evaluating side-chain packing metrics")
      
      for pdb_idx, pdb_ids in enumerate(tqdm(os.listdir(args.eval_dir))):
        path_current = os.path.join(args.eval_dir, pdb_ids)
        total_size = args.total_size
        
        pep_chain_id = pdb_ids.split('_')[-1]
        pdb_gt = os.path.join(path_current, pdb_ids+"_gt.pdb")
        tor_gt, tor_gt_mask = get_torsion_anglgs_form_pdb(pdb_gt, pep_chain_id)
        
        df = df._append({
            'pdb_ids': pdb_ids,
            'psi_mse': 0.0,
            'chi1_mse': 0.0,
            'chi2_mse': 0.0,
            'chi3_mse': 0.0,
            'chi4_mse': 0.0,
            'psi_correct': 1.0,
            'chi1_correct': 1.0,
            'chi2_correct': 1.0,
            'chi3_correct': 1.0,
            'chi4_correct': 1.0,
            'group': 'Ground Truth'
        }, ignore_index=True)
        
        for sp_idx in range(total_size):
          pdb_curr = os.path.join(path_current, f"{pdb_ids}_{sp_idx}.pdb")
          
          cur_dict = {
                'pdb_ids': pdb_ids,
                'group': f'Sample',
            }
          
          for angle_idx, angle_name in enumerate(['psi', 'chi1', 'chi2', 'chi3', 'chi4']):
            tor_curr, tor_curr_mask = get_torsion_anglgs_form_pdb(pdb_curr, pep_chain_id)   
            
            if tor_curr is None:
              logger.info(f"Skipping {pdb_ids}_{sp_idx} due to missing torsion angles")
              continue
            if angle_idx >= tor_curr.shape[1]:
              logger.info(f"Skipping {pdb_ids}_{sp_idx} due to missing angle index {angle_idx}")
              continue
            con_mask = torch.logical_and(tor_gt_mask[:, angle_idx], tor_curr_mask[:, angle_idx])
            # con_mask = tor_gt_mask[:, angle_idx]
            mse = (tor_gt[:, angle_idx] - tor_curr[:, angle_idx])
            mse = torch.stack([mse, mse + 180, mse - 180], dim=-1)
            mse = mse.pow(2).min(dim=-1)[0] * con_mask.float()
            mse = mse.sum(dim=-1) / con_mask.sum(dim=-1)
            mse = mse.mean().item()
            
            scalar.add(f'{angle_name}_mse', mse, batchsize=1, mode='mean')
            correct = (tor_gt[:, angle_idx] - tor_curr[:, angle_idx]).abs() < 20
            correct = correct.float().mean().item()
            scalar.add(f'{angle_name}_correct', correct, batchsize=1, mode='mean')
            
            cur_dict[f'{angle_name}_mse'] = mse
            cur_dict[f'{angle_name}_correct'] = correct
        
          df = df._append(cur_dict, ignore_index=True)
          df.to_csv(os.path.join(logger_dir, 'pac.csv'), index=False)  
        scalar.log(pdb_idx, 'eval'+f'_{pdb_ids}', logger=logger)
        df.to_csv(os.path.join(logger_dir, 'pac.csv'), index=False)  
        
            
            
            
            
          
          
        
      
      
        
    
        
        
        