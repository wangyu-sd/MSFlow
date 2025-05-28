import pyrosetta
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
import torch
import os
import pandas as pd
import subprocess
import numpy as np
import shutil
from tqdm import tqdm
import pickle

from joblib import delayed, Parallel
from eval.utils import *

input_dir=".Tests"
output_dir="./Pack"

def get_chain_dic(input_pdb):
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb)
    chain_dic = {}
    for model in structure:
        for chain in model:
            chain_dic[chain.id] = len([res for res in chain if is_aa(res) and res.has_id('CA')])

    return chain_dic

def get_rosetta_score_base(pdb_path,chain_id='A', iter=1):
    # try:
    init("-mute all -ex1 -ex2 -constant_seed", silent=True)
    keep_backbone_atoms(pdb_path, f'{pdb_path[:-4]}_temp.pdb')
    pose = pyrosetta.pose_from_pdb(f'{pdb_path[:-4]}_temp.pdb')
    chains = list(get_chain_dic(pdb_path).keys())
    chains.remove(chain_id)
    interface = f'{chain_id}_{"".join(chains)}'
    # print(f'Interface: {interface}')
    fast_relax = FastRelax() # cant be pickled
    scorefxn = get_fa_scorefxn()
    fast_relax.set_scorefxn(scorefxn)
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    stabs,binds = [],[]
    for i in range(iter):
        fast_relax.apply(pose)
        stab = scorefxn(pose)
        mover.apply(pose)
        bind = pose.scores['dG_separated']
        stabs.append(stab)
        binds.append(bind)
    os.remove(f'{pdb_path[:-4]}_temp.pdb')
    return {'name':pdb_path,'stab':np.array(stabs).mean(),'bind':np.array(binds).mean()}
    # except KeyboardInterrupt:
    #     print("KeyboardInterrupt: Exiting...")
    #     exit(0)
    # except Exception as e:
    #     print(f"Error processing {pdb_path}: {e}")
    #     return {'name':pdb_path,'stab':999.0,'bind':999.0}


def get_rosetta_score(pdb_path,chain='A'):
    try:
        init()
        temp_pdb = keep_backbone_atoms(pdb_path, f'./_temp_{pdb_path[:-4]}.pdb')
        pose = pyrosetta.pose_from_pdb(temp_pdb)
        # chains = list(get_chain_dic(os.path.join(input_dir,name,'pocket_merge_renum.pdb')).keys())
        # chains.remove(chain)
        # interface = f'{chain}_{"".join(chains)}'
        interface='A_B'
        fast_relax = FastRelax() # cant be pickled
        scorefxn = get_fa_scorefxn()
        fast_relax.set_scorefxn(scorefxn)
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        fast_relax.apply(pose)
        energy = scorefxn(pose)
        mover.apply(pose)
        dg = pose.scores['dG_separated']
        return [pdb_path,energy,dg]
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting...")
        exit(0)
    except:
        return [pdb_path,999.0,999.0]

def pack_sc(name='1a1m_C',num_samples=10):
    try:
        if os.path.exists(os.path.join(output_dir,name,'rosetta')):
            shutil.rmtree(os.path.join(output_dir,name,'rosetta'))
        os.makedirs(os.path.join(output_dir,name,'rosetta'),exist_ok=True)
        init()
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())  # Only repack, don't change amino acid types
        packer = PackRotamersMover()
        packer.task_factory(tf)
        for i in range(num_samples):
            pose = pose_from_pdb(os.path.join(input_dir,name,f'pocket_merge_renum_bb.pdb'))
            packer.apply(pose)
            pose.dump_pdb(os.path.join(output_dir,name,'rosetta',f'packed_{i}.pdb'))
    except:
        return None