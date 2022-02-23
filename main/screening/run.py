from __future__ import print_function

import argparse
import heapq
import yaml
import os
import random
from time import time
from typing import List, Optional

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from tdc import Oracle
from main.optimizer import BaseOptimizer


class Exahsutive_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "screening"

    def _optimize(self, oracle, config):
        
        # import ipdb; ipdb.set_trace()
        all_mols = self.sanitize([Chem.MolFromSmiles(smi) for smi in self.all_smiles])
        np.random.shuffle(all_mols)

        for i in range(0, len(all_mols), 100):
            population_scores = self.score_mol(oracle, all_mols[i: i+100])
            self.log_intermediate()
            if len(self.mol_buffer) >= config["max_n_oracles"]:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

    path_here = os.path.dirname(os.path.realpath(__file__))

    if args.output_dir is None:
        args.output_dir = path_here
    elif not os.path.exist(args.output_dir):
        os.mkdir(args.output_dir)

    try:
        config_default = yaml.safe_load(open(args.config_default))
    except:
        config_default = yaml.safe_load(open(os.path.join(args.output_dir, args.config_default)))

    if args.task == "tune":
        try:
            config_tune = yaml.safe_load(open(args.config_tune))
        except:
            config_tune = yaml.safe_load(open(os.path.join(args.output_dir, args.config_tune)))

    for oracle_name in args.oracles:
        oracle = Oracle(name = oracle_name)
        optimizer = Exahsutive_Optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main()

