from __future__ import print_function

import argparse
import yaml
import os
from copy import deepcopy
import numpy as np
from tdc import Oracle
from main.optimizer import BaseOptimizer

class Exhaustive_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "screening"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        all_mols = deepcopy(self.all_smiles)
        np.random.shuffle(all_mols)

        for i in range(0, len(all_mols), 100):
            scores = self.oracle(all_mols[i: i+100])
            if self.finish:
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

    path_here = os.path.dirname(os.path.realpath(__file__))

    if args.output_dir is None:
        args.output_dir = os.path.join(path_here, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for oracle_name in args.oracles:

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_here, args.config_default)))

        if args.task == "tune":
            try:
                config_tune = yaml.safe_load(open(args.config_tune))
            except:
                config_tune = yaml.safe_load(open(os.path.join(path_here, args.config_tune)))

        oracle = Oracle(name = oracle_name)
        optimizer = Exhaustive_Optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main()

