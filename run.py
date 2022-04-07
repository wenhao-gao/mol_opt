from __future__ import print_function

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
import numpy as np
from tdc import Oracle
from main.graph_ga.run import GB_GA_Optimizer
from main.REINVENT.run import REINVENToptimizer

# def disable_rdkit_logging():
#     """
#     Disables RDKit whiny logging.
#     """
#     import rdkit.rdBase as rkrb
#     import rdkit.RDLogger as rkl
#     logger = rkl.logger()
#     logger.setLevel(rkl.ERROR)
#     rkrb.DisableLog('rdApp.error')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='graph_ga')
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
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    parser.add_argument('--wandb_mode', type=str, default="offline", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = args.wandb_mode

    if not args.log_code:
        os.environ["WANDB_DISABLE_CODE"] = "false"
    
    # Add method name here when adding new ones
    if args.method == 'graph_ga':
        Optimizer = GB_GA_Optimizer
    elif args.method == 'screening':
        Optimizer = None
    elif args.method == 'REINVENT':
        Optimizer = REINVENToptimizer
    else:
        raise ValueError("Unrecognized method name.")

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for oracle_name in args.oracles:

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

        if args.task == "tune":
            try:
                config_tune = yaml.safe_load(open(args.config_tune))
            except:
                config_tune = yaml.safe_load(open(os.path.join(path_main, args.config_tune)))

        oracle = Oracle(name = oracle_name)
        optimizer = Optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    # disable_rdkit_logging()
    main()

