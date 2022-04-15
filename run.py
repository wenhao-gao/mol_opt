from __future__ import print_function

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
import numpy as np
from tdc import Oracle

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
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=500)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    parser.add_argument('--wandb', type=str, default="offline", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = args.wandb

    if not args.log_code:
        os.environ["WANDB_DISABLE_CODE"] = "false"

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(args.method)
    # Add method name here when adding new ones
    if args.method == 'screening':
        from main.screening.run import Exhaustive_Optimizer 
        Optimizer = Exhaustive_Optimizer
    elif args.method == 'molpal':
        from main.molpal.run import MolPAL_Optimizer 
        Optimizer = MolPAL_Optimizer

    elif args.method == 'graph_ga':
        from main.graph_ga.run import GB_GA_Optimizer
        Optimizer = GB_GA_Optimizer
    elif args.method == 'smiles_ga':
        from main.smiles_ga.run import SMILES_GA_Optimizer
        Optimizer = SMILES_GA_Optimizer
    elif args.method == "selfies_ga":
        from main.selfies_ga.run import SELFIES_GA_optimizer 
        Optimizer = SELFIES_GA_optimizer 

    elif args.method == 'graph_mcts':
        from main.graph_mcts.run import Graph_MCTS_Optimizer 
        Optimizer = Graph_MCTS_Optimizer 

    elif args.method == "smiles_lstm_hc":
        from main.smiles_lstm_hc.run import SMILES_LSTM_HC_Optimizer 
        Optimizer = SMILES_LSTM_HC_Optimizer 
    elif args.method == 'selfies_lstm_hc':
        from main.selfies_lstm_hc.run import SELFIES_LSTM_HC_Optimizer
        Optimizer = SELFIES_LSTM_HC_Optimizer 
    elif args.method == 'dog_gen':
        from main.dog_gen.run import DoG_Gen_Optimizer
        Optimizer = DoG_Gen_Optimizer 

    elif args.method == 'boss':
        from main.boss.run import BOSS_Optimizer 
        Optimizer = BOSS_Optimizer 
    elif args.method == 'gpbo':
        from main.gpbo.run import GPBO_optimizer
        Optimizer = GPBO_optimizer 

    elif args.method == "selfies_VAE":
        from main.selfies_vae.run import selfies_VAEBO_optimizer 
        Optimizer = selfies_VAEBO_optimizer 
    elif args.method == "smiles_vae":
        from main.smiles_vae.run import smiles_VAEBO_optimizer 
        Optimizer = smiles_VAEBO_optimizer 
    elif args.method == 'JTVAE':
        from main.JTVAE.run import JTVAEBOoptimizer
        Optimizer = JTVAEBOoptimizer 
    elif args.method == 'dog_ae':
        from main.dog_ae.run import DoG_AE_Optimizer
        Optimizer = DoG_AE_Optimizer 

    elif args.method == 'DST':
        from main.DST.run import DSToptimizer
        Optimizer = DSToptimizer 

    elif args.method == 'MARS':
        from main.MARS.run import MARS_Optimizer 
        Optimizer = MARS_Optimizer 
    elif args.method == 'MIMOSA':
        from main.MIMOSA.run import MIMOSA_Optimizer 
        Optimizer = MIMOSA_Optimizer 
    elif args.method == 'ORGAN':
        from main.ORGAN.run import ORGAN_Optimizer 
        Optimizer = ORGAN_Optimizer 
    elif args.method == 'REINVENT':
        from main.REINVENT.run import REINVENToptimizer
        Optimizer = REINVENToptimizer
    elif args.method == 'REINVENT_SELFIES':
        from main.REINVENT_SELFIES.run import REINVENT_SELFIES_optimizer 
        Optimizer = REINVENT_SELFIES_optimizer 
    elif args.method == "rationaleRL":
        from main.rationaleRL.run import RationaleRLoptimizer 
        Optimizer = RationaleRLoptimizer 
    else:
        raise ValueError("Unrecognized method name.")

    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.pickle_directory is None:
        args.pickle_directory = path_main

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

