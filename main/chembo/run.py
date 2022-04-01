"""
Runner of the Chemist optimization.
Can be used as a usage example.

TODO:
* visualization of synth paths in mols.visualize

NOTE:
* all datasets now are assumed to live in the same folder
  as loaders.py (which contains the Sampler and dataset getters it uses)
"""

import os, pickle, torch, random, argparse
import yaml
import numpy as np 
from tqdm import tqdm 
torch.manual_seed(1)
np.random.seed(2)
random.seed(1)
from tdc import Oracle
import sys
# sys.path.append('../..')
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer

# export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/rdkit_contrib:${PWD}/synth/:${PWD}/synth/rexgen_direct"
sys.path.append(os.path.join(path_here, 'rdkit_contrib'))
sys.path.append(os.path.join(path_here, 'synth/'))
sys.path.append(os.path.join(path_here, 'synth/rexgen_direct'))

from myrdkit import *  # :(

from argparse import Namespace, ArgumentParser
import time
import os
import pickle as pkl
import shutil
import logging
import tensorflow as tf

from rdkit import Chem 

# dragonfly imports
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.utils.reporters import get_reporter

# a few local imports here
from chemist_opt.chemist import Chemist
from mols.mol_functions import get_objective_by_name
from mols.visualize import visualize_mol

# Where to store temporary model checkpoints
EXP_PREFIX = "sum_kernel"
# EXP_DIR = f"experiments/results/{EXP_PREFIX}/chemist_exp_dir_{time.strftime('%Y%m%d%H%M%S')}" 
EXP_DIR = "exp"
EXP_LOG_FILE = os.path.join(EXP_DIR, 'exp_log')
RUN_LOG_FILE = os.path.join(EXP_DIR, 'run_log')
SYN_PATH_FILE = os.path.join(EXP_DIR, 'best_molecule.pkl')
LOGGING_LEVEL = logging.DEBUG #logging.INFO
TF_LOGGING_LEVEL = tf.logging.ERROR
N_WORKERS = 1



# def parse_args():
#     parser = ArgumentParser()
#     # crucial arguments
#     parser.add_argument('-d', '--dataset', default='chembl', type=str,
#                         help='dataset: chembl or zinc250')
#     parser.add_argument('-s', '--seed', default=42, type=int,
#                         help='sampling seed for the dataset')
#     parser.add_argument('-o', '--objective', default='qed', type=str,
#                         help='which objective function to use: qed or logp')
#     parser.add_argument('-b', '--budget', default=1000, type=int,
#                         help='computational budget (in numbers of BO iterations)') ### 2 
#     parser.add_argument('-k', '--kernel', default='similarity_kernel', type=str,
#                         help='kernel to use: wl_kernel (and other graphkernels),' +
#                         'similarity_kernel, or distance_kernel_expsum')
#     parser.add_argument('-i', '--init_pool_size', default=30, type=int,
#                         help='size of initial pool')  ### 10 

#     # optional arguments
#     parser.add_argument('-stp', '--steps', default=10, type=str,
#                         help='number of steps of aquisition optimization')
#     parser.add_argument('-mpl', '--max_pool_size', default='None', type=str,
#                         help='maximum pool size for Explorer, None or int')
#     args = parser.parse_args()
#     if args.max_pool_size == 'None':
#         args.max_pool_size = None
#     else:
#         args.max_pool_size = int(args.max_pool_size)
#     return args


class ChemBOoptimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "chembo"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)

        # args = parse_args()
        # Obtain a reporter and worker manager
        # reporter = get_reporter(open(EXP_LOG_FILE, 'w'))
        worker_manager = SyntheticWorkerManager(num_workers=N_WORKERS,
                                                time_distro='const')

        # Problem settings ------ oracle 
        # objective_func = get_objective_by_name(args.objective) 
        def objective_func(mol):
            # print('input of objective_func', mol, type(mol))
            print(mol[0], type(mol[0])) ### mols.molecule.Molecule "./mols/molecule.py"
            if type(mol)==list:
                mol = mol[0]
            try:
                # smiles = Chem.MolToSmiles(m)
                smiles = mol.to_smiles() 
                values = self.oracle(smiles)                
            except:
                values = 0.0
            print('======== mol, score, used calls', mol, values, len(self.oracle))
            return values 

            # print(self.oracle(mol))
            # exit() 
            # # print()
            # return self.oracle(mol)
            # if type(mol)==list:
            #     print(mol)
            #     smiles_lst = [Chem.MolToSmiles(m) for m in mol]
            #     return self.oracle(smiles_lst)
            # smiles = Chem.MolToSmiles(mol)
            # return self.oracle(smiles)

        chemist_args = {
            'acq_opt_method': 'rand_explorer',
            'init_capital': int(config['init_pool_size']),
            'dom_mol_kernel_type': config['kernel'],  # e.g. 'distance_kernel_expsum', 'similarity_kernel', 'wl_kernel'
            'acq_opt_max_evals': int(config['steps']),
            'objective': 'optimize',
            'max_pool_size': int(config['max_pool_size']),
            'report_results_every': 1,
            'gpb_hp_tune_criterion': 'ml'
        }

        # budget = int(config['max_pool_size']) - int(config['init_pool_size'])
        budget = self.oracle.max_oracle_calls - int(config['init_pool_size'])
        if budget <= 100:
            budget = 100
        # num_run = int(self.oracle.max_oracle_calls / budget) * 10 ### slight enhance number of runs
        # for i in range(num_run):
            # check MolDomain constructor for full argument list:
        domain_config = {'data_source': config['dataset'],
                         'constraint_checker': 'organic',  # not specifying constraint_checker defaults to None
                         'sampling_seed': 0}
        chemist = Chemist(
                objective_func,
                domain_config=domain_config,
                chemist_args=chemist_args,
                is_mf=False,
                worker_manager=worker_manager,
            )
        opt_val, opt_point, history = chemist.run(budget)
        # print(opt_val, opt_point, history)
        # if self.finish:
            # break 

        # print('>>>>>>>>> length of molbuffer', len(self.oracle))


# Create exp directory and point the logger -----------------------------------
# def setup_logging():
#     # Make directories
#     if os.path.exists(EXP_DIR):
#         shutil.rmtree(EXP_DIR)
#     os.makedirs(EXP_DIR, exist_ok=True)

#     # necessary fix for setting the logging after some imports
#     from imp import reload
#     reload(logging)

#     logging.basicConfig(filename=RUN_LOG_FILE, filemode='w',
#                         format='%(asctime)s - %(message)s',
#                         datefmt='%d-%b-%y %H:%M:%S',
#                         level=LOGGING_LEVEL)
#     tf.logging.set_verbosity(TF_LOGGING_LEVEL)


# Runner ----------------------------------------------------------------------
# def main():
#     # setup_logging()
#     args = parse_args()
#     # Obtain a reporter and worker manager
#     reporter = get_reporter(open(EXP_LOG_FILE, 'w'))
#     worker_manager = SyntheticWorkerManager(num_workers=N_WORKERS,
#                                             time_distro='const')

#     # Problem settings ------ oracle 
#     objective_func = get_objective_by_name(args.objective)
#     # check MolDomain constructor for full argument list:
#     domain_config = {'data_source': args.dataset,
#                      'constraint_checker': 'organic',  # not specifying constraint_checker defaults to None
#                      'sampling_seed': args.seed}
#     chemist_args = {
#         'acq_opt_method': 'rand_explorer',
#         'init_capital': args.init_pool_size,
#         'dom_mol_kernel_type': args.kernel,  # e.g. 'distance_kernel_expsum', 'similarity_kernel', 'wl_kernel'
#         'acq_opt_max_evals': args.steps,
#         'objective': args.objective,
#         'max_pool_size': args.max_pool_size,
#         'report_results_every': 1,
#         'gpb_hp_tune_criterion': 'ml'
#     }

#     max_oracle_num = 1000
#     smiles_dict = dict()
#     pickle.dump(smiles_dict, open("smiles_dict.pkl", 'wb')) 

#     chemist = Chemist(
#         objective_func,
#         max_oracle_num = max_oracle_num, 
#         domain_config=domain_config,
#         chemist_args=chemist_args,
#         is_mf=False,
#         worker_manager=worker_manager,
#         reporter=reporter
#     )

#     opt_val, opt_point, history = chemist.run(args.budget)

    # convert to raw format
    # raw_opt_point = chemist.get_raw_domain_point_from_processed(opt_point)
    # opt_mol = raw_opt_point[0]

    # Print the optimal value and visualize the molecule and path.
    # reporter.writeln(f"\nOptimum value found: {opt_val}")
    # reporter.writeln(f"Optimum molecule: {opt_mol} with formula {opt_mol.to_formula()}")
    # reporter.writeln(f"Synthesis path: {opt_mol.get_synthesis_path()}")

    # # visualize mol/synthesis path
    # visualize_file = os.path.join(EXP_DIR, 'optimal_molecule.png')
    # reporter.writeln(f'Optimal molecule visualized in {visualize_file}')
    # visualize_mol(opt_mol, visualize_file)

    # with open(SYN_PATH_FILE, 'wb') as f:
    #     pkl.dump(opt_mol.get_synthesis_path(), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=100)
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
        optimizer = ChemBOoptimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main() 




















