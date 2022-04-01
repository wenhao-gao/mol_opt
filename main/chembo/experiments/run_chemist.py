"""
Runner of the Chemist optimization.
Can be used as a usage example.

TODO:
* visualization of synth paths in mols.visualize

NOTE:
* all datasets now are assumed to live in the same folder
  as loaders.py (which contains the Sampler and dataset getters it uses)
"""









import sys 
sys.path.append('.')
# export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/rdkit_contrib:${PWD}/synth/:${PWD}/synth/rexgen_direct"
sys.path.append('./rdkit_contrib')
sys.path.append('./synth/')
sys.path.append('./synth/rexgen_direct')

from myrdkit import *  # :(

from argparse import Namespace, ArgumentParser
import time
import os
import pickle as pkl
import shutil
import logging
import tensorflow as tf

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

def parse_args():
    parser = ArgumentParser()
    # crucial arguments
    parser.add_argument('-d', '--dataset', default='chembl', type=str,
                        help='dataset: chembl or zinc250')
    parser.add_argument('-s', '--seed', default=42, type=int,
                        help='sampling seed for the dataset')
    parser.add_argument('-o', '--objective', default='qed', type=str,
                        help='which objective function to use: qed or logp')
    parser.add_argument('-b', '--budget', default=1000, type=int,
                        help='computational budget (in numbers of BO iterations)') ### 2 
    parser.add_argument('-k', '--kernel', default='similarity_kernel', type=str,
                        help='kernel to use: wl_kernel (and other graphkernels),' +
                        'similarity_kernel, or distance_kernel_expsum')
    parser.add_argument('-i', '--init_pool_size', default=30, type=int,
                        help='size of initial pool')  ### 10 

    # optional arguments
    parser.add_argument('-stp', '--steps', default=10, type=str,
                        help='number of steps of aquisition optimization')
    parser.add_argument('-mpl', '--max_pool_size', default='None', type=str,
                        help='maximum pool size for Explorer, None or int')
    args = parser.parse_args()
    if args.max_pool_size == 'None':
        args.max_pool_size = None
    else:
        args.max_pool_size = int(args.max_pool_size)
    return args


# Create exp directory and point the logger -----------------------------------
def setup_logging():
    # Make directories
    if os.path.exists(EXP_DIR):
        shutil.rmtree(EXP_DIR)
    os.makedirs(EXP_DIR, exist_ok=True)

    # necessary fix for setting the logging after some imports
    from imp import reload
    reload(logging)

    logging.basicConfig(filename=RUN_LOG_FILE, filemode='w',
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=LOGGING_LEVEL)
    tf.logging.set_verbosity(TF_LOGGING_LEVEL)


# Runner ----------------------------------------------------------------------
def main():
    setup_logging()
    args = parse_args()
    # Obtain a reporter and worker manager
    reporter = get_reporter(open(EXP_LOG_FILE, 'w'))
    worker_manager = SyntheticWorkerManager(num_workers=N_WORKERS,
                                            time_distro='const')

    # Problem settings ------ oracle 
    objective_func = get_objective_by_name(args.objective)
    # check MolDomain constructor for full argument list:
    domain_config = {'data_source': args.dataset,
                     'constraint_checker': 'organic',  # not specifying constraint_checker defaults to None
                     'sampling_seed': args.seed}
    chemist_args = {
        'acq_opt_method': 'rand_explorer',
        'init_capital': args.init_pool_size,
        'dom_mol_kernel_type': args.kernel,  # e.g. 'distance_kernel_expsum', 'similarity_kernel', 'wl_kernel'
        'acq_opt_max_evals': args.steps,
        'objective': args.objective,
        'max_pool_size': args.max_pool_size,
        'report_results_every': 1,
        'gpb_hp_tune_criterion': 'ml'
    }

    max_oracle_num = 1000
    import pickle
    smiles_dict = dict()  
    pickle.dump(smiles_dict, open("smiles_dict.pkl", 'wb')) 

    chemist = Chemist(
        objective_func,
        max_oracle_num = max_oracle_num, 
        domain_config=domain_config,
        chemist_args=chemist_args,
        is_mf=False,
        worker_manager=worker_manager,
        reporter=reporter
    )

    opt_val, opt_point, history = chemist.run(args.budget)

    # convert to raw format
    raw_opt_point = chemist.get_raw_domain_point_from_processed(opt_point)
    opt_mol = raw_opt_point[0]

    # Print the optimal value and visualize the molecule and path.
    reporter.writeln(f"\nOptimum value found: {opt_val}")
    reporter.writeln(f"Optimum molecule: {opt_mol} with formula {opt_mol.to_formula()}")
    reporter.writeln(f"Synthesis path: {opt_mol.get_synthesis_path()}")

    # visualize mol/synthesis path
    visualize_file = os.path.join(EXP_DIR, 'optimal_molecule.png')
    reporter.writeln(f'Optimal molecule visualized in {visualize_file}')
    visualize_mol(opt_mol, visualize_file)

    with open(SYN_PATH_FILE, 'wb') as f:
        pkl.dump(opt_mol.get_synthesis_path(), f)

if __name__ == "__main__":
    main()




















