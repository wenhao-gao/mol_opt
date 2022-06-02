import os
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
import os
import logging
import tensorflow as tf
import copy 

# dragonfly imports
from dragonfly.exd.worker_manager import SyntheticWorkerManager

# a few local imports here
from chemist_opt.chemist import Chemist

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


class ChemBOoptimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "chembo"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        worker_manager = SyntheticWorkerManager(num_workers=N_WORKERS,
                                                time_distro='const')

        #### self.oracle -> objective_func 
        def objective_func(mol): 
            print(mol[0], type(mol[0])) ### mols.molecule.Molecule "./mols/molecule.py"
            if type(mol)==list:
                mol = mol[0]
            try:
                smiles = mol.to_smiles()
                values = self.oracle(smiles)
            except:
                values = 0.0
            print('======== mol, score, used calls', mol, values, len(self.oracle))

            if len(self.oracle) > config['init_pool_size']: 
                self.oracle.sort_buffer()
                new_scores = [item[1][0] for item in list(self.oracle.mol_buffer.items())[:5]]
                print('  >>>>> new_scores', new_scores, 'old_scores', objective_func.old_scores, \
                      'equal to not:', new_scores == objective_func.old_scores, 'patience', objective_func.patience)
                if new_scores == objective_func.old_scores:
                    objective_func.patience += 1
                    if objective_func.patience >= 5:
                        self.oracle.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        objective_func.stop = True
                else:
                    objective_func.patience = 0
                objective_func.old_scores = copy.deepcopy(new_scores)
            return values 

        objective_func.old_scores = [-1 for i in range(5)]
        objective_func.stop = False 
        objective_func.patience = 0 

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
