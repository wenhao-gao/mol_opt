"""
Run pure exploration.

This experiment is both for comparison against Chemist,
and for validation of explored output.

"""

import os
import time
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from explorer.mol_explorer import RandomExplorer
from mols.mol_functions import get_objective_by_name
from datasets.loaders import get_chembl
from dragonfly.utils.reporters import get_reporter
from synth.validators import compute_min_sa_score, check_validity
from datasets.loaders import MolSampler

# Where to store temporary model checkpoints
EXP_DIR = 'experiments/results/extra_exps/rand_exp_dir_%s'%(time.strftime('%Y%m%d%H%M%S'))
EXP_LOG_FILE = os.path.join(EXP_DIR, 'exp_log')
PLOT_FILE = os.path.join(EXP_DIR, 'explorer.eps')
SYN_PATH_FILE = os.path.join(EXP_DIR, 'best_molecule.pkl')
OPT_VALS_FILE = os.path.join(EXP_DIR, 'opt_vals')
if os.path.exists(EXP_DIR):
    shutil.rmtree(EXP_DIR)
os.makedirs(EXP_DIR, exist_ok=True)

def parse_args():
    parser = ArgumentParser()
    # crucial arguments
    parser.add_argument('-d', '--dataset', default='chembl', type=str,
                        help='dataset: chembl or zinc250')
    parser.add_argument('-s', '--seed', default=42, type=int,
                        help='sampling seed for the dataset')
    parser.add_argument('-o', '--objective', default='qed', type=str,
                        help='which objective function to use: qed or logp')
    parser.add_argument('-b', '--budget', default=100, type=int,
                        help='computational budget (# of function `evaluations`)')
    parser.add_argument('-i', '--init_pool_size', default=10, type=int,
                        help='size of initial pool')

    # optional arguments
    parser.add_argument('-mpl', '--max_pool_size', default='None', type=str,
                        help='maximum pool size for Explorer, None or int')

    args = parser.parse_args()
    if args.max_pool_size == 'None':
        args.max_pool_size = None
    else:
        args.max_pool_size = int(args.max_pool_size)
    return args


def explore_and_validate_synth(init_pool_size, seed, budget, objective,
                               dataset, max_pool_size, reporter):
    """
    This experiment is equivalent to unlimited-evaluation optimization.
    It compares optimal found vs optimal over pool, and checks if synthesizeability is improved.
    """
    # obj_func = get_objective_by_name(objective)
    sampler = MolSampler(dataset, sampling_seed=seed)
    pool = sampler(init_pool_size)

    from tdc import Oracle 
    property_name = 'logp'
    f = property_name + '.txt'
    oracle = Oracle(property_name)
    def obj_func(x):
        property_name = 'logp'
        f = property_name + '.txt'
        fout = open(f, 'a+')
        if type(x)==list:
            x = [str(i) for i in x]
            results = oracle(x)
            for i,j in zip(x,results):
                fout.write(i + '\t' + str(j) + '\n')   
        else:
            x = str(x)
            results = oracle(x)
            fout.write(x + '\t' + str(results) + '\n')
        return results 

    exp = RandomExplorer(obj_func, initial_pool=pool, max_pool_size=max_pool_size)
    real_budget = budget - init_pool_size


    props = [obj_func(mol) for mol in pool]
    reporter.writeln(f"Properties of pool: quantity {len(pool)}, min {np.min(props)}, avg {np.mean(props)}, max {np.max(props)}")
    reporter.writeln(f"Starting {objective} optimization")

    t0 = time.time()
    top_value, top_point, history = exp.run(real_budget)

    reporter.writeln("Finished run in {:.3f} minutes".format( (time.time()-t0)/60 ))
    reporter.writeln(f"Is a valid molecule: {check_validity(top_point)}")
    reporter.writeln(f"Resulting molecule: {top_point}")
    reporter.writeln(f"Top score: {obj_func(top_point)}")
    reporter.writeln(f"Minimum synthesis score over the path: {compute_min_sa_score(top_point)}")
    with open(SYN_PATH_FILE, 'wb') as f:
        pkl.dump(top_point.get_synthesis_path(), f)

    sorted_by_prop = sorted(pool, key=obj_func)[-5:]
    for opt_mol in sorted_by_prop:
        min_sa_score = compute_min_sa_score(opt_mol)
        reporter.writeln(f"Minimum synthesis score of optimal molecules: {min_sa_score}")

    vals = history['objective_vals']
    plt.title(f'Optimizing {objective} with random explorer')
    plt.plot(range(len(vals)), vals)
    plt.savefig(PLOT_FILE, format='eps', dpi=1000)
    with open(OPT_VALS_FILE, 'w') as f:
        f.write(' '.join([str(v) for v in vals]))


if __name__ == "__main__":
    reporter = get_reporter(open(EXP_LOG_FILE, 'w'))
    args = parse_args()
    # exp_settings = {'init_pool_size': args.init_pool_size, 'seed': args.seed, 'max_pool_size': args.max_pool_size
    #                 'n_steps': args.budget, 'objective': args.objective, 'dataset': args.dataset}
    exp_settings = vars(args)
    reporter.writeln(f"RandomExplorer experiment settings: objective {exp_settings['objective']}, " +
                     f"init pool of size {exp_settings['init_pool_size']}, " +
                     f"dataset {exp_settings['dataset']}, seed {exp_settings['seed']}, " +
                     f"max_pool_size {exp_settings['max_pool_size']}, budget {exp_settings['budget']}")
    explore_and_validate_synth(**exp_settings, reporter=reporter)





