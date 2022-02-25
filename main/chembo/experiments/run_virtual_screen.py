"""
Run a virtual screen baseline:
starting from an initial pool, we randomly sample the next point
from the rest of the dataset, instead of synthesizing it from that pool.
This simulates a situation of virtual screening and doesn't account for
the cost of discovery of new compounds.
"""

import numpy as np
from argparse import ArgumentParser
from mols.mol_functions import get_objective_by_name
from datasets.loaders import MolSampler
from mols.mol_functions import get_objective_by_name


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', default='chembl', type=str,
                        help='dataset: chembl or zinc250')
    parser.add_argument('-s', '--seed', default=42, type=int,
                        help='sampling seed for the dataset')
    parser.add_argument('-o', '--objective', default='qed', type=str,
                        help='which objective function to use: qed or logp')
    parser.add_argument('-b', '--budget', default=100, type=int,
                        help='computational budget (# of function `evaluations`)')
    parser.add_argument('-i', '--init_pool_size', default=20, type=int,
                        help='size of initial pool')
    parser.add_argument('--num_repl', default=1, type=int,
                        help='number of replications of virtual screening')
    return parser.parse_args()


def run_screen(init_pool_size, seed, budget, objective, dataset, iter_num):
    obj_func = get_objective_by_name(objective)
    sampler = MolSampler(dataset, sampling_seed=seed+iter_num)
    pool = sampler(init_pool_size)
    real_budget = budget - init_pool_size
    opt_val = max([obj_func(mol) for mol in pool])
    for i in range(real_budget):
        # pick a new point randomly
        new_point = sampler(1)[0]
        opt_val = max(obj_func(new_point), opt_val)
        pool.append(new_point)
    print("Optimal value: {:.3f}".format(opt_val))
    return opt_val


if __name__ == "__main__":
    args = parse_args()
    exp_settings = vars(args)
    num_repl = exp_settings.pop('num_repl')
    opt_vals = []
    for iter_num in range(num_repl):
        opt_vals.append(run_screen(**exp_settings, iter_num=iter_num))
    print("Average {} value with virtual screening: {:.3} +- std {:.3}"\
        .format(exp_settings['objective'], np.mean(opt_vals), np.std(opt_vals)))
