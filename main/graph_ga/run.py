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

import crossover as co, mutate as mu
from main.optimizer import BaseOptimizer


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs 
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, config=None, args=None, smi_file=None, n_jobs=-1):
        super().__init__(config, args, smi_file, n_jobs)

    def _optimize(self, oracle, config):
        
        starting_population = np.random.choice(self.all_smiles, config["population_size"])

        # select initial population
        population_smiles = heapq.nlargest(config["population_size"], starting_population, key=oracle)
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_scores = self.score_mol(oracle, population_mol)

        patience = 0

        for generation in range(config["max_generations"]):

            # new_population
            mating_pool = make_mating_pool(population_mol, population_scores, config["population_size"])
            offspring_mol = self.pool(delayed(reproduce)(mating_pool, config["mutation_rate"]) for _ in range(config["offspring_size"]))

            # add new_population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)

            # stats
            old_scores = population_scores
            population_scores = self.score_mol(oracle, population_mol)
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # early stopping
            if population_scores == old_scores:
                patience += 1
                if patience >= config["patience"]:
                    break
            else:
                patience = 0
                
            self.log_intermediate(population_mol, population_scores)
            
            if len(self.mol_buffer) >= config["max_n_oracles"]:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_file', default=None)
    parser.add_argument('--config', default='hparams_default.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--offspring_size', type=int, default=200)
    parser.add_argument('--mutation_rate', type=float, default=0.01)
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--task', type=str, default="single", choices=["tune", "single", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

    path_here = os.path.dirname(os.path.realpath(__file__))

    if args.output_dir is None:
        args.output_dir = path_here
    elif not os.path.exist(args.output_dir):
        os.mkdir(args.output_dir)

    try:
        config = yaml.safe_load(open(args.config))
    except:
        config = yaml.safe_load(open(os.path.join(args.output_dir, args.config)))
        
    oracle = Oracle(name = "QED")

    optimizer = GB_GA_Optimizer(None, smi_file=None, n_jobs=args.n_jobs)

    if args.task == "single":
        optimizer.optimize(oracle=oracle, config=config)
    elif args.task == "tune":
        optimizer.hparam_tune(oracle=oracle, hparam_space=config, count=args.count)
    elif args.task == "production":
        optimizer.optimize(oracle=oracle, config=config)

    # json_file_path = os.path.join(args.output_dir, 'optimization_results.yaml')
    # optimizer.


if __name__ == "__main__":
    main()

