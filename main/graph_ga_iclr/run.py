"""

python main/graph_ga/run.py --smiles_file data/guacamol_v1_all.smiles --output_file main/graph_ga/result.json --max_func_calls=1490

"""


from __future__ import print_function

import argparse
import heapq
import json
import os
import random
from time import time
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from joblib import delayed
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

import crossover as co, mutate as mu


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
#     # scores -> probs
#     sum_scores = sum(population_scores)
#     population_probs = [p / sum_scores for p in population_scores]
#     mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
#     return mating_pool

    # My modification: choose based on uniformly sampling the top N molecules for different N values
    # Sort population
    population_tuples = list(zip(population_scores, population_mol))
    population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)
    population_mol = [t[1] for t in population_tuples]
    population_scores = [t[0] for t in population_tuples]
#     print(population_scores)
    N_list = [5, 10, 25, 100, 250, 1000, len(population_mol)]
    mating_pool = []
    for _ in range(offspring_size):
        N = random.choice(N_list)
        mol = random.choice(population_mol[:N])
        mating_pool.append(mol)
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

    # My modification: copy mols to prevent problematic threading errors
    parent_a = Chem.MolFromSmiles(Chem.MolToSmiles(parent_a))
    parent_b = Chem.MolFromSmiles(Chem.MolToSmiles(parent_b))

    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


def score_mol(mol, score_fn, known_value_dict):
    smiles = Chem.MolToSmiles(mol)
    if smiles not in known_value_dict:
        known_value_dict[smiles] = score_fn(smiles)
    return known_value_dict[smiles]


def sanitize(population_mol):
    new_population = []
    smile_set = set()
    for mol in population_mol:
        if mol is not None:
            try:
                smile = Chem.MolToSmiles(mol)
                if smile is not None and smile not in smile_set:
                    smile_set.add(smile)
                    new_population.append(mol)
            except ValueError:
                print('bad smiles')
    return new_population


# My modification: simplified version of the Goal Directed generator
def generate_optimized_molecules(
    scoring_function,
    start_known_smiles: dict,
    starting_population: List[str],
    n_generation: int = 1000,
    offspring_size: int = 1000,
    mutation_rate: float=1e-2,
    population_size: int = 1000,
    max_total_func_calls: int = 1000
) -> List[str]:

    # Accurately track function evaluations by storing all known scores so far
    f_cache = dict(start_known_smiles)

    # select initial population
    print("Scoring initial population...")
    population_smiles = list(starting_population)
    population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
    population_scores = [score_mol(m, scoring_function, f_cache) for m in population_mol]
    print("Initial population scoring complete!")
    print(f"Max starting score: {max(population_scores)}")

    # evolution: go go go!!
    t0 = time()

    patience = 0

    for generation in range(n_generation):

        # new_population
        mating_pool = make_mating_pool(population_mol, population_scores, population_size)
        offspring_mol = [reproduce(mating_pool, mutation_rate) for _ in range(offspring_size)]

        # add new_population
        population_mol += offspring_mol
        population_mol = sanitize(population_mol)

        # stats
        gen_time = time() - t0
        mol_sec = population_size / gen_time
        t0 = time()

        old_scores = population_scores
        population_scores = [score_mol(m, scoring_function, f_cache) for m in population_mol]
        population_tuples = list(zip(population_scores, population_mol))
        population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:population_size]
        population_mol = [t[1] for t in population_tuples]
        population_scores = [t[0] for t in population_tuples]


        print(f'{generation} | '
              f'oracle call: {len(f_cache)} | '
              f'max: {np.max(population_scores):.3f} | '
              f'avg: {np.mean(population_scores):.3f} | '
              f'min: {np.min(population_scores):.3f} | '
              f'std: {np.std(population_scores):.3f} | '
              f'sum: {np.sum(population_scores):.3f} | '
              f'{gen_time:.2f} sec/gen | '
              f'{mol_sec:.2f} mol/sec | '
        )

        # Potential early stopping
        if len(f_cache) > max_total_func_calls:
            print("Max function calls hit, aborting")
            break

    return [Chem.MolToSmiles(m) for m in population_mol], f_cache


from tdc import Oracle
from tdc import Evaluator
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
qed = Oracle(name = 'qed')
from sa import sa
def oracle(smiles):
	scores = [qed(smiles), sa(smiles), jnk(smiles), gsk(smiles)]
	return np.mean(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_file', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', type=str, required=True)

    parser.add_argument('--population_size', type=int, default=1000)  # large to ensure diversity
    parser.add_argument('--offspring_size', type=int, default=10)  # small to help sample efficiency
    parser.add_argument('--mutation_rate', type=float, default=0.01)  # same as standard guacamol
    parser.add_argument('--generations', type=int, default=10_000) # large because we are function call limited
    parser.add_argument('--max_func_calls', type=int, default=14_950) # match DST eval setting, with small error margin


    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.smiles_file) as f:
        start_smiles = set([l.strip() for l in f.readlines()])

    # Run GA
    print("begin running graph-GA")
    end_population, all_func_evals = generate_optimized_molecules(
        scoring_function = oracle,
        start_known_smiles = dict(),
        starting_population=list(start_smiles),
        n_generation=args.generations,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        population_size=args.population_size,
        max_total_func_calls=args.max_func_calls
    )

    # Evaluate 
    new_score_tuples = [(v, k) for k, v in all_func_evals.items() if k not in start_smiles]  # scores of new molecules
    new_score_tuples.sort(reverse=True)
    top100_mols = [(k, v) for (v, k) in new_score_tuples[:100]]
    diversity = Evaluator(name = 'Diversity')
    div = diversity([t[0] for t in top100_mols])
    output = dict(
        top_mols=top100_mols,
        AST=np.average([t[1] for t in top100_mols]),
        diversity=div,
        all_func_evals=dict(all_func_evals),
    )
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
