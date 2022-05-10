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
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
from tdc import Oracle
rdBase.DisableLog('rdApp.error')

import main.graph_ga.crossover as co, main.graph_ga.mutate as mu
from main.optimizer import BaseOptimizer


MINIMUM = 1e-10

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
    population_scores = [s + MINIMUM for s in population_scores]
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

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "graph_ga"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        pool = joblib.Parallel(n_jobs=self.n_jobs)
        
        if self.smi_file is not None:
            # Exploitation run
            starting_population = self.all_smiles[:config["population_size"]]
        else:
            # Exploration run
            starting_population = np.random.choice(self.all_smiles, config["population_size"])

        # select initial population
        # population_smiles = heapq.nlargest(config["population_size"], starting_population, key=oracle)
        population_smiles = starting_population
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])

        patience = 0

        while True:

            # new_population
            mating_pool = make_mating_pool(population_mol, population_scores, config["population_size"])
            offspring_mol = pool(delayed(reproduce)(mating_pool, config["mutation_rate"]) for _ in range(config["offspring_size"]))

            # add new_population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)

            # stats
            old_scores = population_scores
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # early stopping
            if population_scores == old_scores:
                patience += 1
                if patience >= self.args.patience:
                    self.log_intermediate(finish=True)
                    break
            else:
                patience = 0
                
            if self.finish:
                break

