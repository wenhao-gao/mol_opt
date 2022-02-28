import random
from typing import List, Union
import logging

import numpy as np
from rdkit import Chem, RDLogger
import joblib

from function_utils import CachedFunction
from . import crossover as co, mutate as mu


# Logger with standard handler
ga_logger = logging.getLogger("graph_ga")
if len(ga_logger.handlers) == 0:
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    ga_logger.addHandler(ch)


rd_logger = RDLogger.logger()


def make_mating_pool(population_mol: List, population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights

    Args:
        population_mol: list of population
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return

    Returns: a list of RDKit Mol (probably not unique)

    """
    # scores -> probs
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(
        population_mol, p=population_probs, size=offspring_size, replace=True
    )
    return mating_pool


def reproduce(mating_pool, mutation_rate, crossover_kwargs: dict = None):
    """

    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation

    Returns:

    """

    # turn off rdkit logging
    rd_logger.setLevel(RDLogger.CRITICAL)

    parent_a = Chem.MolFromSmiles(random.choice(mating_pool))
    parent_b = Chem.MolFromSmiles(random.choice(mating_pool))
    if crossover_kwargs is None:
        crossover_kwargs = dict()
    new_child = co.crossover(parent_a, parent_b, **crossover_kwargs)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


def score_mol(mol, score_fn):
    return score_fn(Chem.MolToSmiles(mol))


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
                print("bad smiles")
    return new_population


def sanitize_smiles(population_smiles):
    new_population = []
    smile_set = set()
    for smiles in population_smiles:
        if Chem.MolFromSmiles(smiles) is not None and smiles not in smile_set:
            new_population.append(smiles)
    return new_population


def run_ga_maximization(
    starting_population_smiles: list,
    scoring_function: Union[callable, CachedFunction],
    max_generations: int,
    population_size: int,
    offspring_size: int,
    mutation_rate: float,
    patience: int = None,
    max_func_calls: int = None,
    min_func_val: float = 0.0,
    crossover_kwargs: dict = None,
    y_transform: callable = None,  # only used if scoring function is not a cached function already
    num_cpu: int = 1,
):
    """
    Runs a genetic algorithm to MAXIMIZE a score function.

    It does accurate budgeting by tracking which function calls have been made already.
    Note that the function will always be called with canonical smiles.
    """
    ga_logger.info("Starting GA maximization...")
    if crossover_kwargs is None:
        crossover_kwargs = dict()

    # Create the cached function
    if not isinstance(scoring_function, CachedFunction):
        scoring_function = CachedFunction(scoring_function, transform=y_transform)
    start_cache = dict(scoring_function.cache)
    start_cache_size = len(start_cache)
    ga_logger.debug(f"Starting cache made, has size {start_cache_size}")

    # Budget will just be measured by the cache size,
    # But since the starting cache is free update the budget
    # to account for this
    if max_func_calls is not None:
        max_func_calls += start_cache_size

    # Init population and scores
    population_smiles = list(set(starting_population_smiles))
    num_start_eval = len(set(population_smiles) - set(start_cache.keys()))
    ga_logger.debug(
        "Scoring initial population. "
        f"{num_start_eval}/{len(population_smiles)} "
        f"({num_start_eval/len(population_smiles)*100:.1f}%) "
        "not in start cache and will need evaluation."
    )
    del num_start_eval  # not needed later
    population_scores = scoring_function(population_smiles, batch=True)
    queried_smiles = list(population_smiles)
    ga_logger.debug(
        f"Initial population scoring done. Pop size={len(population_smiles)}, Max={max(population_scores)}"
    )

    # Run GA
    early_stop = False
    reached_budget = False
    num_no_change_gen = 0
    gen_info = []
    with joblib.Parallel(n_jobs=num_cpu) as parallel:
        for generation in range(max_generations):
            ga_logger.info(f"Start generation {generation}")

            # Make mating pool
            # We use an adjusted score to weight each sample in the mating pool
            bottom_score = np.min(population_scores)
            if min_func_val is not None:
                bottom_score = min(min_func_val, bottom_score)
            mating_pool = make_mating_pool(
                population_smiles,
                np.asarray(population_scores) - bottom_score,
                population_size,
            )

            # Create offspring in parallel to be more efficient
            offspring_mol = parallel(
                joblib.delayed(reproduce)(
                    mating_pool, mutation_rate, **crossover_kwargs
                )
                for _ in range(offspring_size)
            )
            # offspring_mol = [
            #     reproduce(mating_pool, mutation_rate, **crossover_kwargs)
            #     for _ in range(offspring_size)
            # ]
            ga_logger.debug(f"\t{len(offspring_mol)} created")

            # Convert offspring to SMILES and add to new population
            offspring_smiles = []
            for mol in offspring_mol:
                try:
                    if mol is not None:
                        offspring_smiles.append(Chem.MolToSmiles(mol))
                except ValueError:
                    pass
            ga_logger.debug(
                f"\t{len(offspring_smiles)}/{len(offspring_mol)} converted to SMILES"
            )
            population_and_offspring_smiles = population_smiles + offspring_smiles
            population_and_offspring_smiles = list(set(population_and_offspring_smiles))
            ga_logger.debug(
                f"\tPopulation sanitized, now contains {len(population_and_offspring_smiles)} members."
            )

            # Find out scores, but don't go over budget
            old_scores = population_scores
            population_smiles = []
            planned_cached_smiles = set(
                scoring_function.cache.keys()
            )  # make a copy to not go over budget
            planned_cache_start_size = len(planned_cached_smiles)
            for smiles in population_and_offspring_smiles:
                if (
                    max_func_calls is None
                    or smiles in scoring_function.cache
                    or len(planned_cached_smiles) < max_func_calls
                ):
                    population_smiles.append(smiles)
                    planned_cached_smiles.add(smiles)
            ga_logger.debug(
                "\tDecided on which SMILES to evaluate. Plan to make "
                f"{len(planned_cached_smiles) - planned_cache_start_size} new function calls."
            )
            ga_logger.debug("\tStarting function calls...")
            population_scores = scoring_function(population_smiles, batch=True)
            queried_smiles += population_smiles
            ga_logger.debug(f"\tScoring done, best score now {max(population_scores)}.")

            # Trim population (take highest few values)
            argsort = np.argsort(-np.asarray(population_scores))[:population_size]
            population_smiles = [population_smiles[i] for i in argsort]
            population_scores = [population_scores[i] for i in argsort]
            ga_logger.debug(f"\tPopulation trimmed to size {len(population_smiles)}.")

            # Record results of generation
            gen_stats_dict = dict(
                max=np.max(population_scores),
                avg=np.mean(population_scores),
                median=np.median(population_scores),
                min=np.min(population_scores),
                std=np.std(population_scores),
                size=len(population_scores),
                num_func_eval=len(scoring_function.cache) - start_cache_size,
            )
            stats_str = " ".join(
                ["\tGen stats:\n"] + [f"{k}={v}" for k, v in gen_stats_dict.items()]
            )
            ga_logger.info(stats_str)
            gen_info.append(dict(smiles=population_smiles, **gen_stats_dict))

            # early stopping if population doesn't change
            if len(population_scores) == len(old_scores) and np.allclose(
                population_scores, old_scores
            ):

                num_no_change_gen += 1
                ga_logger.info(
                    f"\tPopulation unchanged for {num_no_change_gen} generations"
                )
                if patience is not None and num_no_change_gen > patience:
                    ga_logger.info(
                        f"\tThis exceeds patience of {patience}. Terminating GA."
                    )
                    early_stop = True
                    break
            else:
                num_no_change_gen = 0

            # early stopping if budget is reached
            if (
                max_func_calls is not None
                and len(scoring_function.cache) >= max_func_calls
            ):
                ga_logger.info(
                    f"\tBudget of {max_func_calls - start_cache_size} has been reached. Terminating..."
                )
                reached_budget = True
                break

    # Before returning, filter duplicates from the queried SMILES
    queried_smiles_set = set()
    new_queried_smiles = list()
    for s in queried_smiles:
        if s not in queried_smiles_set:
            queried_smiles_set.add(s)
            new_queried_smiles.append(s)
    queried_smiles = new_queried_smiles

    # Return values
    ga_logger.info("End of GA. Returning results.")
    return (
        queried_smiles,  # all smiles queried in order, *without* duplicates
        scoring_function.cache,  # holds function values
        (gen_info, early_stop, reached_budget),  # GA logs
    )
