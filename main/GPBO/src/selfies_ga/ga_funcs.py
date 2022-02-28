# Code largely copied from:
# https://github.com/aspuru-guzik-group/stoned-selfies/blob/main/GA_rediscover.py
import random
import functools
from typing import Union
import logging

import selfies
import numpy as np
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import RDLogger
from selfies import encoder, decoder

# Custom imports
from function_utils import CachedFunction


# Logger with standard handler
ga_logger = logging.getLogger("selfies_ga")
if len(ga_logger.handlers) == 0:
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    ga_logger.addHandler(ch)


rd_logger = RDLogger.logger()


def sanitize_smiles(smi):
    """Return a canonical smile representation of smi

    Parameters:
    smi (string) : smile string to be canonicalized

    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful
    """
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)


def mutate_selfie(selfie, max_molecules_len, write_fail_cases=False, max_fails=1000):
    """Return a mutated selfie string (only one mutation on slefie is performed)

    Mutations are done until a valid molecule is obtained
    Rules of mutation: With a 50% propbabily, either:
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
        3. Delete random SELFIE character (**note: not in original implementation**)

    Parameters:
    selfie            (string)  : SELFIE string to be mutated
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"

    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    """
    valid = False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)

    while not valid:
        fail_counter += 1

        alphabet = list(selfies.get_semantic_robust_alphabet())  # 34 SELFIE characters

        choice_ls = [1, 2, 3]  # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]

        # Insert a character in a Random Location
        if random_choice == 1:
            random_index = np.random.randint(len(chars_selfie) + 1)
            random_character = np.random.choice(alphabet, size=1)[0]

            selfie_mutated_chars = (
                chars_selfie[:random_index]
                + [random_character]
                + chars_selfie[random_index:]
            )

        # Replace a random character
        elif random_choice == 2:
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[
                    random_index + 1 :
                ]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index]
                    + [random_character]
                    + chars_selfie[random_index + 1 :]
                )

        # Delete a random character
        elif random_choice == 3:
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index + 1 :]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index] + chars_selfie[random_index + 1 :]
                )

        else:
            raise Exception("Invalid Operation trying to be performed")

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)

        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon == "":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid = False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write(
                    "Tried to mutate SELFIE: "
                    + str(sf)
                    + " To Obtain: "
                    + str(selfie_mutated)
                    + "\n"
                )
                f.close()
        finally:
            if fail_counter >= max_fails:

                # Exit to avoid infinite looping
                return None, None

    return (selfie_mutated, smiles_canon)


def get_selfie_chars(selfie):
    """Obtain a list of all selfie characters in string selfie

    Parameters:
    selfie (string) : A selfie string - representing a molecule

    Example:
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    """
    chars_selfie = []  # A list of all SELFIE sybols from string selfie
    while selfie != "":
        chars_selfie.append(selfie[selfie.find("[") : selfie.find("]") + 1])
        selfie = selfie[selfie.find("]") + 1 :]
    return chars_selfie


def run_ga_maximization(
    starting_population_smiles: list,
    scoring_function: Union[callable, CachedFunction],
    max_generations: int,
    population_size: int,
    offspring_size: int,
    patience: int = None,
    max_func_calls: int = None,
    max_mol_len: int = 100,
    y_transform: callable = None,  # only used if scoring function is not a cached function already
):
    """
    Runs a genetic algorithm to MAXIMIZE a score function.

    It does accurate budgeting by tracking which function calls have been made already.
    Note that the function will always be called with canonical smiles.
    """
    ga_logger.info("Starting GA maximization...")

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

    # Encoder to make SMILES -> SELFIES conversion quick
    sf_encoder_cached = functools.lru_cache(int(1e6))(encoder)

    # Run GA
    early_stop = False
    reached_budget = False
    num_no_change_gen = 0
    gen_info = []
    for generation in range(max_generations):
        ga_logger.info(f"Start generation {generation}")

        # Make random offspring
        offspring_smiles = []
        for _ in range(offspring_size):
            chosen_smiles = random.choice(population_smiles)
            chosen_selfie = str(sf_encoder_cached(chosen_smiles))
            _, mutated_smiles = mutate_selfie(chosen_selfie, max_mol_len)
            if mutated_smiles is not None:  # Can be None if invalid
                offspring_smiles.append(mutated_smiles)
            del chosen_smiles, chosen_selfie, mutated_smiles
        ga_logger.debug(f"\t{len(offspring_smiles)} offspring SMILES created")

        # Add to population SMILES
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
        if max_func_calls is not None and len(scoring_function.cache) >= max_func_calls:
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
