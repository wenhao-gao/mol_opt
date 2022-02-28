import argparse
import functools
import math
from typing import Tuple

import numpy as np
import pandas as pd
import dockstring

from dockstring_data import DATASET_PATH, process_dataframe
from mol_funcs.dockstring_funcs import safe_dock_function
from mol_funcs.simple_funcs import (
    logP as logP_fn,
    QED as qed_fn,
    guacamol_funcs,
    molecular_weight as mol_wt_fn,
    penalized_logP as plogP_fn,
)
from function_utils import CachedFunction

# Potential objective functions
FNAME_LOGP = "logP"
FNAME_PEN_LOGP = "plogP"
FNAME_QED = "QED"
FNAME_MOLWT = "MolWt"
DOCKING_TARGETS = set(dockstring.list_all_target_names())
EXPENSIVE_FUNCTIONS = set(DOCKING_TARGETS)
GUACAMOL_FUNCTIONS = set(guacamol_funcs.keys())
CHEAP_FN_DICT = {
    FNAME_LOGP: logP_fn,
    FNAME_QED: qed_fn,
    FNAME_MOLWT: mol_wt_fn,
    FNAME_PEN_LOGP: plogP_fn,
}
CHEAP_FUNCTIONS = set(CHEAP_FN_DICT.keys()) | GUACAMOL_FUNCTIONS
SUPPORTED_OBJECTIVES = (
    list(EXPENSIVE_FUNCTIONS)
    + list(CHEAP_FUNCTIONS)
    + [
        "PPAR-all",  # promiscuous binding to PPAR A, D, G
        "JAK2-not-LCK-v1",  # bind to JAK2 and not LCK
    ]
)

NAN_REPLACE_VALUES = dict(
    [(target_name, 0.0) for target_name in DOCKING_TARGETS]
)  # What values should NaNs be replaced with?


def _sum_scalarization(score_dict, mult=1.0):
    return mult * sum(score_dict.values())


def _max_scalarization(score_dict):
    return max(score_dict.values())


def _add_molwt_pen_v1(score_dict, prev_scoring_fn):
    score_dict_no_mol = dict(score_dict)
    mol_wt = score_dict_no_mol.pop(FNAME_MOLWT)
    prev_score = prev_scoring_fn(score_dict_no_mol)
    penalty = 1e-2 * max(mol_wt - 500.0, 0.0)
    return prev_score + penalty


def _add_qed_pen_v1(score_dict, prev_scoring_fn):

    # Penalty is multiplying by QED
    score_dict_copy = dict(score_dict)
    qed = score_dict_copy.pop(FNAME_QED)
    prev_score = prev_scoring_fn(score_dict_copy)
    return qed * prev_score


def _add_qed_pen_v2(score_dict, prev_scoring_fn):

    # Penalty is multiplying by QED
    score_dict_copy = dict(score_dict)
    qed = score_dict_copy.pop(FNAME_QED)
    prev_score = prev_scoring_fn(score_dict_copy)
    qed_pen = min(qed / 0.6, 1.0)
    return qed_pen * prev_score


def _add_qed_pen_v3(score_dict, prev_scoring_fn):

    # Penalty + 10 * (1-QED)
    score_dict_copy = dict(score_dict)
    qed = score_dict_copy.pop(FNAME_QED)
    prev_score = prev_scoring_fn(score_dict_copy)
    return prev_score + 10.0 * (1.0 - qed)


def _add_qed_pen_v4(score_dict, prev_scoring_fn):

    # Penalty + 30 * (1-QED)
    score_dict_copy = dict(score_dict)
    qed = score_dict_copy.pop(FNAME_QED)
    prev_score = prev_scoring_fn(score_dict_copy)
    return prev_score + 30.0 * (1.0 - qed)


def _jak2_not_lck_v1(score_dict):

    # Selectivity towards JAK2 and not LCK, with a penalty for not binding to JAK2 well
    jak2 = score_dict["JAK2"]
    lck = score_dict["LCK"]
    bind_to_jak2_coeff = 1 / (1 + math.exp(jak2 + 6))  # sigmoid centered around -6
    score_diff = jak2 - lck
    return score_diff - 5.0 * bind_to_jak2_coeff


def _jak2_not_lck_v2(score_dict):

    # Selectivity towards JAK2, with a penalty for binding well to LCK
    jak2 = score_dict["JAK2"]
    lck = score_dict["LCK"]
    lck_cutoff = -8.1  # median for LCK
    return jak2 - min(0, lck - lck_cutoff)


def _get_safe_dock_function(target_name, **dock_kwargs):
    return functools.partial(safe_dock_function, target_name=target_name, **dock_kwargs)


def get_cached_objective_and_dataframe(
    objective_name: str,
    dataset: pd.DataFrame,
    minimize: bool = True,
    keep_nan=False,
    max_docking_score=0.0,
    evaluate_cheap_functions: bool = True,
    dock_kwargs: dict = None,
    process_df_kwargs: dict = None,
) -> Tuple[CachedFunction, pd.DataFrame,]:

    # Handle various kwargs
    if dock_kwargs is None:
        dock_kwargs = dict()
    if process_df_kwargs is None:
        process_df_kwargs = dict()

    # Does the objective name have a penalty at the end?
    mol_wt_pen_v1 = "_mol-wt-pen-v1"
    qed_pen_v1 = "_qed-pen-v1"
    qed_pen_v2 = "_qed-pen-v2"
    qed_pen_v3 = "_qed-pen-v3"
    qed_pen_v4 = "_qed-pen-v4"
    if objective_name.endswith(mol_wt_pen_v1):
        objective_suffix = mol_wt_pen_v1
        objective_name = objective_name[: objective_name.index(mol_wt_pen_v1)]
    elif objective_name.endswith(qed_pen_v1):
        objective_suffix = qed_pen_v1
        objective_name = objective_name[: objective_name.index(qed_pen_v1)]
    elif objective_name.endswith(qed_pen_v2):
        objective_suffix = qed_pen_v2
        objective_name = objective_name[: objective_name.index(qed_pen_v2)]
    elif objective_name.endswith(qed_pen_v3):
        objective_suffix = qed_pen_v3
        objective_name = objective_name[: objective_name.index(qed_pen_v3)]
    elif objective_name.endswith(qed_pen_v4):
        objective_suffix = qed_pen_v4
        objective_name = objective_name[: objective_name.index(qed_pen_v4)]
    else:
        objective_suffix = None

    # Which things are calculated to make the score?
    items_to_calculate = dict()
    if objective_name in DOCKING_TARGETS:
        items_to_calculate[objective_name] = _get_safe_dock_function(
            objective_name, **dock_kwargs
        )
        scalar_fn = _sum_scalarization
    elif objective_name == "PPAR-all":
        for protein in ["PPARG", "PPARD", "PPARA"]:
            items_to_calculate[protein] = _get_safe_dock_function(
                protein, **dock_kwargs
            )
        scalar_fn = _max_scalarization
    elif objective_name == "JAK2-not-LCK-v1":
        for protein in ["JAK2", "LCK"]:
            items_to_calculate[protein] = _get_safe_dock_function(
                protein, **dock_kwargs
            )
        scalar_fn = _jak2_not_lck_v1
    elif objective_name == "JAK2-not-LCK-v2":
        for protein in ["JAK2", "LCK"]:
            items_to_calculate[protein] = _get_safe_dock_function(
                protein, **dock_kwargs
            )
        scalar_fn = _jak2_not_lck_v2
    elif objective_name in GUACAMOL_FUNCTIONS:
        items_to_calculate[objective_name] = guacamol_funcs[objective_name]
        # Guacamol functions are scaled up so variance is closer to 1
        scalar_fn = functools.partial(_sum_scalarization, mult=10.0)
    elif objective_name == FNAME_LOGP:
        items_to_calculate[objective_name] = logP_fn
        scalar_fn = _sum_scalarization
    elif objective_name == FNAME_PEN_LOGP:
        items_to_calculate[objective_name] = plogP_fn
        scalar_fn = _sum_scalarization
    elif objective_name == FNAME_QED:
        items_to_calculate[objective_name] = qed_fn

        # QED is scaled up by 10 for modelling so its variance is closer to 1
        scalar_fn = functools.partial(_sum_scalarization, mult=10.0)
    else:
        raise ValueError(objective_name)

    # Add a possible penalty
    if objective_suffix is None:
        pass
    elif objective_suffix == mol_wt_pen_v1:
        items_to_calculate[FNAME_MOLWT] = mol_wt_fn
        scalar_fn = functools.partial(_add_molwt_pen_v1, prev_scoring_fn=scalar_fn)
    elif objective_suffix == qed_pen_v1:
        items_to_calculate[FNAME_QED] = qed_fn
        scalar_fn = functools.partial(_add_qed_pen_v1, prev_scoring_fn=scalar_fn)
    elif objective_suffix == qed_pen_v2:
        items_to_calculate[FNAME_QED] = qed_fn
        scalar_fn = functools.partial(_add_qed_pen_v2, prev_scoring_fn=scalar_fn)
    elif objective_suffix == qed_pen_v3:
        items_to_calculate[FNAME_QED] = qed_fn
        scalar_fn = functools.partial(_add_qed_pen_v3, prev_scoring_fn=scalar_fn)
    elif objective_suffix == qed_pen_v4:
        items_to_calculate[FNAME_QED] = qed_fn
        scalar_fn = functools.partial(_add_qed_pen_v4, prev_scoring_fn=scalar_fn)

    # Get processed dataset for necessary targets
    targets_to_process = set(items_to_calculate.keys())
    if evaluate_cheap_functions:
        targets_to_process -= set(CHEAP_FUNCTIONS)
    targets_to_process = list(targets_to_process)
    dataset_processed = process_dataframe(
        dataset, targets=targets_to_process, **process_df_kwargs
    )

    # Potentially evaluate cheap targets
    for fname, f in items_to_calculate.items():
        if fname not in dataset_processed.columns:
            assert fname in CHEAP_FUNCTIONS, fname
            dataset_processed[fname] = dataset_processed.smiles.map(f)

    # Now produce start cache with all values known
    start_cache = dict()
    for _, row in dataset_processed.iterrows():
        start_cache[row.smiles] = {
            fname: row[fname] for fname in items_to_calculate.keys()
        }

    # Define final objective!
    def objective(smiles: str) -> dict:
        return {fname: f(smiles) for fname, f in items_to_calculate.items()}

    # Define the transform that scalarized this function (into something to be minimized)
    def scalarize(score_dict):

        # If there are NaNs then the score is undefined
        if keep_nan and any(np.isnan(score) for score in score_dict.values()):
            return math.nan

        # Adjust the scores to handle NaNs, positive docking scores, etc
        adjusted_score_dict = dict()
        for name, score in score_dict.items():
            new_score = score

            if np.isnan(score):
                new_score = NAN_REPLACE_VALUES[name]

            if name in DOCKING_TARGETS:
                new_score = min(new_score, max_docking_score)
            adjusted_score_dict[name] = new_score

        # Final scalarization
        return scalar_fn(adjusted_score_dict)

    # Whether to maximize or minimize the objective
    def _max_objective(v):
        return float(v)

    def _min_objective(v):
        return -float(v)

    if minimize:
        sign_fn = _min_objective
    else:
        sign_fn = _max_objective

    # Final transform
    def final_transform(score_dict):
        v = scalarize(score_dict)
        return sign_fn(v)

    cached_objective = CachedFunction(
        objective, cache=start_cache, transform=final_transform
    )

    return cached_objective, dataset_processed


def get_base_molopt_parser():

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--objective",
        type=str,
        required=True,
        help="Objective to optimize.",
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_PATH, help="Path to dataset tsv file."
    )
    parser.add_argument(
        "--max_func_calls",
        type=int,
        default=None,
        help="Maximum number of calls to objective function. "
        "Default is no explicit maximum.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output file (json)."
    )
    parser.add_argument(
        "--num_cpu",
        type=int,
        default=-1,
        help="Number of CPUs to use for docking, etc.",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Flag to maximize function (default is to minimize it).",
    )
    parser.add_argument(
        "--extra_output_path",
        type=str,
        default=None,
        help="Optional path to save extra outputs/logs.",
    )

    return parser
