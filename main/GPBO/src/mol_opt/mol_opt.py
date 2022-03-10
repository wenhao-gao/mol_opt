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
    oracle, 
    dataset: pd.DataFrame,
    minimize: bool = True,
    keep_nan=False,
    max_docking_score=0.0,
    evaluate_cheap_functions: bool = True,
    dock_kwargs: dict = None,
    process_df_kwargs: dict = None,
) -> Tuple[CachedFunction, pd.DataFrame,]:

    # # Handle various kwargs
    # if dock_kwargs is None:
    #     dock_kwargs = dict()
    if process_df_kwargs is None:
        process_df_kwargs = dict()
    items_to_calculate = dict()
    items_to_calculate['QED'] = oracle


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

    cached_objective = CachedFunction(
        objective, cache=start_cache, 
    )

    return cached_objective, dataset_processed





































# def get_base_molopt_parser():

#     parser = argparse.ArgumentParser(add_help=False)

#     parser.add_argument(
#         "--objective",
#         type=str,
#         required=True,
#         help="Objective to optimize.",
#     )
#     parser.add_argument(
#         "--dataset", type=str, default=DATASET_PATH, help="Path to dataset tsv file."
#     )
#     parser.add_argument(
#         "--max_func_calls",
#         type=int,
#         default=None,
#         help="Maximum number of calls to objective function. "
#         "Default is no explicit maximum.",
#     )
#     parser.add_argument(
#         "--output_path", type=str, required=True, help="Path to output file (json)."
#     )
#     parser.add_argument(
#         "--num_cpu",
#         type=int,
#         default=-1,
#         help="Number of CPUs to use for docking, etc.",
#     )
#     parser.add_argument(
#         "--maximize",
#         action="store_true",
#         help="Flag to maximize function (default is to minimize it).",
#     )
#     parser.add_argument(
#         "--extra_output_path",
#         type=str,
#         default=None,
#         help="Optional path to save extra outputs/logs.",
#     )

#     return parser



