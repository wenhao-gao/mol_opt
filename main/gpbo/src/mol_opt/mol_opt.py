from typing import Tuple

import numpy as np
import pandas as pd

# from dockstring_data import DATASET_PATH, process_dataframe
from dockstring_data import process_dataframe
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
# DOCKING_TARGETS = set(dockstring.list_all_target_names())
# EXPENSIVE_FUNCTIONS = set(DOCKING_TARGETS)
GUACAMOL_FUNCTIONS = set(guacamol_funcs.keys())
CHEAP_FN_DICT = {
    FNAME_LOGP: logP_fn,
    FNAME_QED: qed_fn,
    FNAME_MOLWT: mol_wt_fn,
    FNAME_PEN_LOGP: plogP_fn,
}
CHEAP_FUNCTIONS = set(CHEAP_FN_DICT.keys()) | GUACAMOL_FUNCTIONS


def get_cached_objective_and_dataframe(
    oracle, 
    dataset: pd.DataFrame,
    minimize: bool = True,
    keep_nan=False,
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



