import argparse
import json
import logging
import pickle
import random
import functools
import gpytorch

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import rdMolDescriptors
from gp import (
    TanimotoGP,
    fit_gp_hyperparameters,
)
from fingerprints import smiles_to_fp_array
from bo import acquisition_funcs, gp_bo

from mol_opt import get_base_molopt_parser, get_cached_objective_and_dataframe


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--n_train_gp_best",
        type=int,
        default=2000,
        help="Number of top-scoring training points to use for GP.",
    )
    parser.add_argument(
        "--n_train_gp_rand",
        type=int,
        default=3000,
        help="Number of random training points to use for GP.",
    )
    parser.add_argument(
        "--max_bo_iter",
        type=int,
        default=10000,
        help="Maximum number of iterations of BO.",
    )
    parser.add_argument(
        "--bo_batch_size", type=int, default=1, help="Batch size for BO."
    )
    parser.add_argument(
        "--ga_max_generations",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--ga_offspring_size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--ga_mutation_rate",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--ga_pop_params",
        type=int,
        nargs=3,
        default=[250, 250, 1000],
        help="Num best/carryover/max num for GA starting population.",
    )
    parser.add_argument(
        "--fp_radius", type=int, default=2, help="Morgan fingerprint radius."
    )
    parser.add_argument(
        "--fp_nbits", type=int, default=1024, help="Morgan fingerprint nBits."
    )
    parser.add_argument(
        "--no_log_debug",
        dest="log_debug",
        action="store_false",
        help="Flag to not log debug level messages.",
    )
    return parser


def get_trained_gp(
    X_train,
    y_train,
):

    # Fit model using type 2 maximum likelihood
    model = TanimotoGP(
        train_x=torch.as_tensor(X_train), train_y=torch.as_tensor(y_train)
    )
    fit_gp_hyperparameters(model)
    return model


if __name__ == "__main__":

    # Ensure float64 for good accuracy
    torch.set_default_dtype(torch.float64)
    NP_DTYPE = np.float64
    # np_dtype = np.float32

    # Arguments
    parser = argparse.ArgumentParser(parents=[get_parser(), get_base_molopt_parser()])
    args = parser.parse_args()

    # Load dataset
    dataset = pd.read_csv(args.dataset, sep="\t", header=0)

    # Get function to be optimized
    opt_func, df_processed = get_cached_objective_and_dataframe(
        objective_name=args.objective,
        dataset=dataset,
        minimize=not args.maximize,
        keep_nan=False,
        max_docking_score=0.0,
        dock_kwargs=dict(
            num_cpus=args.num_cpu,
        ),
    )
    dataset_smiles = set(map(str, df_processed.smiles))

    # Functions to do retraining
    def get_inducing_indices(y: np.array):
        argsort = np.argsort(-y)  # Biggest first
        best_idxs = list(argsort[: args.n_train_gp_best])
        remaining_idxs = list(argsort[args.n_train_gp_best :])
        if len(remaining_idxs) <= args.n_train_gp_rand:
            rand_idxs = remaining_idxs
        else:
            rand_idxs = random.sample(remaining_idxs, k=args.n_train_gp_rand)
        return sorted(best_idxs + rand_idxs)

    def refit_gp_change_subset(bo_iter: int, gp_model: TanimotoGP, bo_state_dict: dict):
        gp_model.train()
        x = gp_model.train_inputs[0]
        y = gp_model.train_targets.detach().cpu().numpy()
        idxs = get_inducing_indices(y)
        gp_model.set_train_data(
            inputs=x[idxs].clone(),
            targets=gp_model.train_targets[idxs].clone(),
            strict=False,
        )
        gp_model.eval()

    # Create data; fit exact GP
    fingerprint_func = functools.partial(
        rdMolDescriptors.GetMorganFingerprintAsBitVect,
        radius=args.fp_radius,
        nBits=args.fp_nbits,
    )
    my_smiles_to_fp_array = functools.partial(
        smiles_to_fp_array, fingerprint_func=fingerprint_func
    )
    all_smiles = list(dataset_smiles)
    x_train = np.stack([my_smiles_to_fp_array(s) for s in all_smiles]).astype(NP_DTYPE)
    y_train = np.asarray(opt_func(all_smiles, batch=True)).astype(NP_DTYPE)
    ind_idx_start = get_inducing_indices(y_train)
    x_train = torch.as_tensor(x_train)
    y_train = torch.as_tensor(y_train)
    gp_model = gp_model_exact = get_trained_gp(
        x_train[ind_idx_start], y_train[ind_idx_start]
    )

    # Decide on acquisition function
    def acq_f_of_time(bo_iter, bo_state_dict):
        # Beta log-uniform between ~0.3 and ~30
        # beta = 10 ** (x ~ Uniform(-0.5, 1.5))
        beta_curr = 10 ** float(np.random.uniform(-0.5, 1.5))
        gp_bo.logger.debug(f"Acquisition UCB beta set to {beta_curr:.2e}")
        return functools.partial(
            acquisition_funcs.upper_confidence_bound,
            beta=beta_curr ** 2,  # due to different conventions of what beta is in UCB
        )

    # Run GP-BO
    if args.log_debug:
        gp_bo.logger.setLevel(logging.DEBUG)
    else:
        gp_bo.logger.setLevel(logging.INFO)
    gp_bo.logger.debug(f"GP hparams: {gp_model.hparam_dict}")
    with gpytorch.settings.sgpr_diagonal_correction(False):
        bo_res = gp_bo.gp_bo_loop(
            gp_model=gp_model,
            scoring_function=opt_func,
            smiles_to_np_fingerprint=my_smiles_to_fp_array,
            acq_func_of_time=acq_f_of_time,
            max_bo_iter=args.max_bo_iter,
            bo_batch_size=args.bo_batch_size,
            gp_train_smiles=all_smiles,
            smiles_pool=dataset_smiles,
            max_func_calls=args.max_func_calls,
            log_ga_smiles=True,
            refit_gp_func=refit_gp_change_subset,
            numpy_dtype=NP_DTYPE,
            # GA parameters
            ga_max_generations=args.ga_max_generations,
            ga_offspring_size=args.ga_offspring_size,
            ga_mutation_rate=args.ga_mutation_rate,
            ga_num_cpu=args.num_cpu,
            ga_pool_num_best=args.ga_pop_params[0],
            ga_pool_num_carryover=args.ga_pop_params[1],
            max_ga_start_population_size=args.ga_pop_params[2],
        )

    # Format results by providing new SMILES + scores
    new_smiles = [r["smiles"] for r in bo_res[0] if r["smiles"] not in dataset_smiles]
    new_smiles_scores = [opt_func(s) for s in new_smiles]
    new_smiles_raw_info = [opt_func.cache[s] for s in new_smiles]
    json_res = dict(
        gp_params=gp_model.hparam_dict,
        new_smiles=new_smiles,
        scores=new_smiles_scores,
        raw_scores=new_smiles_raw_info,
    )

    # Save results
    with open(args.output_path, "w") as f:
        json.dump(json_res, f, indent=2)
    if args.extra_output_path is not None:
        with open(args.extra_output_path, "wb") as f:
            pickle.dump(bo_res, f)
