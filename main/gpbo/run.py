import argparse
import json
import logging
import pickle
import random
import functools, os 
import yaml
from tdc import Oracle
import sys
from random import shuffle 
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path_here, 'src'))
sys.path.append('.')
from main.optimizer import BaseOptimizer


import numpy as np
import pandas as pd
import torch
from rdkit.Chem import rdMolDescriptors
import gpytorch
from gp import (
    TanimotoGP,
    fit_gp_hyperparameters,
)
from fingerprints import smiles_to_fp_array
from bo import acquisition_funcs, gp_bo

from mol_opt.mol_opt import get_cached_objective_and_dataframe


def get_trained_gp(X_train, y_train,):

    # Fit model using type 2 maximum likelihood
    model = TanimotoGP(
        train_x=torch.as_tensor(X_train), train_y=torch.as_tensor(y_train)
    )
    fit_gp_hyperparameters(model)
    return model

class GPBO_optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "gp_bo"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        # Ensure float64 for good accuracy
        torch.set_default_dtype(torch.float64)
        NP_DTYPE = np.float64
        # np_dtype = np.float32

        # Load dataset   args.dataset: DATASET_PATH
        dataset = pd.read_csv(config['dataset'], sep="\t", header=0)
        # import ipdb; ipdb.set_trace()

        # Get function to be optimized
        opt_func, df_processed = get_cached_objective_and_dataframe(
            oracle = self.oracle, 
            dataset=dataset,
            minimize=not config['maximize'],
            keep_nan=False,
        )

        dataset_smiles = set(map(str, df_processed.smiles))

        # Functions to do retraining
        def get_inducing_indices(y):
            """
            To reduce the training cost of GP model, we only select 
            top-n_train_gp_best and n_train_gp_rand random samples from data.
            """
            argsort = np.argsort(-y)  # Biggest first
            best_idxs = list(argsort[: config['n_train_gp_best']])
            remaining_idxs = list(argsort[config['n_train_gp_best'] :])
            if len(remaining_idxs) <= config['n_train_gp_rand']:
                rand_idxs = remaining_idxs
            else:
                rand_idxs = random.sample(remaining_idxs, k=config['n_train_gp_rand'])
            return sorted(best_idxs + rand_idxs)

        def refit_gp_change_subset(bo_iter, gp_model, bo_state_dict):
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
            radius=config['fp_radius'],
            nBits=config['fp_nbits'],
        )
        my_smiles_to_fp_array = functools.partial(
            smiles_to_fp_array, fingerprint_func=fingerprint_func
        )
        all_smiles = list(dataset_smiles)
        # all_smiles = self.all_smiles
        shuffle(all_smiles)
        # all_smiles = all_smiles[:config['start_train_num']]
        x_train = np.stack([my_smiles_to_fp_array(s) for s in all_smiles]).astype(NP_DTYPE)
        # values = opt_func(all_smiles, batch= True)
        values = self.oracle(all_smiles)
        # import ipdb; ipdb.set_trace()
        # print(values) 
        y_train = np.asarray(values).astype(NP_DTYPE)
        # y_train = np.asarray(opt_func(all_smiles, batch=True)).astype(NP_DTYPE)

        ind_idx_start = get_inducing_indices(y_train)
        x_train = torch.as_tensor(x_train)
        y_train = torch.as_tensor(y_train)
        # import ipdb; ipdb.set_trace()
        gp_model = get_trained_gp(
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


        assert config['ga_pop_params'][0] == 250 
        # Run GP-BO
        gp_bo.logger.setLevel(logging.INFO)
        gp_bo.logger.debug(f"GP hparams: {gp_model.hparam_dict}")
        with gpytorch.settings.sgpr_diagonal_correction(False):
            bo_res = gp_bo.gp_bo_loop(
                gp_model=gp_model,
                scoring_function=self.oracle,
                smiles_to_np_fingerprint=my_smiles_to_fp_array,
                acq_func_of_time=acq_f_of_time,
                max_bo_iter=config['max_bo_iter'],
                bo_batch_size=config['bo_batch_size'],
                gp_train_smiles=all_smiles,
                smiles_pool=dataset_smiles,
                max_func_calls=config['max_n_oracles'],
                log_ga_smiles=True,
                refit_gp_func=refit_gp_change_subset,
                numpy_dtype=NP_DTYPE,
                # GA parameters
                ga_max_generations=config['ga_max_generations'],
                ga_offspring_size=config['ga_offspring_size'],
                ga_mutation_rate=config['ga_mutation_rate'],
                ga_num_cpu=config['num_cpu'],
                ga_pool_num_best=config['ga_pop_params'][0],
                ga_pool_num_carryover=config['ga_pop_params'][1],
                max_ga_start_population_size=config['ga_pop_params'][2],
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=500)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

    path_here = os.path.dirname(os.path.realpath(__file__))

    if args.output_dir is None:
        args.output_dir = os.path.join(path_here, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    for oracle_name in args.oracles:

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_here, args.config_default)))

        if args.task == "tune":
            try:
                config_tune = yaml.safe_load(open(args.config_tune))
            except:
                config_tune = yaml.safe_load(open(os.path.join(path_here, args.config_tune)))

        oracle = Oracle(name = oracle_name)
        optimizer = GPBO_optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main() 
