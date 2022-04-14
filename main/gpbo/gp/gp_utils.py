""" Various GP utility functions """

import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import botorch

from .tanimoto_gp import TanimotoGP


def batch_predict_mu_var_numpy(
    gp_model: ExactGP,
    x: torch.Tensor,
    batch_size: int = 2048,
    include_var: bool = True,
):
    """Utility function to predict mean/variance of GP"""
    gp_model.eval()
    mu = []
    var = []
    with gpytorch.settings.fast_computations(False, False, False), torch.no_grad():
        for batch_start in range(0, len(x), batch_size):
            batch_end = batch_start + batch_size
            output = gp_model(x[batch_start:batch_end])
            mu_batch = output.mean.detach().cpu().numpy()
            if include_var:
                var_batch = output.variance.detach().cpu().numpy()
            else:
                var_batch = np.zeros_like(mu_batch)
            mu.append(mu_batch)
            var.append(var_batch)
    mu = np.concatenate(mu, axis=0)
    var = np.concatenate(var, axis=0)
    return mu, var


def fit_gp_hyperparameters(gp_model: ExactGP):
    """Optimize train MLL to fit GP hyperparameters"""

    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    opt_res = botorch.optim.fit.fit_gpytorch_scipy(mll)
    return opt_res


def transfer_gp_hyperparameters(gp_model_src: TanimotoGP, gp_model_dest: TanimotoGP):
    hp_src = gp_model_src.hparam_dict
    hp_dest = gp_model_dest.hparam_dict

    # Clear caches by putting into train mode
    gp_model_dest.train()

    # Do the transfer
    for key in hp_dest.keys():
        param_name = key.split(".")[-1]
        for src_key, val in hp_src.items():
            src_param_name = src_key.split(".")[-1]
            if src_param_name == param_name:
                hp_dest[key] = val

    # Reinitialize variables
    gp_model_dest.initialize(**hp_dest)
