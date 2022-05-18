
import torch
import numpy as np

MINIMUM_VAR = 1e-20
MINIMUM_LOG_VAR = np.log(MINIMUM_VAR)


def gauss_kl_with_std_norm(z_mean, z_log_var, reduce_fully=False):
    """
    Computes the KL divergence between a multivariate Gaussian distribution (with diagonal covariance) and a
    standard normal Gaussian distribution,

    Works over a batch, b, of multivariate Gaussian distributions.

    eg see p.13 of
    http://web.stanford.edu/~jduchi/projects/general_notes.pdf
    :param z_mean: [b, n] parametrises the mean of the multivariate Gaussian distribution
    :param z_var: [b, n] parameterises the elementwise log
     diagonal of the variance matrix of the multivariate Gaussian distribution.
    :return: [b] for KL for each element in batch, unless reduce_fully is True in which case returns a scalar.
    """
    n = z_mean.shape[1]
    #todo: consider changing to softplus

    z_log_var = torch.max(z_log_var, torch.tensor(MINIMUM_LOG_VAR, device=z_log_var.device))
    term1 = - torch.sum(z_log_var, dim=1)
    term2 = -n

    term3 = torch.sum(z_log_var.exp(), dim=1)
    term4 = torch.sum(z_mean * z_mean, dim=1)

    kl = 0.5 * (term1 + term2 + term3 + term4)
    if reduce_fully:
        kl = torch.sum(kl)
    return kl
