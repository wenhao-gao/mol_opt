import typing

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .. import kl_div
from .. import settings
from .. import similarity_funcs
from .base_parameterised_distribution import BaseParameterisedDistribution

sett = settings.get_settings_manager()


class ShallowDistributions(BaseParameterisedDistribution):
    """
    Shallow distributions do not include any layers. They are directly paramterized distributions by a Tensor.
    """
    def __init__(self, parameterisation: typing.Union[torch.Tensor, nn.Parameter]=None):
        """
        :param parameterisation: [b, ...]
        """
        super().__init__()
        self._params = parameterisation
        self._tb_logger = None

    def update(self, x):
        self._params = x


class DeltaFunctionDistribution(ShallowDistributions):
    """
    Deterministic and does not sample.
    """
    def sample_via_reparam(self, num_samples: int=1):
        samples = [self._params.clone() for _ in range(num_samples)]
        return samples

    def mode(self):
        return self._params

    def nlog_like_of_obs(self, obs: torch.Tensor) -> torch.Tensor:
        matching_locs = (obs == self._param).all(dim=tuple(range(1, len(obs.shape))))

        batch_size = obs.shape[0]
        out = torch.full(batch_size, - np.inf,  device=str(obs.device))

        out[matching_locs] = 0.
        return out

    def kl_with_other(self, other):
        raise RuntimeError

    def convolve_with_function(self, obs: torch.Tensor, function: similarity_funcs.BaseSimilarityFunctions):
        return function.pairwise_similarities(obs, self._params)


class IndependentGaussianDistribution(ShallowDistributions):
    """
    Standard independent Gaussian distribution used eg for latents in the original VAE.

    Either mean and log_variance are parameterized (leave fixed_logvar_value as None in constructor)
    or variance is fixed ( by setting fixed_logvar_value other than None)

    If parameterizing both mean and log variance:
    mean parameterised by the first half of parameters in final dimension.
    The log of the variance by the second half
    """
    def __init__(self, parameterisation: torch.Tensor=None, fixed_logvar_value=None):
        super().__init__(parameterisation)
        self.fixed_logvar_value = fixed_logvar_value

    def sample_via_reparam(self, num_samples: int=1) -> list:
        mean, log_var = self.mean_log_var
        std_dev = torch.exp(0.5 * log_var)
        if self._tb_logger is not None:
            self._tb_logger.add_histogram('z_mean', mean.detach().cpu().numpy())
            self._tb_logger.add_histogram('z_std', std_dev.detach().cpu().numpy())
        samples = mean.unsqueeze(1) + torch.randn(log_var.shape[0], num_samples, *log_var.shape[1:],
                                     dtype=std_dev.dtype, device=str(std_dev.device)) * std_dev.unsqueeze(1)

        samples = list(samples.transpose(0, 1).unbind(dim=0))
        return samples

    @property
    def mean_log_var(self):
        if self.fixed_logvar_value is None:
            params = self._params
            split_point = params.shape[-1] // 2
            return params[..., :split_point], params[..., split_point:]
        else:
            log_var = torch.full_like(self._params, self.fixed_logvar_value)
            return self._params, log_var

    def mode(self) -> torch.Tensor:
        mean, _ = self.mean_log_var
        return mean

    def is_std(self):
        mean, log_var = self.mean_log_var
        return (mean == 0).all() and (log_var == 0).all()

    def kl_with_other(self, other):
        if isinstance(other, IndependentGaussianDistribution):
            if other.is_std():
                mean, log_var = self.mean_log_var
                return kl_div.gauss_kl_with_std_norm(mean, log_var, reduce_fully=False)
        super().kl_with_other(other)

    def nlog_like_of_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.is_std():
            sq_error = obs*2
            term1 = 0.5 * np.log(2 * np.pi).astype(sett.np_float_type)
            nll = sq_error/2 + term1
        else:
            mean, log_var = self.mean_log_var
            sq_error = (obs - mean)**2
            term1 = 0.5 * np.log(2 * np.pi).astype(sett.np_float_type)
            term2 = 0.5 * log_var
            term3 = sq_error / (2 * torch.exp(log_var))
            nll = term2 + term3 + term1
        nll = nll.sum(dim=tuple(range(1, len(nll.shape))))
        return nll  # sum over all but batch.


class BernoulliOnLogits(ShallowDistributions):
    """
    Independent Bernoulli distribution.
    Paramterers are logits, ie go through sigmoid to parameterise Bernoulli
    """


    def bernoulli_params(self):
        params = self._params
        return F.sigmoid(params)

    def sample_via_reparam(self, num_samples: int = 1) -> torch.Tensor:
        raise RuntimeError("Reparameterisation trick not applicable for Bernoulli distribution")

    def sample_no_grad(self, num_samples: int = 1) -> list:
        params = self._params
        params = params.unsqueeze(1).repeat(1, num_samples, *[1 for _ in params.shape[1:]])
        samples = torch.bernoulli(params)
        samples = list(samples.transpose(0, 1).unbind(dim=0))
        return samples

    def mode(self) -> torch.Tensor:
        params = self._params
        return (params > 0.5).type(params.dtype)

    def kl_with_other(self, other):
        raise NotImplementedError("KL for Bernoulli not yet implemented -- need to be careful when distributions do not overlap")

    def nlog_like_of_obs(self, obs: torch.Tensor) -> torch.Tensor:
        params = self._params
        return F.binary_cross_entropy_with_logits(params, obs, reduction='none'
                                                  ).sum(dim=tuple(range(1, len(params.shape))))
