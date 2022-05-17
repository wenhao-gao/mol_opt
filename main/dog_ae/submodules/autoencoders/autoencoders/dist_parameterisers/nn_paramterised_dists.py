import typing

import torch
from torch import nn

from autoencoders import similarity_funcs
from autoencoders.dist_parameterisers.base_parameterised_distribution import T
from . import base_parameterised_distribution
from . import shallow_distributions


class NNParamterisedDistribution(base_parameterised_distribution.BaseParameterisedDistribution):
    """
    Thin wrapper over shallow distributions that allows the input to go through a PyTorch Module before parameterising
    the distribution.
    """
    def __init__(self, network: nn.Module,
                 final_parameterised_dist: shallow_distributions.ShallowDistributions):
        super().__init__()
        self.net = network

        self._net_out = None # temp storage for network output
        self.shallow_dist = final_parameterised_dist

    def update(self, *input, **kwargs) -> T:
        self._net_out = self.net(*input, **kwargs)
        self.shallow_dist.update(self._net_out)
        return self

    def sample_via_reparam(self, num_samples: int = 1) -> list:
        shallow_dist = self.shallow_dist
        return shallow_dist.sample_via_reparam(num_samples)

    def mode(self) -> torch.Tensor:
        shallow_dist = self.shallow_dist
        return shallow_dist.mode()

    def sample_no_grad(self, num_samples: int = 1) -> list:
        shallow_dist = self.shallow_dist
        return shallow_dist.sample_no_grad(num_samples)

    def kl_with_other(self, other):
        shallow_dist = self.shallow_dist
        return shallow_dist.kl_with_other(other)

    def nlog_like_of_obs(self, obs: torch.Tensor) -> torch.Tensor:
        shallow_dist = self.shallow_dist
        return shallow_dist.nlog_like_of_obs(obs)

    def convolve_with_function(self, obs: torch.Tensor,
                               function: similarity_funcs.BaseSimilarityFunctions) -> torch.Tensor:
        shallow_dist = self.shallow_dist
        return shallow_dist.convolve_with_function(obs, function)


