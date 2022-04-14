""" Code for GPs with Tanimot kernel """

import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, Kernel, InducingPointKernel
from gpytorch.models import ExactGP
import botorch


def batch_tanimoto_sim(x1: torch.Tensor, x2: torch.Tensor):
    """tanimoto between two batched tensors, across last 2 dimensions"""
    assert x1.ndim >= 2 and x2.ndim >= 2
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_sum = torch.sum(x1 ** 2, dim=-1, keepdims=True)
    x2_sum = torch.sum(x2 ** 2, dim=-1, keepdims=True)
    return (dot_prod) / (x1_sum + torch.transpose(x2_sum, -1, -2) - dot_prod)


class TanimotoKernel(Kernel):
    """Tanimoto coefficient kernel"""

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        return batch_tanimoto_sim(x1, x2)


class TanimotoGP(ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1  # looks like botorch needs this

    def __init__(
        self,
        train_x,
        train_y,
        likelihood=None,
    ):

        # Fill in likelihood
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        botorch.models.gpytorch.GPyTorchModel.__init__(self)
        ExactGP.__init__(self, train_x, train_y, likelihood)

        self.covar_module = ScaleKernel(TanimotoKernel())
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):

        # Normal mean + covar
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def hparam_dict(self):
        return {
            "likelihood.noise": self.likelihood.noise.item(),
            "covar_module.outputscale": self.covar_module.outputscale.item(),
            "mean_module.constant": self.mean_module.constant.item(),
        }


class TanimotoSGP(TanimotoGP):
    """SGPR with Tanimoto GP"""

    def __init__(self, *args, inducing_points=None, **kwargs):
        assert inducing_points is not None
        super().__init__(*args, **kwargs)

        # now use a base covar module
        self.base_covar_module = self.covar_module
        self.covar_module = InducingPointKernel(
            self.base_covar_module,
            inducing_points=inducing_points,
            likelihood=self.likelihood,
        )

    @property
    def hparam_dict(self):
        return {
            "likelihood.noise": self.likelihood.noise.item(),
            "covar_module.base_kernel.outputscale": self.covar_module.base_kernel.outputscale.item(),
            "mean_module.constant": self.mean_module.constant.item(),
        }
