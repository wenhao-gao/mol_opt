
import torch
from torch import nn
import numpy as np

from ..dist_parameterisers import nn_paramterised_dists
from ..dist_parameterisers import shallow_distributions
from .. import variational


class MnistVaeParams:
    def __init__(self):
        self.data_dim = 28 * 28
        self.latent_space_dim = 20

        self.gaussian_out = False  # if not will be Bernoulli


def create_mnist_vae_model(params: MnistVaeParams):
    """
    ::
        @ARTICLE{Kingma2013-it,
          title         = "{Auto-Encoding} Variational Bayes",
          author        = "Kingma, Diederik P and Welling, Max",
          abstract      = "How can we perform efficient inference and learning in
                           directed probabilistic models, in the presence of continuous
                           latent variables with intractable posterior distributions,
                           and large datasets? We introduce a stochastic variational
                           inference and learning algorithm that scales to large
                           datasets and, under some mild differentiability conditions,
                           even works in the intractable case. Our contributions is
                           two-fold. First, we show that a reparameterization of the
                           variational lower bound yields a lower bound estimator that
                           can be straightforwardly optimized using standard stochastic
                           gradient methods. Second, we show that for i.i.d. datasets
                           with continuous latent variables per datapoint, posterior
                           inference can be made especially efficient by fitting an
                           approximate inference model (also called a recognition
                           model) to the intractable posterior using the proposed lower
                           bound estimator. Theoretical advantages are reflected in
                           experimental results.",
          month         =  dec,
          year          =  2013,
          archivePrefix = "arXiv",
          primaryClass  = "stat.ML",
          eprint        = "1312.6114v10"
        }
    """
    # Sort out the Encoder
    num_params_for_latent_space = 2 * params.latent_space_dim

    encoder_net = nn.Sequential(
        nn.Linear(params.data_dim, 400),
        nn.ReLU(),
        nn.Linear(400, num_params_for_latent_space)
    )
    encoder = nn_paramterised_dists.NNParamterisedDistribution(
        network=encoder_net, final_parameterised_dist=shallow_distributions.IndependentGaussianDistribution()
    )

    # Sort out the latent prior
    latent_prior = shallow_distributions.IndependentGaussianDistribution(torch.zeros(1, num_params_for_latent_space))

    # Sort out the decoder
    num_dims_out = 2 * params.data_dim if params.gaussian_out else params.data_dim
    decoder_net = nn.Sequential(
        nn.Linear(params.latent_space_dim, 400),
        nn.ReLU(),
        nn.Linear(400, num_dims_out)
    )
    decoder_paramaterised_net = shallow_distributions.IndependentGaussianDistribution() \
        if params.gaussian_out else shallow_distributions.BernoulliOnLogits()
    decoder = nn_paramterised_dists.NNParamterisedDistribution(
        network=decoder_net, final_parameterised_dist=decoder_paramaterised_net
    )

    vae = variational.VAE(encoder=encoder, decoder=decoder, latent_prior=latent_prior)
    return vae

