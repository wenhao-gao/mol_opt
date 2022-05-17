
import torch
from torch import nn
import numpy as np

from ..dist_parameterisers import nn_paramterised_dists
from ..dist_parameterisers import shallow_distributions
from .. import wasserstein
from .. import similarity_funcs
from .. import ae_utils


class MnistWaeParams:
    def __init__(self):
        self.img_width_height = 28
        self.latent_space_dim = 8


class FlattenAllButFirst(nn.Module):
    def forward(self, tensor_in):
        return tensor_in.flatten(start_dim=1)

class ReshapeIntoImage(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels

    def forward(self, tensor_in):
        return tensor_in.view(tensor_in.shape[0], self.channels, self.height, self.width)


def compute_output_size_no_dilation(input_size, padding, kernel_size, stride_size):
    return int(np.floor(float(input_size + 2. * padding - kernel_size) / stride_size + 1))


def create_mnist_wae_mmd_model(params: MnistWaeParams):
    """
    We try to generally match Tolstikhin et al. in our implementation but differ in some places, partly from just
    coding in a different framework.

    In particular see Algorithm 2, section 4 and Appendix C.1 of:

    ::
        @ARTICLE{Tolstikhin2017-jz,
          title         = "Wasserstein {Auto-Encoders}",
          author        = "Tolstikhin, Ilya and Bousquet, Olivier and Gelly, Sylvain
                           and Schoelkopf, Bernhard",
          abstract      = "We propose the Wasserstein Auto-Encoder (WAE)---a new
                           algorithm for building a generative model of the data
                           distribution. WAE minimizes a penalized form of the
                           Wasserstein distance between the model distribution and the
                           target distribution, which leads to a different regularizer
                           than the one used by the Variational Auto-Encoder (VAE).
                           This regularizer encourages the encoded training
                           distribution to match the prior. We compare our algorithm
                           with several other techniques and show that it is a
                           generalization of adversarial auto-encoders (AAE). Our
                           experiments show that WAE shares many of the properties of
                           VAEs (stable training, encoder-decoder architecture, nice
                           latent manifold structure) while generating samples of
                           better quality, as measured by the FID score.",
          month         =  nov,
          year          =  2017,
          archivePrefix = "arXiv",
          primaryClass  = "stat.ML",
          eprint        = "1711.01558"
        }
    """
    # Convolution parameters:
    kernel_size = 4
    stride = 2
    padding = 1

    # Create the encoder
    def create_block(channel_size_in, channel_size_out):
        # Nb in paper they use padding SAME. There may be at some point a nice way to do this in PyTorch:
        # eg see https://github.com/pytorch/pytorch/issues/3867
        # However at the moment I'm just using a padding of size 1 so that we ensure we do not cut off some of the
        # outer image pixels, although tbh maybe this doesnt even matter too much on MNIST.
        return nn.Sequential(
            nn.Conv2d(channel_size_in, channel_size_out, kernel_size=kernel_size, stride=stride, bias=False, padding=padding),
            # ^ no bias as using batch norm
            nn.BatchNorm2d(channel_size_out),
            nn.ReLU()
        )

    channel_sizes = [1, 128, 256, 512, 1024]
    final_conv_width = ae_utils.repeated(lambda x: compute_output_size_no_dilation(x, padding=padding, kernel_size=kernel_size,
                                                                stride_size=stride), len(channel_sizes)-1)(params.img_width_height)


    encoder_net = nn.Sequential(*[create_block(a, b) for a,b in zip(channel_sizes[:-1], channel_sizes[1:])],
                                FlattenAllButFirst(),
                                nn.Linear(final_conv_width*final_conv_width*1024, params.latent_space_dim))
    encoder = nn_paramterised_dists.NNParamterisedDistribution(
        network=encoder_net, final_parameterised_dist=shallow_distributions.DeltaFunctionDistribution()
    )

    # Create the latent prior
    latent_prior = shallow_distributions.IndependentGaussianDistribution(
        nn.Parameter(torch.zeros(1, 2*params.latent_space_dim), requires_grad=False)
                     )

    # Create the kernel for doing MMD in the latent space
    kernel = similarity_funcs.InverseMultiquadraticsKernel(c=2*params.latent_space_dim*(1**2))  # see p.9 of paper

    # Create the decoder
    decoder_net = nn.Sequential(nn.Linear(params.latent_space_dim, 7*7*1024),
                                nn.ReLU(),
                                ReshapeIntoImage(7, 7, 1024),
                                nn.ConvTranspose2d(1024, 512, kernel_size=kernel_size, stride=stride, padding=padding,
                                                   bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.ConvTranspose2d(512, 256, kernel_size=kernel_size, stride=stride, padding=padding,
                                                   bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0,
                                                   bias=True),
                                # ^ just for compressing down the number of channels. NB I think the official implementation
                                # still keeps a kernel size of four here (and theyre still using transposed convolutions)
                                # However, for this to work I think you need to adjust the padding above.
                                nn.Sigmoid()
                                )
    decoder = nn_paramterised_dists.NNParamterisedDistribution(network=decoder_net,
                                    final_parameterised_dist=shallow_distributions.DeltaFunctionDistribution())

    # Create the WAE
    wae = wasserstein.WAEnMMD(encoder=encoder, decoder=decoder, latent_prior=latent_prior, kernel=kernel)
    return wae


