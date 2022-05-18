
import torch
from torch.nn import functional as F

from .dist_parameterisers.base_parameterised_distribution import BaseParameterisedDistribution
from . import similarity_funcs
from . import base_ae




class WAEnMMD(base_ae.SingleLatentWithPriorAE):
    """
    Wasserstein Auto-encoder MMD

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
    def __init__(self,
                encoder: BaseParameterisedDistribution,
                decoder: BaseParameterisedDistribution,
                latent_prior: BaseParameterisedDistribution,
                kernel: similarity_funcs.BaseSimilarityFunctions,
                c_function: similarity_funcs.SquaredEuclideanDistSimilarity()=None
                ):
        super().__init__(encoder, decoder, latent_prior)

        self.kernel = kernel
        self.c_function = similarity_funcs.SquaredEuclideanDistSimilarity() if c_function is None else c_function
        self._last_z_sample_on_obj = None

    def forward(self, x, lambda_):
        """
        convenience function calculates the WAE-MMD objective
        """
        return self.objective_to_maximise(x, lambda_)

    def objective_to_maximise(self, x, lambda_=1.):
        self.encoder.update(x)
        z_sample = self.encoder.sample_via_reparam(1)[0]
        self._last_z_sample_on_obj = z_sample
        # todo: move this property to base class

        self.decoder.update(z_sample)

        expected_cost = self.decoder.convolve_with_function(x, self.c_function)

        obj = -expected_cost

        collect_extra_stats = self._collect_extra_stats_flag
        if collect_extra_stats:
            extra_statistics = {}


        if lambda_ != 0.:
            samples_from_latent_prior = torch.cat(self.latent_prior.sample_no_grad(num_samples=z_sample.shape[0]))
            divergence_term = similarity_funcs.estimate_mmd(self.kernel, z_sample, samples_from_latent_prior)
            obj += -lambda_*divergence_term  # nb note that this is a scalar, we're just gonna share across each term.

            if collect_extra_stats:
                extra_statistics['sum-divergence_term(no_lambda)(smaller_better)'] = divergence_term.sum().item()

        if collect_extra_stats:
            extra_statistics.update({
                'sum-reconstruction_term(smaller_better)': expected_cost.sum().item(),
                'sum-wae_objective(larger_better)': obj.sum().item(),
                'raw-batchsize': expected_cost.shape[0]
            })
            self._logger_manager.add_statistics(extra_statistics)

        return obj






