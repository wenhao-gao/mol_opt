
from . import base_ae


class VAE(base_ae.SingleLatentWithPriorAE):
    """
    Basic Variational Autoencoder

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
    def forward(self, x, beta):
        """
        convenience function calculates the ELBO term
        """
        return self.elbo(x, beta, return_extra_vals=False)

    def elbo(self, x, beta=1., return_extra_vals=False):
        self.encoder.update(x)
        z_sample = self.encoder.sample_via_reparam(1)[0]
        self._last_z_sample_on_obj = z_sample

        self.decoder.update(z_sample)
        log_like = -self.decoder.nlog_like_of_obs(x)

        elbo = log_like

        collect_extra_stats = self._collect_extra_stats_flag
        if collect_extra_stats:
            extra_statistics = {
                'sum-reconstruction_term(larger_better)': log_like.sum().item(),
                'raw-batchsize': elbo.shape[0],
                'raw-beta': float(beta)
            }


        if beta != 0.:
            kl_term = -self.encoder.kl_with_other(self.latent_prior)
            elbo += beta * kl_term

            if collect_extra_stats:
                extra_statistics['sum-neg_kl_(no_beta)(larger_better)'] = kl_term.sum().item()

        if collect_extra_stats:
            extra_statistics['sum-elbo(larger_better)'] = log_like.sum().item()
            self._logger_manager.add_statistics(extra_statistics)
        return elbo

