from typing import Optional, Tuple

import tensorflow as tf

import numpy as np

import gpflow

from gpflow.kernels import Kernel
from gpflow import set_trainable
from gpflow.mean_functions import MeanFunction
from gpflow.logdensities import multivariate_normal
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor




class StringGPR(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression over string inputs


    Added a new attribute to allow training of kernel params (over batches if required)
    This is required to interface with the hard-coded kernel gradients in the String kernel

    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)


    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()


    def log_marginal_likelihood(self,batch = None) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.
        Allow calculation across batches of data 
        .. math::
            \log p(Y | \theta).
        """
        if batch is not None:
            X, Y = batch
            Y = tf.reshape(Y,(-1,1))
        else:
            X, Y = self.data
        K = self.kernel(X)
        num_data = tf.shape(X)[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill([num_data], self.likelihood.variance)
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        print(tf.reduce_sum(log_prob))
        return tf.reduce_sum(log_prob)

    

    def fit_kernel_params(self,steps=1,batch_size=None,verbose=True):
        r"""
        Fit kernel parameters usign ADAM 
        Can specifiy batch_size to split up training data
        """      

        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        # prepare data for batching
        if batch_size is not None:
            X, Y =self.data
            train = tf.data.Dataset.from_tensor_slices((X,Y))
            train = train.shuffle(buffer_size = len(X) , seed=1234)
            if len(X) < batch_size:
                raise ValueError("batch_size larger than training data")

            train = train.repeat(steps // (len(X) // batch_size) + 1)
            train = train.batch(batch_size)
            batches = iter(train)

        # perform ADAM optimization
        for step in range(steps):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.trainable_variables)
                if batch_size is None:
                    loss = -self.log_marginal_likelihood()
                else:
                    loss = -self.log_marginal_likelihood(next(batches))
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            if verbose:
                tf.print(f"Step {step} has batch log likelihood of {loss}")
                tf.print(f"Model params are {[p.numpy() for p in self.parameters]}")


        return loss 



    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points
        .. math::
            p(F* | Y)
        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)

        num_data = X_data.shape[0]
        s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm + s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var



