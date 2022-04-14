from typing import Tuple
import itertools
import re
import numpy as np
import GPy
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

from ..parameters.cfg_parameter import CFGParameter
from ..parameters.string_parameter import StringParameter
from ..parameters.candidate_parameter import CandidateStringParameter
from ..parameters.protein_base_parameter import ProteinBaseParameter
from ..parameters.protein_codon_parameter import ProteinCodonParameter
from ..GPy_wrappers.GPy_string_kernel import StringKernel
from ..GPy_wrappers.GPy_split_string_kernel import SplitStringKernel


class SSK_model(GPyModelWrapper):
    """
    This is a wrapper around GPy providing an SSK model for emukit
    """
    def __init__(self, space , X_init, Y_init,max_subsequence_length: int = 5,  n_restarts: int = 3,
        num_splits: int = 1, observation_noise: bool = False):
        """
        :param space: string space
        :param X_init: starting strings
        :param Y_init: evals of starting strings
        :param max_subsequence_length: length of longest considered subsequence
        :param num_splits: number of times split up string when applying SSK
        :param n_restarts: Number of restarts during hyper-parameter optimization
        :param observation_noise: have noise term in likelihood
        """

        # check space is either StringParameter, ProteinParameter or CFGParameters
        if not (isinstance(space.parameters[0],StringParameter) or isinstance(space.parameters[0],CFGParameter) or (isinstance(space.parameters[0],CandidateStringParameter)) or isinstance(space.parameters[0],ProteinCodonParameter) or isinstance(space.parameters[0],ProteinBaseParameter)):
            raise ValueError("Not a valid string space")    

        # if space is a protein in base form, then string 3*longer
        if isinstance(space.parameters[0],ProteinBaseParameter):
            length = 3*space.parameters[0].length
        # if CFG space, then get max length of string
        elif isinstance(space.parameters[0],CFGParameter):
            length = space.parameters[0].max_length
        else:
            length = space.parameters[0].length
        # first fit of model
        if num_splits==1:
            kernel = StringKernel(order_coefs=[1]*max_subsequence_length,maxlen=length,alphabet=space.parameters[0].alphabet)
        else:
            kernel = SplitStringKernel(order_coefs=[1]*max_subsequence_length,maxlen=length,alphabet=space.parameters[0].alphabet,num_splits=num_splits)
        gpy_model= GPy.models.GPRegression(X_init, Y_init,kernel,normalizer=True)
        if not observation_noise:
            # no observation noise here but keep a little to help 
            # with matrix inversions (i.e jitter)
            gpy_model.Gaussian_noise.variance.constrain_fixed(1e-6)
        gpy_model.sk.Gap_decay.constrain_bounded(0,1)
        gpy_model.sk.Match_decay.constrain_bounded(0,1)
        gpy_model.sk.Order_coefs.constrain_fixed([1]*max_subsequence_length)
        gpy_model.optimize_restarts(n_restarts,verbose=False)
        self.model = gpy_model


        self.n_restarts = n_restarts

    
def dSigma(x_predict: np.ndarray, x_train: np.ndarray, kern: GPy.kern, w_inv: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the posterior covariance with respect to the prediction input
    :param x_predict: Prediction inputs of shape (q, d)
    :param x_train: Training inputs of shape (n, d)
    :param kern: Covariance of the GP model
    :param w_inv: Woodbury inverse of the posterior fit of the GP
    :return: Gradient of the posterior covariance of shape (q, q, q, d)
    """
    q, d, n = x_predict.shape[0], x_predict.shape[1], x_train.shape[0]
    dkxX_dx = np.empty((q, n, d))
    dkxx_dx = np.empty((q, q, d))
    for i in range(d):
        dkxX_dx[:, :, i] = kern.dK_dX(x_predict, x_train, i)
        dkxx_dx[:, :, i] = kern.dK_dX(x_predict, x_predict, i)
    K = kern.K(x_predict, x_train)

    dsigma = np.zeros((q, q, q, d))
    for i in range(q):
        for j in range(d):
            Ks = np.zeros((q, n))
            Ks[i, :] = dkxX_dx[i, :, j]
            dKss_dxi = np.zeros((q, q))
            dKss_dxi[i, :] = dkxx_dx[i, :, j]
            dKss_dxi[:, i] = dkxx_dx[i, :, j].T
            dKss_dxi[i, i] = 0
            dsigma[:, :, i, j] = dKss_dxi - Ks @ w_inv @ K.T - K @ w_inv @ Ks.T
    return dsigma


def dmean(x_predict: np.ndarray, x_train: np.ndarray, kern: GPy.kern, w_vec: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the posterior mean with respect to prediction input
    :param x: Prediction inputs of shape (q, d)
    :param X: Training inputs of shape (n, d)
    :param kern: Covariance of the GP model
    :param w_inv: Woodbury vector of the posterior fit of the GP
    :return: Gradient of the posterior mean of shape (q, q, d)
    """
    q, d, n = x_predict.shape[0], x_predict.shape[1], x_train.shape[0]
    dkxX_dx = np.empty((q, n, d))
    dmu = np.zeros((q, q, d))
    for i in range(d):
        dkxX_dx[:, :, i] = kern.dK_dX(x_predict, x_train, i)
        for j in range(q):
            dmu[j, j, i] = (dkxX_dx[j, :, i][None, :] @ w_vec[:, None]).flatten()
    return dmu

