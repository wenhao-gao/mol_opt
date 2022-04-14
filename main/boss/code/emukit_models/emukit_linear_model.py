from typing import Tuple
import itertools
import re
import numpy as np
import GPy
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

from ..parameters.cfg_parameter import CFGParameter
from ..parameters.string_parameter import StringParameter
from ..parameters.protein_base_parameter import ProteinBaseParameter
from ..parameters.protein_codon_parameter import ProteinCodonParameter

class linear_model(GPyModelWrapper):
    """
    This is a thin wrapper around GPy models that one-hot encodes each character and applies linear kernel
    """
    def __init__(self, space , X_init, Y_init, n_restarts: int = 1,observation_noise: bool = False):
        """
        :param space: string space
        :param X_init: starting strings
        :param Y_init: evals of starting strings
        :param n_restarts: Number of restarts during hyper-parameter optimization
        :param observation_noise: have noise term in likelihood
        """

        # check space is either StringParameter, ProteinParameter (CFG params not okay cause variable length)
        if not (isinstance(space.parameters[0],StringParameter) or isinstance(space.parameters[0],ProteinCodonParameter) or isinstance(space.parameters[0],ProteinBaseParameter)):
            raise ValueError("Not a valid string space (needs to be fixed length for linear model)")    
    

        # generate encoding for characters
        encoding = {}
        for i in range(len(space.parameters[0].alphabet)):
            encoding[space.parameters[0].alphabet[i]]=[0]*len(space.parameters[0].alphabet)
            encoding[space.parameters[0].alphabet[i]][i]=1
        self.encoding = encoding


        # transform initialization
        X_feature_init = map_to_feature_space(X_init,self.encoding)

        # first fit of model
        kernel = GPy.kern.Linear(X_feature_init.shape[1], ARD=False)
        gpy_model= GPy.models.GPRegression(X_feature_init, Y_init,kernel,normalizer=True)
        if not observation_noise:
            # no observation noise here but keep a little to help 
            # with matrix inversions (i.e jitter)
            gpy_model.Gaussian_noise.variance.constrain_fixed(1e-6)
        gpy_model.optimize_restarts(n_restarts)
        self.model = gpy_model
        # store inputs as strings seperately
        self.X_strings = X_init

        self.n_restarts = n_restarts

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 of the predictive distribution at each input location
        """
        X = map_to_feature_space(X,self.encoding)
        return self.model.predict(X)

    def predict_with_full_covariance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 and n_points x n_points of the predictive
                 mean and variance at each input location
        """
        X = map_to_feature_space(X,self.encoding)
        return self.model.predict(X, full_cov=True)

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get gradient of the predictions
        :return: (mean gradient, variance gradient) n_points x n_dimensions arrays of the gradients of the predictive
                 distribution at each input location
        """
        X = map_to_feature_space(X,self.encoding)
        d_mean_dx, d_variance_dx = self.model.predictive_gradients(X)
        return d_mean_dx[:, :, 0], d_variance_dx

    def get_joint_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes and returns model gradients of mean and full covariance matrix at given points
        :param X: points to compute gradients at, nd array of shape (q, d)
        :return: Tuple with first item being gradient of the mean of shape (q) at X with respect to X (return shape is (q, q, d)).
                 The second item is the gradient of the full covariance matrix of shape (q, q) at X with respect to X
                 (return shape is (q, q, q, d)).
        """
        X = map_to_feature_space(X,self.encoding)
        dmean_dx = dmean(X, self.model.X, self.model.kern, self.model.posterior.woodbury_vector[:, 0])
        dvariance_dx = dSigma(X, self.model.X, self.model.kern, self.model.posterior.woodbury_inv)
        return dmean_dx, dvariance_dx

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model
        :param X: New training features
        :param Y: New training outputs
        """
        # keep track of strings
        self.X_strings = X
        X = map_to_feature_space(X,self.encoding)
        self.model.set_XY(X, Y)

    def optimize(self):
        """
        Optimizes model hyper-parameters
        """
        self.model.optimize_restarts(self.n_restarts, robust=True)

    def calculate_variance_reduction(self, x_train_new: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        raise ValueError("not implemented for this model")

    def predict_covariance(self, X: np.ndarray, with_noise: bool=True) -> np.ndarray:
        """
        Calculates posterior covariance between points in X
        :param X: Array of size n_points x n_dimensions containing input locations to compute posterior covariance at
        :param with_noise: Whether to include likelihood noise in the covariance matrix
        :return: Posterior covariance matrix of size n_points x n_points
        """
        X = map_to_feature_space(X,self.encoding)
        _, v = self.model.predict(X, full_cov=True, include_likelihood=with_noise)
        v = np.clip(v, 1e-10, np.inf)

        return v

    def get_covariance_between_points(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate posterior covariance between two points
        :param X1: An array of shape 1 x n_dimensions that contains a data single point. It is the first argument of the
                   posterior covariance function
        :param X2: An array of shape n_points x n_dimensions that may contain multiple data points. This is the second
                   argument to the posterior covariance function.
        :return: An array of shape n_points x 1 of posterior covariances between X1 and X2
        """
        X1 = map_to_feature_space(X1,self.encoding)
        X2 = map_to_feature_space(X2,self.encoding)
        return self.model.posterior_covariance_between_points(X1, X2)

    @property
    def X(self) -> np.ndarray:
        """
        :return: An array of shape n_points x n_dimensions containing training inputs
        """
        return self.X_strings

    @property
    def Y(self) -> np.ndarray:
        """
        :return: An array of shape n_points x 1 containing training outputs
        """
        return self.model.Y

    

# make map from string to one-hoteconded feature space
# each char -- > one-hot of length len(alphabet)
def map_to_feature_space(x,encoding):
    x_mapped = []
    for row in x:
        row_chars = row[0].split(" ")
        row_temp=[]
        for char in row_chars:
            row_temp.extend(encoding[char])
        x_mapped.append(row_temp)
    return np.array(x_mapped) 




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

