from typing import Tuple
import itertools
import re
import numpy as np
import GPy
import collections

from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from ..parameters.cfg_parameter import CFGParameter
from ..parameters.string_parameter import StringParameter
from ..parameters.protein_base_parameter import ProteinBaseParameter
from ..parameters.protein_codon_parameter import ProteinCodonParameter


class BIO_Features_model(GPyModelWrapper):
    """
    This is a thin wrapper around GPy models that converts raw strings to BOW and applies RBF kernel in features spaces
    This is an implementation specifically for protein parameters
    Implements the method of Gonzalez et al. 2016
    Collects frequency counts of codons and some extra biologically inspired features
    """
    def __init__(self, space , X_init, Y_init, max_feature_length: int=5, n_restarts: int = 1,observation_noise: bool = False):
        """
        :param space: string space
        :param X_init: starting strings
        :param Y_init: evals of starting strings
        :param max_feature_length: maximum length of features considered
        :param n_restarts: Number of restarts during hyper-parameter optimization
        :param observation_noise: have noise term in likelihood
        """

        # check space is a ProteinCodonParameter 
        if not isinstance(space.parameters[0],ProteinBaseParameter):
            raise ValueError("Not a valid string space")    


        # generate possible features 
        tuples=[]
        for i in range(1,max_feature_length+1):
            tuples.extend(list(itertools.combinations_with_replacement(space.parameters[0].alphabet, i)))
        features=[]
        for t in tuples:
            features.append(" ".join(list(t)))
        self.features = features

        # transform initialization
        X_feature_init = map_to_feature_space(X_init)

        # first fit of model
        kernel = GPy.kern.RBF(X_feature_init.shape[1], ARD=False)
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
        X = map_to_feature_space(X)
        return self.model.predict(X)

    def predict_with_full_covariance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 and n_points x n_points of the predictive
                 mean and variance at each input location
        """
        X = map_to_feature_space(X)
        return self.model.predict(X, full_cov=True)

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get gradient of the predictions
        :return: (mean gradient, variance gradient) n_points x n_dimensions arrays of the gradients of the predictive
                 distribution at each input location
        """
        X = map_to_feature_space(X)
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
        X = map_to_feature_space(X)
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
        X = map_to_feature_space(X)
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
        X = map_to_feature_space(X)
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
        X1 = map_to_feature_space(Xs)
        X2 = map_to_feature_space(Xs)
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


# functions to extract features


def create_codon_table():
    bases = ['t', 'c', 'a', 'g']
    codons = [a+b+c for a in bases for b in bases for c in bases]
    amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
    codon_table = dict(zip(codons, amino_acids))
    return(codon_table)
## counts the number of basis
def count_number_basis(seq):
    number = len(seq)
    return number
## function to counts from a sequence
def count_codons(seq):
    codon_table = create_codon_table()
    counts = collections.defaultdict(int)
    for codon in codon_table.keys(): counts[codon] = 0
    for i in range(0,len(seq)-len(seq)%3,3): codon = seq[i:i+3] ; counts[codon] += 1
    return(counts)
## calculates the GC content
def gc_content(seq,loc=0):
    ## loc represents the location of the bases position
    if loc==1: seq = seq[0::3]
    if loc==2: seq = seq[1::3]
    if loc==3: seq = seq[2::3]
    content = (seq.count('g')+seq.count('c'))/float(count_number_basis(seq))
    return(content)
## calculates the GC ratio
def gc_ratio(seq):
    content = (seq.count('g')+seq.count('c'))/float((seq.count('a')+seq.count('t')))
    return(content)
## calculates the AT content
def at_content(seq):
    content = (seq.count('a')+seq.count('t'))/float(count_number_basis(seq))*100
    return(content)
## calculates the AT ratio
def at_ratio(seq):
    content = (seq.count('a')+seq.count('t'))/float((seq.count('g')+seq.count('c')))
    return(content)
codons = ['ttt','ttc','tta','ttg','tct','tcc','tca','tcg','tat','tac','taa','tag','tgt','tgc','tga','tgg','ctt','ctc','cta','ctg','cct','ccc','cca','ccg','cat','cac','caa',
 'cag','cgt','cgc','cga','cgg','att','atc','ata','atg','act','acc','aca','acg','aat',
 'aac','aaa','aag','agt','agc','aga','agg','gtt','gtc','gta','gtg','gct','gcc','gca','gcg','gat','gac','gaa','gag','ggt','ggc','gga','ggg']
# collect features
def collect_features(seq):
    features=[]
    # count codon occurences
    occurences = count_codons(seq)
    for codon in codons:
        features.append(occurences[codon])
    # GC content
    features.append(gc_content(seq,loc=0))
    # AT content
    features.append(at_content(seq))
    # GC ratio
    features.append(gc_ratio(seq))
    # AT-ratio
    features.append(at_ratio(seq))
    return features    


# define map from string to featrure space
def map_to_feature_space(x):
    #x is a np.array of strings (2-d)
    # e.g. np.array([["0 1 0 1 0 0 0 1 0 1"],["0 1 0 1 0 0 0 1 0 1"]])
    # and a list of features to look for
    representations=np.zeros((x.shape[0],68))
    for i in range(x.shape[0]):
        string = "".join(x[i][0].split(" "))
        representations[i]=np.array(collect_features(string))
    return representations



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

