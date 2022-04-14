from GPy.kern import Kern
from GPy.core.parameterization import Param
import numpy as np
import sys
from paramz.transformations import Logexp
from ..kernels.tree.C_tree_kernel import wrapper_raw_SubsetTreeKernel



class SubsetTreeKernel(Kern):
    """
    The SST kernel by Moschitti(2006), with two hyperparameters (lambda and sigma).
    small lambda restricts the influence of large fragments, sigma controls the sparsity (sigma=0 only allows fragments with terminal symbols)
    We calculate gradients w.r.t kernel phyperparameters following Beck (2015)
    This is mainly a wrapper for a Cython implementation (see C_tree_kernel.pyx).
    The Cython kernel is stored on the "kernel" attribute.
    

    Following the GPy stanard, we require input in the form of 2-d numpy arrays of strings with dtype=object

    e.g

    X=np.array([['(S (NP ns) (VP v))'],
                          ['(S (NP n) (VP v))'],
                          ['(S (NP (N a)) (VP (V c)))'],
                          ['(S (NP (Det a) (N b)) (VP (V c)))'],
                          ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                         dtype=object)


    Each inidivudal string should be in the prolog format e.g. "(C (B c) (D a))" for 
        C
       / \
      B   D
      |   |
      c   a

    """

    def __init__(self, _lambda=1, _sigma=1, normalize=True, active_dims=None):
        super(SubsetTreeKernel, self).__init__(1, active_dims, 'sstk')
        self._lambda = Param('Lambda', _lambda,Logexp())
        self._sigma = Param('Sigma', _sigma,Logexp())
        self.link_parameters(self._lambda, self._sigma)
        self.normalize = normalize
        self.kernel = wrapper_raw_SubsetTreeKernel(_lambda, _sigma, normalize)
        
    def _get_params(self):
        # return kernel parameter values
        return np.hstack((self._lambda, self._sigma))

    def _set_params(self, x):
        # set kernel parameters
        self._lambda = x[0]
        self._sigma = x[1]

    def _get_param_names(self):
        # return parameter names
        return ['Lambda', 'Sigma']

    def K(self, X, X2):
        # calc the kernel for input X
        # also calc the gradients w.r.t kernel parameters
        self.kernel._lambda = self._lambda
        self.kernel._sigma = self._sigma
        result, dl, ds = self.kernel.K(X, X2)
        self.dlambda = dl
        self.dsigma = ds
        return result

    def Kdiag(self, X):
        # Calc just the diagonal elements of a kernel matrix
        self.kernel._lambda = self._lambda
        self.kernel._sigma = self._sigma
        if self.normalize:
            # if normalizing then this will just be ones
            return np.ones(X.shape[0])
        else:
            return self.kernel.Kdiag(X)

    def dK_dtheta(self, dL_dK, X, X2):
        # return the kerenl gradients w.r.t kernel parameter over the dataset
        self.K(X,X2)
        return np.array([np.sum(self.dlambda * dL_dK),
                np.sum(self.dsigma * dL_dK)])

    def update_gradients_full(self, dL_dK, X, X2):
        # update gradients for optimization of kernel parameters
        self._lambda.gradient = np.sum(self.dlambda * dL_dK)
        self._sigma.gradient = np.sum(self.dsigma * dL_dK)


if __name__ == "__main__":
    #Simple Demo
    X=np.array([['(S (NP ns) (VP v))'],
                          ['(S (NP n) (VP v))'],
                          ['(S (NP (N a)) (VP (V c)))'],
                          ['(S (NP (Det a) (N b)) (VP (V c)))'],
                          ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                         dtype=object)

    kern = SubsetTreeKernel(_lambda=1)
    print("test calculations with normalization")
    print(str(kern.K(X))+"\n should be\n"+str(np.array([[ 1.,          0.5,         0.10540926,  0.08333333,  0.06711561],
                      [ 0.5,         1.,          0.10540926,  0.08333333,  0.06711561],
                      [ 0.10540926,  0.10540926,  1.,          0.31622777,  0.04244764],
                      [ 0.08333333,  0.08333333,  0.31622777,  1.,          0.0335578 ],
                      [ 0.06711561,  0.06711561,  0.04244764,  0.0335578,   1.        ]])))






