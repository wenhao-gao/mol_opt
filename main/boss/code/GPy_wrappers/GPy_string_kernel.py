from GPy.kern import Kern
from GPy.core.parameterization import Param
import numpy as np
import sys
from paramz.transformations import Logexp
from ..kernels.string.np_string_kernel import NPStringKernel



class StringKernel(Kern):
    """
    The string kernel described by Beck (2017)
    
   with hyperparameters:
    1) match_decay float
        decrease the contribution of long subsequences
    2) gap_decay float
        decrease the contribtuion of subsequences with large gaps (penalize non-contiguous)
    3) order_coefs list(floats) 
          n-gram weights to help tune the signal coming from different sub-sequence lengths
    We calculate gradients w.r.t kernel hyperparameters following Beck (2017)
    This is mainly a wrapper for a numpy or numba implementation stored on the "kernel" attribute.

    We recommend normalize = True to allow meaningful comparrison of strings of different length

    X is a numpy array of size (n,1) where each element is a string with characters seperated by spaces
    """
    def __init__(self, gap_decay=1.0, match_decay=2.0, order_coefs=[1.0],
                 alphabet = [], maxlen=0, active_dims=None, normalize = True,batch_size=1000):
        super(StringKernel, self).__init__(1, active_dims, 'sk')
        self._name = "sk"
        self.gap_decay = Param('Gap_decay', gap_decay,Logexp())
        self.match_decay = Param('Match_decay', match_decay,Logexp())
        self.order_coefs = Param('Order_coefs',order_coefs, Logexp())
        self.link_parameters(self.gap_decay, self.match_decay,self.order_coefs)
        
        self.alphabet = alphabet
        self.maxlen = maxlen
        self.normalize = normalize

        self.kernel = NPStringKernel(_gap_decay=gap_decay, _match_decay=match_decay,
                                     _order_coefs=list(order_coefs), alphabet = self.alphabet, 
                                     maxlen=maxlen,normalize=normalize)

    def K(self, X, X2):
        # calc the kernel for input X
        # also calc the gradients w.r.t kernel parameters
        # need to update the TF stored hyper-parameters
        self.kernel._gap_decay = self.gap_decay[0]
        self.kernel._match_decay = self.match_decay[0]
        self.kernel._order_coefs = list(self.order_coefs.values)
        #calc kernel and store grads
        k, gap_grads, match_grads, coef_grads = self.kernel.K(X, X2)
        self.gap_grads = gap_grads
        self.match_grads = match_grads
        self.coef_grads = coef_grads
        return k


    def Kdiag(self, X):
        # Calc just the diagonal elements of a kernel matrix
        # need to update the TF stored hyper-parameters
        self.kernel._gap_decay = self.gap_decay[0]
        self.kernel._match_decay = self.match_decay[0]
        self.kernel._order_coefs = list(self.order_coefs.values)
        return self.kernel.Kdiag(X)

    def dK_dtheta(self, dL_dK, X, X2):
        # return the kernel gradients w.r.t kernel parameter over the dataset
        self.K(X,X2)
        return np.array([np.sum(self.gap_grads  * dL_dK),
                np.sum(self.match_grads * dL_dK),
                np.sum(self.coef_grads * dL_dK)])

    def update_gradients_full(self, dL_dK, X, X2):
        # update gradients for optimization of kernel parameters
        self.gap_decay.gradient = np.sum(self.gap_grads * dL_dK)
        self.match_decay.gradient = np.sum(self.match_grads * dL_dK)
        for i in range(len(self.order_coefs.values)):
            self.order_coefs.gradient[i] = np.sum(self.coef_grads[:, :, i] * dL_dK)





if __name__ == "__main__":
    #Simple Demo
    X = np.array([['c a t a'],['g a t t a'],['c g t a g c t a g c g a c g c a g c c a a t c g a t c g'],
                ['c g a g a t g c c a a t a g a g a g a g c g c t g t a']])
    kern=StringKernel(normalize=False,maxlen=28,alphabet=["c","a","t","g"],order_coefs=[1,1,1,1,1],match_decay=2,gap_decay=2)
    K = kern.K(X,X)
    print(str(K[0][1])+" should be 504")
    kern=StringKernel(normalize=False,maxlen=28,alphabet=["c","a","t","g"],order_coefs=[1,1,1,1,1],match_decay=0.8,gap_decay=0.8)
    K = kern.K(X,X)
    print(str(K[0][1])+" should be 5.943705")