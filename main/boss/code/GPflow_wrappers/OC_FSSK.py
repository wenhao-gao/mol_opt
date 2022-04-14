from gpflow.kernels import Kernel
from gpflow.utilities import positive
from gpflow import Parameter
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp
import numpy as np

class OC_FSSK(Kernel):
    """
    Code to run a soft-matched SSK with gpflow
    
   with hyperparameters:
    1) match_decay float
        decrease the contribution of long subsequences
    2) gap_decay float
        decrease the contribtuion of subsequences with large gaps (penalize non-contiguous)
    3) max_subsequence_length int 
        largest subsequence considered
    4) rank int
         rank of decomposition of similairty matrix (total free similarity parameters = alpahabet_size * (rank+1))
       """

    def __init__(self,rank=1,active_dims=[0],gap_decay=0.1, match_decay=0.9,max_subsequence_length=3,
                 alphabet = [], maxlen=0):
        super().__init__(active_dims=active_dims)
        # constrain decay kernel params to between 0 and 1
        logistic_gap = tfb.Chain([tfb.Shift(tf.cast(0,tf.float64))(tfb.Scale(tf.cast(1,tf.float64))),tfb.Sigmoid()])
        logisitc_match = tfb.Chain([tfb.AffineScalar(shift=tf.cast(0,tf.float64),scale=tf.cast(1,tf.float64)),tfb.Sigmoid()])
        self.gap_decay= Parameter(gap_decay, transform=logistic_gap ,name="gap_decay")
        self.match_decay = Parameter(match_decay, transform=logisitc_match,name="match_decay")

        # prepare similarity matrix parameters
        self.rank=rank
        W = 0.1 * tf.ones((len(alphabet), self.rank))
        kappa = 0.99*tf.ones(len(alphabet))
        self.W = Parameter(W,name="W")
        self.kappa = Parameter(kappa, transform=positive(),name="kappa")
  
        # prepare order coefs params
        order_coefs=tf.ones(max_subsequence_length)
        self.order_coefs =  Parameter(order_coefs, transform=positive(),name="order_coefs")

        # store additional kernel parameters
        self.max_subsequence_length = tf.constant(max_subsequence_length)
        self.alphabet =  tf.constant(alphabet)
        self.alphabet_size=tf.shape(self.alphabet)[0]
        self.maxlen =  tf.constant(maxlen)

        # build a lookup table of the alphabet to encode input strings
        self.table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(["PAD"]+alphabet),
                values=tf.constant(range(0,len(alphabet)+1)),),default_value=0)


    def K_diag(self, X):
        r"""
        The diagonal elements of the string kernel are always unity (due to normalisation)
        """
        return tf.ones(tf.shape(X)[:-1],dtype=tf.float64)



    def K(self, X1, X2=None):
        r"""
        Vectorized kernel calc.
        Following notation from Beck (2017), i.e have tensors S,D,Kpp,Kp
        Input is two tensors of shape (# strings , # characters)
        and we calc the pair-wise kernel calcs between the elements (i.e n kern calcs for two lists of length n)
        D is the tensor than unrolls the recursion and allows vecotrizaiton
        """

        # Turn our inputs into lists of integers using one-hot embedding
        # first split up strings and pad to fixed length and prep for gpu
        # pad until all have length of self.maxlen
        X1 = tf.strings.split(tf.squeeze(X1,1)).to_tensor("PAD",shape=[None,self.maxlen])
        X1 = self.table.lookup(X1)
        if X2 is None:
            X2 = X1
            self.symmetric = True
        else:
            self.symmetric = False
            X2 = tf.strings.split(tf.squeeze(X2,1)).to_tensor("PAD",shape=[None,self.maxlen])
            X2 = self.table.lookup(X2)
        # keep track of original input sizes
        X1_shape = tf.shape(X1)[0]
        X2_shape = tf.shape(X2)[0]

        # prep the decay tensor D 
        self.D = self._precalc()


        # turn into one-hot  i.e. shape (# strings, #characters+1, alphabet size)
        X1 = tf.one_hot(X1,self.alphabet_size+1,dtype=tf.float64)
        X2 = tf.one_hot(X2,self.alphabet_size+1,dtype=tf.float64)
        # remove the ones in the first column that encode the padding (i.e we dont want them to count as a match)
        X1 = X1[:,:,1:]
        X2 = X2[:,:,1:]


        # get indicies of all possible pairings from X and X2
        # this way allows maximum number of kernel calcs to be squished onto the GPU (rather than just doing individual rows of gram)
        indicies_2, indicies_1 = tf.meshgrid(tf.range(0, tf.shape(X2)[0]),tf.range(0, tf.shape(X1)[0]))
        indicies = tf.concat([tf.reshape(indicies_1,(-1,1)),tf.reshape(indicies_2,(-1,1))],axis=1)
        # if symmetric then only calc upper matrix (fill in rest later)
        if self.symmetric:
            indicies = tf.boolean_mask(indicies,tf.greater_equal(indicies[:,1],indicies[:,0]))
        
        X1_full= tf.gather(X1,indicies[:,0],axis=0)
        X2_full = tf.gather(X2,indicies[:,1],axis=0)

        if not self.symmetric:
            # also need to calculate some extra kernel evals for the normalization terms
            X1_full = tf.concat([X1_full,X1,X2],0)
            X2_full = tf.concat([X2_full,X1,X2],0)
     

        # make similarity matrix
        self.sim = tf.linalg.matmul(self.W, self.W, transpose_b=True) + tf.linalg.diag(self.kappa)
        self.sim = self.sim/tf.math.maximum(tf.reduce_max(self.sim),1)



        # Make S: the similarity tensor of shape (# strings, #characters, # characters)
        S = tf.matmul( tf.matmul(X1_full,self.sim),tf.transpose(X2_full,perm=(0,2,1)))

        # store squared match coef
        match_sq = tf.square(self.match_decay)


        # initialize final kernel results
        k = tf.zeros((tf.shape(S)[0]),dtype=tf.float64)
        # initialize Kp for dynamic programming
        Kp = tf.ones(shape=tf.stack([tf.shape(S)[0], self.maxlen,self.maxlen]), dtype=tf.float64)
        
        # need to do 1st step
        Kp_temp = tf.multiply(S, Kp)
        Kp_temp = tf.reduce_sum(Kp_temp, -1)
        Kp_temp = tf.reduce_sum(Kp_temp, -1)
        Kp_temp = Kp_temp * match_sq
        # add to kernel result
        k = Kp_temp * self.order_coefs[0]


        # do all remaining steps
        for i in tf.range(self.max_subsequence_length-1):
            Kp_temp = tf.multiply(S, Kp)
            Kp_temp =  match_sq * Kp_temp
            Kp_temp = tf.matmul(Kp_temp,self.D)
            # save part required for next dynamic programming step
            Kp = tf.matmul(self.D,Kp_temp,transpose_a=True)
            Kp_temp = tf.multiply(S, Kp)
            Kp_temp = tf.reduce_sum(Kp_temp, -1)
            Kp_temp = tf.reduce_sum(Kp_temp, -1)
            Kp_temp = Kp_temp * match_sq
            # add to kernel result
            k += Kp_temp * self.order_coefs[i+1]

        k = tf.expand_dims(k,1)

        #put results into the right places in the gram matrix and normalize
        if self.symmetric:
            # if symmetric then only put in top triangle (inc diag)
            mask = tf.linalg.band_part(tf.ones((X1_shape,X2_shape),dtype=tf.int64), 0, -1)
            non_zero = tf.not_equal(mask, tf.constant(0, dtype=tf.int64))
            # Extracting the indices of upper triangle elements
            indices = tf.where(non_zero)
            out = tf.SparseTensor(indices,tf.squeeze(k),dense_shape=tf.cast((X1_shape,X2_shape),dtype=tf.int64))
            k_results = tf.sparse.to_dense(out)
            #add in mising elements (lower diagonal)
            k_results = k_results + tf.linalg.set_diag(tf.transpose(k_results),tf.zeros(X1_shape,dtype=tf.float64))
            # normalise
            X_diag_Ks = tf.linalg.diag_part(k_results)
            norm = tf.tensordot(X_diag_Ks, X_diag_Ks,axes=0)
            k_results = tf.divide(k_results, tf.sqrt(norm))
        else:
            # otherwise can just reshape into gram matrix
            # but first take extra kernel calcs off end of k

            # COULD SPEED THIS UP FOR PREDICTIONS, AS MANY NORM TERMS ALREADY IN GRAM

            X_diag_Ks = tf.reshape(k[X1_shape*X2_shape:X1_shape*X2_shape+X1_shape],(-1,))

            X2_diag_Ks = tf.reshape(k[-X2_shape:],(-1,))

            k = k[0:X1_shape*X2_shape]
            k_results = tf.reshape(k,[X1_shape,X2_shape])
            # normalise
            norm = tf.tensordot(X_diag_Ks, X2_diag_Ks,axes=0)
            k_results = tf.divide(k_results, tf.sqrt(norm))


        return k_results


    def _precalc(self):
        r"""
        Precalc D matrix as required for kernel calcs
        following notation from Beck (2017)
        """
        tril =  tf.linalg.band_part(tf.ones((self.maxlen,self.maxlen),dtype=tf.float64), -1, 0)
        # get upper triangle matrix of increasing intergers
        values = tf.TensorArray(tf.int32, size= self.maxlen)
        for i in tf.range(self.maxlen):
            values = values.write(i,tf.range(-i-1,self.maxlen-1-i)) 
        power = tf.cast(values.stack(),tf.float64)
        values.close()
        power = tf.linalg.band_part(power, 0, -1) - tf.linalg.band_part(power, 0, 0) + tril
        tril = tf.transpose(tf.linalg.band_part(tf.ones((self.maxlen,self.maxlen),dtype=tf.float64), -1, 0))-tf.eye(self.maxlen,dtype=tf.float64)
        return tf.pow(self.gap_decay*tril, power)
