from gpflow.kernels import Kernel
from gpflow.utilities import positive
from gpflow import Parameter
import tensorflow as tf
from tensorflow_probability import bijectors as tfb

class Batch_FSSK(Kernel):
    """
    Code to run the flexible SSK with gpflow
    
   with hyperparameters:
    1) match_decay float
        decrease the contribution of long subsequences
    2) gap_decay float
        decrease the contribtuion of subsequences with large gaps (penalize non-contiguous)
    3) max_subsequence_length int 
        largest subsequence considered

    TO DO 
    """

    def __init__(self,rank=1,active_dims=[0],gap_decay=0.1, match_decay=0.9,max_subsequence_length=3,
                 alphabet = [], maxlen=0, batch_size=100):
        super().__init__(active_dims=active_dims)
        # constrain decay kernel params to between 0 and 1
        self.logistic_gap = tfb.Chain([tfb.Shift(tf.cast(0,tf.float64))(tfb.Scale(tf.cast(1,tf.float64))),tfb.Sigmoid()])
        self.logisitc_match = tfb.Chain([tfb.AffineScalar(shift=tf.cast(0,tf.float64),scale=tf.cast(1,tf.float64)),tfb.Sigmoid()])
        
        self.gap_decay_param= Parameter(gap_decay, transform=self.logistic_gap ,name="gap_decay")
        self.match_decay_param = Parameter(match_decay, transform=self.logisitc_match,name="match_decay")

        self.W_param = Parameter(0.1 * tf.ones((len(alphabet), rank)),name="W")
        
        self.kappa_param = Parameter(tf.ones(len(alphabet)), transform=positive(),name="kappa")


        # use will use copies of the kernel params to stop building expensive computation graph
        # we instead efficientely calculate gradients using dynamic programming
        # These params are updated at every call to K and K_diag (to check if parameters have been updated)
        self.match_decay = self.match_decay_param.numpy()
        self.gap_decay = self.gap_decay_param.numpy()
        self.kappa = self.kappa_param.numpy()
        self.W = self.W_param.numpy()
        self.match_decay_unconstrained = self.match_decay_param.unconstrained_variable.numpy()
        self.gap_decay_unconstrained = self.gap_decay_param.unconstrained_variable.numpy()
        self.kappa_unconstrained = self.kappa_param.unconstrained_variable.numpy()
        self.W_unconstrained = self.W_param.unconstrained_variable.numpy()

        # store additional kernel parameters
        self.max_subsequence_length = tf.constant(max_subsequence_length)
        self.alphabet =  tf.constant(alphabet)
        self.alphabet_size=tf.shape(self.alphabet)[0]
        self.maxlen =  tf.constant(maxlen)
        self.batch_size = tf.constant(batch_size)
        self.rank = tf.constant(rank)

        # build a lookup table of the alphabet to encode input strings
        self.table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(["PAD"]+alphabet),
                values=tf.constant(range(0,len(alphabet)+1)),),default_value=0)

        # initialize helful construction matricies to be lazily computed once needed
        self.D = None
        self.dD_dgap = None


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
        # turn into one-hot  i.e. shape (# strings, #characters+1, alphabet size)
        X1 = tf.strings.split(tf.squeeze(X1,1)).to_tensor("PAD",shape=[None,self.maxlen])
        X1 = self.table.lookup(X1)
        # keep track of original input sizes
        X1_shape = tf.shape(X1)[0]
        X1 = tf.one_hot(X1,self.alphabet_size+1,dtype=tf.float64)
        if X2 is None:
            X2 = X1
            X2_shape = X1_shape
            self.symmetric = True
        else:
            self.symmetric = False
            X2 = tf.strings.split(tf.squeeze(X2,1)).to_tensor("PAD",shape=[None,self.maxlen])
            X2 = self.table.lookup(X2)
            X2_shape = tf.shape(X2)[0]
            X2 = tf.one_hot(X2,self.alphabet_size+1,dtype=tf.float64)
  
        # prep the decay tensors 
        self._precalc()
      


        # combine all target strings and remove the ones in the first column that encode the padding (i.e we dont want them to count as a match)
        X_full = tf.concat([X1,X2],0)[:,:,1:]

        # get indicies of all possible pairings from X and X2
        # this way allows maximum number of kernel calcs to be squished onto the GPU (rather than just doing individual rows of gram)
        indicies_2, indicies_1 = tf.meshgrid(tf.range(0, X1_shape ),tf.range(X1_shape , tf.shape(X_full)[0]))
        indicies = tf.concat([tf.reshape(indicies_1,(-1,1)),tf.reshape(indicies_2,(-1,1))],axis=1)
        if self.symmetric:
            # if symmetric then only calc upper matrix (fill in rest later)
            indicies = tf.boolean_mask(indicies,tf.greater_equal(indicies[:,1]+ X1_shape ,indicies[:,0]))
        else:
            # if not symmetric need to calculate some extra kernel evals for the normalization later on
            indicies = tf.concat([indicies,tf.tile(tf.expand_dims(tf.range(tf.shape(X_full)[0]),1),(1,2))],0)

        # make similarity matrix
        self.sim = tf.linalg.matmul(self.W, self.W, transpose_b=True) + tf.linalg.diag(self.kappa)
        self.sim = self.sim/tf.math.maximum(tf.reduce_max(self.sim),1)
        


        # make kernel calcs in batches
        num_batches = tf.cast(tf.math.ceil(tf.shape(indicies)[0]/self.batch_size),dtype=tf.int32)
        k_split =  tf.TensorArray(tf.float64, size=num_batches,clear_after_read=False,infer_shape=False)
        
    

        # iterate through batches
        for j in tf.range(num_batches):
            # collect strings for this batch
            indicies_batch = indicies[self.batch_size*j:self.batch_size*(j+1)]
            X_batch = tf.gather(X_full,indicies_batch[:,0],axis=0)
            X2_batch = tf.gather(X_full,indicies_batch[:,1],axis=0)

            # collect results for the batch
            result = self.kernel_calc(X_batch,X2_batch)
            k_split = k_split.write(j,result)

        # combine batch results
        k = tf.expand_dims(k_split.concat(),1)
        k_split.close()

        # put results into the right places in the gram matrix and normalize
        if self.symmetric:
            # if symmetric then only put in top triangle (inc diag)
            mask = tf.linalg.band_part(tf.ones((X1_shape,X2_shape),dtype=tf.int64), 0, -1)
            non_zero = tf.not_equal(mask, tf.constant(0, dtype=tf.int64))
            
            # Extracting the indices of upper triangle elements
            indices = tf.where(non_zero)
            out = tf.SparseTensor(indices,tf.squeeze(k),dense_shape=tf.cast((X1_shape,X2_shape),dtype=tf.int64))
            k_results = tf.sparse.to_dense(out)
            
            # add in mising elements (lower diagonal)
            k_results = k_results + tf.linalg.set_diag(tf.transpose(k_results),tf.zeros(X1_shape,dtype=tf.float64))
            
            # normalise
            X_diag_Ks = tf.linalg.diag_part(k_results)
            norm = tf.tensordot(X_diag_Ks, X_diag_Ks,axes=0)
            k_results = tf.divide(k_results, tf.sqrt(norm))
        else:
            # otherwise can just reshape into gram matrix
            # but first take extra kernel calcs off end of k and use them to normalise
            X_diag_Ks = tf.reshape(k[X1_shape*X2_shape:X1_shape*X2_shape+X1_shape],(-1,))
            X2_diag_Ks = tf.reshape(k[-X2_shape:],(-1,))
            k = k[0:X1_shape*X2_shape]
            k_results = tf.transpose(tf.reshape(k,[X2_shape,X1_shape]))
            
            # normalise
            norm = tf.tensordot(X_diag_Ks, X2_diag_Ks,axes=0)
            k_results = tf.divide(k_results, tf.sqrt(norm))

        return k_results


    def _precalc(self):
        r"""
        Update stored kernel params (incase they have changed)
        and precalc D and dD_dgap as required for kernel calcs
        following notation from Beck (2017)
        """
        self.match_decay = self.match_decay_param.numpy()
        self.gap_decay = self.gap_decay_param.numpy()
        self.kappa = self.kappa_param.numpy()
        self.W = self.W_param.numpy()
        self.match_decay_unconstrained = self.match_decay_param.unconstrained_variable.numpy()
        self.gap_decay_unconstrained = self.gap_decay_param.unconstrained_variable.numpy()
        self.kappa_unconstrained = self.kappa_param.unconstrained_variable.numpy()
        self.W_unconstrained = self.W_param.unconstrained_variable.numpy()
        
        # get upper triangle matrix of increasing intergers
        tril =  tf.linalg.band_part(tf.ones((self.maxlen,self.maxlen),dtype=tf.float64), -1, 0)
        values = tf.TensorArray(tf.int32, size= self.maxlen)
        for i in tf.range(self.maxlen):
            values = values.write(i,tf.range(-i-1,self.maxlen-1-i)) 
        power = tf.cast(values.stack(),tf.float64)
        values.close()
        power = tf.linalg.band_part(power, 0, -1) - tf.linalg.band_part(power, 0, 0) + tril
        tril = tf.transpose(tf.linalg.band_part(tf.ones((self.maxlen,self.maxlen),dtype=tf.float64), -1, 0))-tf.eye(self.maxlen,dtype=tf.float64)
        gaps = tf.fill([self.maxlen, self.maxlen],self.gap_decay)
        
        self.D = tf.pow(gaps*tril, power)
        self.dD_dgap = tf.pow((tril * gaps), (power - 1.0)) * tril * power



    @tf.custom_gradient
    def kernel_calc(self,X_batch,X2_batch):

        # fake computations to ensure we take the custom gradients for kernel params
        _ = tf.square(self.gap_decay_param)
        _ = tf.square(self.match_decay_param)
        _ = tf.square(self.kappa_param)
        _ = tf.square(self.W_param)


        if self.symmetric:
            k, dk_dgap, dk_dmatch, dk_dkappa, dk_dW = self.kernel_calc_with_grads(X_batch,X2_batch)
        else:
            k = tf.stop_gradient(self.kernel_calc_without_grads(X_batch,X2_batch))


        def grad(dy, variables=None):
            # get gradients of unconstrained params
            grads= {}
            if self.symmetric:
                grads['gap_decay:0'] = tf.reduce_sum(tf.multiply(dy,dk_dgap*tf.math.exp(self.logistic_gap.forward_log_det_jacobian(self.gap_decay_unconstrained,0))))
                grads['match_decay:0'] = tf.reduce_sum(tf.multiply(dy,dk_dmatch*tf.math.exp(self.logisitc_match.forward_log_det_jacobian(self.match_decay_unconstrained,0))))
                dy_tiled = tf.tile(tf.expand_dims(dy,1),(1,self.alphabet_size))
                grads['kappa:0'] = tf.reduce_sum(tf.multiply(dy_tiled,dk_dkappa*tf.math.exp(positive().forward_log_det_jacobian(self.kappa_unconstrained,0))),0)
                dy_tiled = tf.tile(tf.expand_dims(dy,1),(1,self.alphabet_size*self.rank)) 
                grads_temp = tf.reduce_sum(tf.multiply(dy_tiled,dk_dW),0)
                grads['W:0'] = tf.reshape(grads_temp,(-1,self.alphabet_size,self.rank))
                gradient = [grads[v.name] for v in variables]
            else:
                gradient = [None for v in variables]
            return ((None,None),gradient)


        return k, grad

    def kernel_calc_without_grads(self,X_batch,X2_batch):

        # store squared match coef for easier calc later
        match_sq = tf.square(self.match_decay)

        # Make S: the similarity tensor of shape (# strings, #characters, # characters)
        S = tf.matmul( tf.matmul(X_batch,self.sim),tf.transpose(X2_batch,perm=(0,2,1)))
        

        # initialize Kp for dynamic programming
        Kp = tf.ones(shape=tf.stack([tf.shape(S)[0], self.maxlen,self.maxlen]), dtype=tf.float64)
        
        # do all remaining steps
        for i in tf.range(self.max_subsequence_length-1):
            Kp = tf.multiply(S, Kp)
            Kp =  match_sq * Kp
            Kp = tf.matmul(Kp,self.D)
            Kp = tf.matmul(self.D,Kp,transpose_a=True)

        # final kernel calc
        Kp = tf.multiply(S, Kp)
        Kp = tf.reduce_sum(Kp, -1)
        Kp = tf.reduce_sum(Kp, -1)
        Kp = Kp * match_sq
        k = tf.expand_dims(Kp,1)



        return k

    

    def kernel_calc_with_grads(self,X_batch,X2_batch):
        # store squared match coef for easier calc later

        match_sq = tf.square(self.match_decay)

        # Make S: the similarity tensor of shape (# strings, #characters, # characters)
        S = tf.matmul( tf.matmul(X_batch,self.sim),tf.transpose(X2_batch,perm=(0,2,1)))


        # build dS_dkappa for each elemnt in kappa (same number as alphabet size)
        dS_dkappa = tf.TensorArray(tf.float64, size=self.alphabet_size,clear_after_read=False)
        for i in range(0,len(self.alphabet)):
            temp_sim = tf.cast(tf.scatter_nd([[i,i]], [1], self.sim.shape),dtype=tf.float64)
            dS_dkappa = dS_dkappa .write(i,tf.matmul( tf.matmul(X_batch,temp_sim),tf.transpose(X2_batch,perm=(0,2,1))))
        dS_dkappa = dS_dkappa.stack()
        
        # build dS_dW for each elemnt in W (same number as alphabet size * rank)
        dS_dW = tf.TensorArray(tf.float64, size=self.alphabet_size*self.rank,clear_after_read=False)
        for j in range(self.rank):
            for i in range(0,len(self.alphabet)):
                dW = tf.cast(tf.scatter_nd([[i,j]], [1], self.W.shape),dtype=tf.float64)
                temp_sim = tf.linalg.matmul(dW, self.W, transpose_b=True) + tf.linalg.matmul(self.W, dW, transpose_b=True)
                dS_dW = dS_dW .write(i*self.rank+j,tf.matmul( tf.matmul(X_batch,temp_sim),tf.transpose(X2_batch,perm=(0,2,1))))
        dS_dW = dS_dW.stack()

        dS_dcomb = tf.concat([dS_dkappa,dS_dW],0)




        Kp = tf.ones(shape=tf.stack([tf.shape(S)[0], self.maxlen,self.maxlen]), dtype=tf.float64)
        dKp_dgap = tf.zeros(shape=tf.stack([tf.shape(S)[0], self.maxlen,self.maxlen]), dtype=tf.float64)
        dKp_dmatch = tf.zeros(shape=tf.stack([tf.shape(S)[0], self.maxlen,self.maxlen]), dtype=tf.float64)
        dKp_dcomb = tf.zeros(shape=tf.stack([self.alphabet_size*(1+self.rank),tf.shape(S)[0], self.maxlen,self.maxlen]), dtype=tf.float64)
    

        # calculate dynamic programs
        for i in tf.range(self.max_subsequence_length-1):
            dKp_dcomb = tf.multiply(S, dKp_dcomb)
            dKp_dcomb += tf.multiply(dS_dcomb, Kp) 
            Kp = tf.multiply(S, Kp)
            Kp_temp =  match_sq * Kp
            Kp = tf.matmul(Kp_temp,self.D)
            dKp_dgap =  tf.multiply(S, dKp_dgap)
            dKp_dgap = dKp_dgap * match_sq
            dKp_dgap = tf.matmul(dKp_dgap,self.D)
            dKp_dgap = dKp_dgap + tf.matmul(Kp_temp,self.dD_dgap)
            dKp_dgap = tf.matmul(self.D,dKp_dgap,transpose_a=True)
            dKp_dgap = tf.matmul(self.dD_dgap,Kp,transpose_a=True) + dKp_dgap
            Kp = tf.matmul(self.D,Kp,transpose_a=True)
            dKp_dmatch =  tf.multiply(S, dKp_dmatch)
            dKp_dmatch = dKp_dmatch * match_sq
            dKp_dmatch = tf.matmul(dKp_dmatch,self.D)
            dKp_dmatch = tf.matmul(self.D,dKp_dmatch,transpose_a=True)        
            dKp_dmatch = 2*tf.divide(Kp,self.match_decay) + dKp_dmatch
            dKp_dcomb = dKp_dcomb * match_sq
            dKp_dcomb = tf.matmul(dKp_dcomb,self.D)
            dKp_dcomb = tf.matmul(self.D,dKp_dcomb,transpose_a=True)
 

        # Final calculation. 
        dk_dcomb = tf.multiply(tf.expand_dims(S,0), dKp_dcomb)
        dk_dcomb = dk_dcomb + tf.multiply(dS_dcomb, tf.expand_dims(Kp,0))

        Kp = tf.multiply(S, Kp)
        Kp = tf.reduce_sum(Kp, -1)
        Kp = tf.reduce_sum(Kp, -1)
        k = tf.expand_dims(Kp * match_sq,1)

        dKp_dgap = tf.multiply(S, dKp_dgap)
        dKp_dgap = tf.reduce_sum(dKp_dgap, -1)
        dKp_dgap = tf.reduce_sum(dKp_dgap, -1)
        dk_dgap = tf.expand_dims(dKp_dgap * match_sq,1)

        dKp_dmatch = tf.multiply(S, dKp_dmatch)
        dKp_dmatch = tf.reduce_sum(dKp_dmatch, -1)
        dKp_dmatch = tf.reduce_sum(dKp_dmatch, -1)
        dKp_dmatch = dKp_dmatch * match_sq
        dk_dmatch = tf.expand_dims(dKp_dmatch + 2 * self.match_decay * Kp,1)

        dk_dcomb = tf.reduce_sum(dk_dcomb, -1)
        dk_dcomb = tf.reduce_sum(dk_dcomb, -1)
        dk_dcomb = dk_dcomb * match_sq
        dk_dcomb  = tf.expand_dims(dk_dcomb,1)

        dk_dkappa = dk_dcomb[:,:self.alphabet_size]
        dk_dW = dk_dcomb[:,self.alphabet_size:]


    


        return k, dk_dgap, dk_dmatch, dk_dkappa, dk_dW

