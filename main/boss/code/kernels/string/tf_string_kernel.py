import numpy as np
import tensorflow as tf
import itertools
import math



class TFStringKernel(object):
    """
    Code to run the SSK of Moss et al. 2020 on a GPU
    """
    def __init__(self, _gap_decay=1.0, _match_decay=1.0,batch_size=1000,
                  _order_coefs=[1.0], alphabet = [], maxlen=0,normalize=True):    
        self._gap_decay = _gap_decay
        self._match_decay = _match_decay
        self._order_coefs = _order_coefs
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.normalize = normalize
        # build a lookup table of the alphabet to encode input strings
        self.table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(["PAD"]+alphabet),
                values=tf.constant(range(0,len(alphabet)+1)),),default_value=0)
        self.maxlen = maxlen

        
    def Kdiag(self,X):   
        # X looks like np.array([[s1],[s2],[s3]]) where s1 is a string with spaces between characters
        # check if string is not longer than max length
        observed_maxlen = max([len(x[0].split(" ")) for x in X])
        if  observed_maxlen > self.maxlen:
            raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")
        # if normalizing then diagonal will just be ones
        if self.normalize:
            return np.ones(X.shape[0])
        else:
            #otherwise have to calc
            # first split up strings and pad to fixed length and prep for gpu
            # pad until all same length
            X = tf.strings.split(tf.squeeze(tf.convert_to_tensor(X),1)).to_tensor("PAD")
            # pad until all have length of self.maxlen
            if X.shape[1]<self.maxlen:
                paddings = tf.constant([[0, 0,], [0, self.maxlen-X.shape[1]]])
                X = tf.pad(X, paddings, "CONSTANT",constant_values="PAD") 
            # X has shape (#strings,# characters in longest string)
            # now map from chars to indicies
            X = self.table.lookup(X)
            return self._diag_calculations(X)[0]

    def K(self, X, X2=None):
        # input of form X = np.array([[s1],[s2],[s3]])
        # check if symmetric (no provided X2), if so then only need to calc upper gram matrix 
        symmetric = True if (X2 is None) else False
        
        # check if input strings are longer than max allowed length
        observed_maxlen = max([len(x[0].split(" ")) for x in X])
        if not symmetric:
            observed_maxlen_2 = max([len(x[0].split(" ")) for x in X])
            observed_maxlen = max(observed_maxlen,observed_maxlen_2)
        if  observed_maxlen > self.maxlen:
            raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")
        
        # Turn our inputs into lists of integers using one-hot embedding
        # pad until all same length
        X = tf.strings.split(tf.squeeze(tf.convert_to_tensor(X),1)).to_tensor("PAD")
        # pad until all have length of self.maxlen
        if X.shape[1]<self.maxlen:
            paddings = tf.constant([[0, 0,], [0, self.maxlen-X.shape[1]]])
            X = tf.pad(X, paddings, "CONSTANT",constant_values="PAD") 
        X = self.table.lookup(X)
        if symmetric:
            X2 = X
        else:
            # pad until all same length
            X2 = tf.strings.split(tf.squeeze(tf.convert_to_tensor(X2),1)).to_tensor("PAD")
            # pad until all have length of self.maxlen
            if X2.shape[1]<self.maxlen:
                paddings = tf.constant([[0, 0,], [0, self.maxlen-X2.shape[1]]])
                X2 = tf.pad(X2, paddings, "CONSTANT",constant_values="PAD") 
            X2 = self.table.lookup(X2)

        # Make D: a upper triangular matrix over decay powers.
        tf_tril =  tf.linalg.band_part(tf.ones((self.maxlen,self.maxlen),dtype=tf.float64), -1, 0)
        power = [[0]*i+list(range(0,self.maxlen-i)) for i in range(1,self.maxlen)]+[[0]*self.maxlen]
        tf_power=tf.constant(np.array(power).reshape(self.maxlen,self.maxlen), dtype=tf.float64) + tf_tril
        tf_tril = tf.transpose(tf_tril)-tf.eye(self.maxlen,dtype=tf.float64)
        tf_gap_decay = tf.constant(self._gap_decay,dtype=tf.float64)
        gaps = tf.fill([self.maxlen, self.maxlen], tf_gap_decay)
        
        D = tf.pow(gaps*tf_tril, tf_power)
        dD_dgap = tf.pow((tf_tril * gaps), (tf_power - 1.0)) * tf_tril * tf_power

        #if needed calculate the values needed for normalization
        if self.normalize:
            X_diag_Ks, X_diag_gap_grads, X_diag_match_grads, X_diag_coef_grads = self._diag_calculations(X)
            if not symmetric:
                X2_diag_Ks, X2_diag_gap_grads, X2_diag_match_grads, X2_diag_coef_grads = self._diag_calculations(X2)

        # Initialize return values
        k_results = np.zeros(shape=(len(X), len(X2)))
        gap_grads = np.zeros(shape=(len(X), len(X2)))
        match_grads = np.zeros(shape=(len(X), len(X2)))
        coef_grads = np.zeros(shape=(len(X), len(X2), len(self._order_coefs)))
        


        # prepare batches to send to _k
        # get indicies of all possible pairings from X and X2
        # this way allows maximum number of kernel calcs to be squished onto the GPU (rather than just doing individual rows of gram)
        tuples = list(itertools.product(range(X.shape[0]), range(X2.shape[0])))
        # if  symmetric only need to calc upper gram matrix 
        if symmetric:
            tuples = [t for t in tuples if t[0]<=t[1]]       
        num_batches = math.ceil(len(tuples)/self.batch_size)
        for i in range(num_batches):
            tuples_batch = tuples[self.batch_size*i:self.batch_size*(i+1)]
            X_batch_indicies = [t[0] for t in tuples_batch]
            X2_batch_indicies = [t[1] for t in tuples_batch]
            # collect strings for this batch
            X_batch = tf.gather(X,X_batch_indicies,axis=0)

            X2_batch = tf.gather(X2,X2_batch_indicies,axis=0)
            result = self._k(X_batch, X2_batch,D,dD_dgap)
            # this bit is probably slow, should vectorize
            # put results into the right places in the return values and normalize if required
            for i in range(0,len(tuples_batch)):
                if not self.normalize:
                    k_results[tuples_batch[i][0],tuples_batch[i][1]] =result[0][i] 
                    gap_grads[tuples_batch[i][0],tuples_batch[i][1]] =result[1][i] 
                    match_grads[tuples_batch[i][0],tuples_batch[i][1]] =result[2][i] 
                    coef_grads[tuples_batch[i][0],tuples_batch[i][1],:] =result[3][i] 
                else:
                    if symmetric:
                        k_result_norm, gap_grad_norm, match_grad_norm, coef_grad_norm = self._normalize(result[0][i], result[1][i], result[2][i] ,result[3][i],
                                                                                            X_diag_Ks[tuples_batch[i][0]], X_diag_Ks[tuples_batch[i][1]],
                                                                                            X_diag_gap_grads[tuples_batch[i][0]], X_diag_match_grads[tuples_batch[i][0]],X_diag_coef_grads[tuples_batch[i][0]],
                                                                                            X_diag_gap_grads[tuples_batch[i][1]], X_diag_match_grads[tuples_batch[i][1]],X_diag_coef_grads[tuples_batch[i][1]])
                    else:
                        k_result_norm, gap_grad_norm, match_grad_norm, coef_grad_norm = self._normalize(result[0][i], result[1][i], result[2][i] ,result[3][i],
                                                                                            X_diag_Ks[tuples_batch[i][0]], X2_diag_Ks[tuples_batch[i][1]],
                                                                                            X_diag_gap_grads[tuples_batch[i][0]], X_diag_match_grads[tuples_batch[i][0]],X_diag_coef_grads[tuples_batch[i][0]],
                                                                                            X2_diag_gap_grads[tuples_batch[i][1]], X2_diag_match_grads[tuples_batch[i][1]],X2_diag_coef_grads[tuples_batch[i][1]])

                    k_results[tuples_batch[i][0],tuples_batch[i][1]] = k_result_norm 
                    gap_grads[tuples_batch[i][0],tuples_batch[i][1]] = gap_grad_norm
                    match_grads[tuples_batch[i][0],tuples_batch[i][1]] = match_grad_norm
                    coef_grads[tuples_batch[i][0],tuples_batch[i][1],:] = coef_grad_norm 



        # if symmetric then need to fill in rest of matrix (lower gram)
        if symmetric:
            for i in range(X.shape[0]):
                for j in range(i):
                    k_results[i, j] = k_results[j, i]
                    gap_grads[i, j] = gap_grads[j, i]
                    match_grads[i, j] = match_grads[j, i]
                    coef_grads[i, j,:] = coef_grads[j, i,:]
        return k_results, gap_grads, match_grads, coef_grads


        


    def _diag_calculations(self, X):
        """
        Calculate the K(x,x) values first because
        they are used in normalization.
        This is pre-normalization (otherwise diag is just ones)
        This function is not to be called directly, as requires preprocessing on X
        """


        # Make D: a upper triangular matrix over decay powers.
        tf_tril =  tf.linalg.band_part(tf.ones((self.maxlen,self.maxlen),dtype=tf.float64), -1, 0)
        power = [[0]*i+list(range(0,self.maxlen-i)) for i in range(1,self.maxlen)]+[[0]*self.maxlen]
        tf_power=tf.constant(np.array(power).reshape(self.maxlen,self.maxlen), dtype=tf.float64) + tf_tril
        tf_tril = tf.transpose(tf_tril)-tf.eye(self.maxlen,dtype=tf.float64)
        tf_gap_decay = tf.constant(self._gap_decay,dtype=tf.float64)
        gaps = tf.fill([self.maxlen, self.maxlen], tf_gap_decay)
        D = tf.pow(gaps*tf_tril, tf_power)
        dD_dgap = tf.pow((tf_tril * gaps), (tf_power - 1.0)) * tf_tril * tf_power

        # initialize return values
        k_result = np.zeros(shape=(len(X)))
        gap_grads = np.zeros(shape=(len(X)))
        match_grads = np.zeros(shape=(len(X)))
        coef_grads = np.zeros(shape=(len(X), len(self._order_coefs)))               

        # All set up. Proceed with kernel matrix calculations (in batches if required) 
        num_batches = math.ceil(len(X)/self.batch_size)
        for i in range(num_batches):
            X_batch = X[self.batch_size*i:self.batch_size*(i+1),:]
            result = self._k(X_batch, X_batch,D,dD_dgap)
            k_result[self.batch_size*i:self.batch_size*(i+1)] = result[0].numpy()
            gap_grads [self.batch_size*i:self.batch_size*(i+1)] = result[1].numpy()
            match_grads [self.batch_size*i:self.batch_size*(i+1)] = result[2].numpy()
            coef_grads [self.batch_size*i:self.batch_size*(i+1),:] = result[3].numpy()
        return (k_result,gap_grads,match_grads,coef_grads)
        


    def _k(self, X1, X2, D, dD_dgap):
        """
        TF code for vectorized kernel calc.
        Following notation from Beck (2017), i.e have tensors S,D,Kpp,Kp
        Input is two tensors of shape (# strings , # characters)
        and we calc the pair-wise kernel calcs between the elements (i.e n kern calcs for two lists of length n)
        D and dD_gap are the matricies than unroll the recursion and allow vecotrizaiton
        """

        # init
        tf_gap_decay = tf.constant(self._gap_decay,dtype=tf.float64)
        tf_match_decay = tf.constant(self._match_decay,dtype=tf.float64)
        tf_order_coefs = tf.convert_to_tensor(self._order_coefs, dtype=tf.float64)

        # Strings will be represented as matrices of
        # one-hot embeddings and the similarity is just the dot product. (ie. checking for matches of characters)
        # turn into one-hot  i.e. shape (# strings, #characters+1, alphabet size)
        X1 = tf.one_hot(X1,len(self.alphabet)+1,dtype=tf.float64)
        X2 = tf.one_hot(X2,len(self.alphabet)+1,dtype=tf.float64)
        # remove the ones in the first column that encode the padding (i.e we dont want them to count as a match)
        paddings = tf.constant([[0, 0], [0, 0],[0,len(self.alphabet)]])
        X1 = X1 - tf.pad(tf.expand_dims(X1[:,:,0], 2),paddings,"CONSTANT")
        X2 = X2 - tf.pad(tf.expand_dims(X2[:,:,0], 2),paddings,"CONSTANT")
        
        # store squared match coef
        match_sq = tf.square(tf_match_decay)
        # Make S: the similarity tensor of shape (# strings, #characters, # characters)
        S = tf.matmul( X1,tf.transpose(X2,perm=(0,2,1)))
        # Main loop, where Kp, Kpp values and gradients are calculated.
        Kp = []
        dKp_dgap = []
        dKp_dmatch = []
        Kp.append(tf.ones(shape=(X1.shape[0],self.maxlen, self.maxlen), dtype=tf.float64))
        dKp_dgap.append(tf.zeros(shape=(X1.shape[0],self.maxlen, self.maxlen),dtype=tf.float64))
        dKp_dmatch.append(tf.zeros(shape=(X1.shape[0],self.maxlen, self.maxlen),dtype=tf.float64))

        for i in range(len(tf_order_coefs)-1):
            # calc subkernels for each subsequence length
            aux = tf.multiply(S, Kp[i])
            aux1 = tf.reshape(aux, tf.stack([X1.shape[0] * self.maxlen, self.maxlen]))
            aux2 = tf.matmul(aux1, D)
            aux = aux2 * match_sq
            aux = tf.reshape(aux, tf.stack([X1.shape[0], self.maxlen, self.maxlen]))
            aux = tf.transpose(aux, perm=[0, 2, 1])
            aux3 = tf.reshape(aux, tf.stack([X1.shape[0] * self.maxlen, self.maxlen]))
            aux = tf.matmul(aux3, D)
            aux = tf.reshape(aux, tf.stack([X1.shape[0], self.maxlen, self.maxlen]))
            Kp.append(tf.transpose(aux, perm=[0, 2, 1]))
            

            aux = tf.multiply(S, dKp_dgap[i])
            aux = tf.reshape(aux, tf.stack([X1.shape[0] *self.maxlen,self.maxlen]))
            aux = tf.matmul(aux, D) + tf.matmul(aux1, dD_dgap)
            aux = aux * match_sq
            aux = tf.reshape(aux, tf.stack([X1.shape[0],self.maxlen,self.maxlen]))
            aux = tf.transpose(aux, perm=[0, 2, 1])
            aux = tf.reshape(aux, tf.stack([X1.shape[0] *self.maxlen,self.maxlen]))
            aux = tf.matmul(aux, D) + tf.matmul(aux3, dD_dgap)
            aux = tf.reshape(aux, tf.stack([X1.shape[0],self.maxlen,self.maxlen]))
            dKp_dgap.append(tf.transpose(aux, perm=[0, 2, 1]))
            

            aux = tf.multiply(S, dKp_dmatch[i])
            aux = tf.reshape(aux, tf.stack([X1.shape[0] *self.maxlen,self.maxlen]))
            aux = tf.matmul(aux, D)
            aux = (aux * match_sq) + (2 * tf_match_decay * aux2)
            aux = tf.reshape(aux, tf.stack([X1.shape[0],self.maxlen,self.maxlen]))
            aux = tf.transpose(aux, perm=[0, 2, 1])
            aux = tf.reshape(aux, tf.stack([X1.shape[0] *self.maxlen,self.maxlen]))
            aux = tf.matmul(aux, D)
            aux = tf.reshape(aux, tf.stack([X1.shape[0],self.maxlen,self.maxlen]))
            dKp_dmatch.append(tf.transpose(aux, perm=[0, 2, 1]))


        Kp = tf.stack(Kp)
        dKp_dgap = tf.stack(dKp_dgap)
        dKp_dmatch = tf.stack(dKp_dmatch)

        # Final calculation. We gather all Kps and
        # multiply then by their coeficients.

        # get k
        aux = tf.multiply(S, Kp)
        aux = tf.reduce_sum(aux, 2)
        sum2 = tf.reduce_sum(aux, 2, keepdims=True)
        Ki = tf.multiply(sum2, match_sq)
        Ki = tf.squeeze(Ki, [2])
        # reshape in case batch size 1
        k = tf.reshape(tf.squeeze(tf.matmul(tf.reshape(tf_order_coefs,(1,-1)), Ki)),(X1.shape[0],))

        # get gap decay grads
        aux = tf.multiply(S, dKp_dgap)
        aux = tf.reduce_sum(aux, 2)
        aux = tf.reduce_sum(aux, 2, keepdims=True)
        aux = tf.multiply(aux, match_sq)
        aux = tf.squeeze(aux, [2])
        dk_dgap = tf.reshape(tf.squeeze(tf.matmul(tf.reshape(tf_order_coefs,(1,-1)), aux)),(X1.shape[0],))

        # get match decay grads
        aux = tf.multiply(S, dKp_dmatch)
        aux = tf.reduce_sum(aux, 2)
        aux = tf.reduce_sum(aux, 2, keepdims=True)
        aux = tf.multiply(aux, match_sq) + (2 * tf_match_decay * sum2)
        aux = tf.squeeze(aux, [2])
        dk_dmatch = tf.reshape( tf.squeeze(tf.matmul(tf.reshape(tf_order_coefs,(1,-1)), aux)),(X1.shape[0],))

        # get coefs grads
        dk_dcoefs = tf.transpose(Ki)
        return (k, dk_dgap, dk_dmatch, dk_dcoefs, Ki)

    def _normalize(self, K_result, gap_grads, match_grads, coef_grads,diag_Ks_i,
                    diag_Ks_j, diag_gap_grads_i, diag_match_grads_i, diag_coef_grads_i,
                    diag_gap_grads_j, diag_match_grads_j, diag_coef_grads_j,):
        """
        Normalize the kernel and kernel derivatives.
        Following the derivation of Beck (2015)
        """
        norm = diag_Ks_i * diag_Ks_j
        sqrt_norm = np.sqrt(norm)
        K_norm = K_result / sqrt_norm
        
                
        diff_gap = ((diag_gap_grads_i * diag_Ks_j) +
                       (diag_Ks_i * diag_gap_grads_j))
        diff_gap /= 2 * norm

        gap_grads_norm = ((gap_grads / sqrt_norm) -
                        (K_norm * diff_gap))
        
        diff_match = ((diag_match_grads_i * diag_Ks_j) +
                       (diag_Ks_i * diag_match_grads_j))
        diff_match /= 2 * norm
        match_grads_norm = ((match_grads / sqrt_norm) -
                        (K_norm * diff_match))
        

        diff_coef = ((diag_coef_grads_i * diag_Ks_j) +
                       (diag_Ks_i * diag_coef_grads_j))

        diff_coef /= 2 * norm

        coef_grads_norm = ((coef_grads / sqrt_norm) -
                        (K_norm * diff_coef))

        return K_norm, gap_grads_norm, match_grads_norm, coef_grads_norm

