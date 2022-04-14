import numpy as np

class NPStringKernel(object):
	"""
	Code based on https://github.com/beckdaniel/flakes
	We make following changes
	1) provide kernel normalization to make meaningful comparissons between strings of different lengths
	2) changed structure and conventions to match our Tree kernel implemenentation
	3) simplified to only allow one-hot encoidng of alphabet (i.e remove support for pre-trained embeddings)
	4) a collection of performance tweaks to improve vectorization
	"""

	def __init__(self, _gap_decay=1.0, _match_decay=1.0,
				  _order_coefs=[1.0], alphabet = [], maxlen=0,normalize=True):    
		self._gap_decay = _gap_decay
		self._match_decay = _match_decay
		self._order_coefs = _order_coefs
		self.alphabet = alphabet
		self.normalize = normalize
		# prepare one-hot representation of the alphabet to encode input strings
		try:
			self.embs, self.index = build_one_hot(self.alphabet)
		except Exception:
			raise Exception(" check input alphabet covers X")
		self.embs_dim = self.embs.shape[1]
		self.maxlen = maxlen
		
	def Kdiag(self,X):   
		# input of form X = np.array([[s1],[s2],[s3]])
		# check if string is not longer than max length
		observed_maxlen = max([len(x[0].split(" ")) for x in X])
		if  observed_maxlen > self.maxlen:
			raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")
		
		# if normalizing then diagonal will just be ones
		if self.normalize:
			return np.ones(X.shape[0])
		else:
			#otherwise have to calc
			self.D, self.dD_dgap = self._precalc()
			# We turn our inputs into lists of integers using one-hot embedding
			# additional zero padding if their length is smaller than max-len
			X = np.array([[pad(encode_string(x[0].split(" "), self.index),self.maxlen)] for x in X],dtype=np.int32)
			return self._diag_calculations(X)[0]

		
		
	def K(self, X, X2=None):
		# input of form X = np.array([[c c c c c],[c c],[c c c]])
		# check if symmetric (no provided X2), if so then only need to calc upper gram matrix 
		symmetric = True if (X2 is None) else False
		
		# check if input strings are longer than max allowed length
		observed_maxlen = max([len(x[0].split(" ")) for x in X])
		if not symmetric:
			observed_maxlen_2 = max([len(x[0].split(" ")) for x in X])
			observed_maxlen = max(observed_maxlen,observed_maxlen_2)
		if  observed_maxlen > self.maxlen:
			raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")
		self.D, self.dD_dgap = self._precalc()
		# Turn our inputs into lists of integers using one-hot embedding
		# additional zero padding if their length is smaller than max-len
		X = np.array([[pad(encode_string(x[0].split(" "), self.index),self.maxlen)] for x in X],dtype=np.int32)
		if symmetric:
			X2 = X
		else:
			X2 = np.array([[pad(encode_string(x[0].split(" "), self.index),self.maxlen)] for x in X2],dtype=np.int32)
		

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
		
		# All set up. Proceed with kernel matrix calculations
		for i, x1 in enumerate(X):
			for j, x2 in enumerate(X2):
				# if symmetric then only need to actually calc the upper gram matrix
				if symmetric and (j < i):
					k_results[i, j] = k_results[j, i]
					gap_grads[i, j] = gap_grads[j, i]
					match_grads[i, j] = match_grads[j, i]
					coef_grads[i, j,:] = coef_grads[j, i,:]
				else:
					k_result, gap_grad, match_grad, coef_grad = _k(x1[0],x2[0],self.embs, self.maxlen, self._gap_decay, self._match_decay, np.array(self._order_coefs),self.D,self.dD_dgap) 

					# Normalize kernel and gradients if required
					if self.normalize:
						if symmetric:
							k_result_norm, gap_grad_norm, match_grad_norm, coef_grad_norm = _normalize(k_result, gap_grad, match_grad,coef_grad,
																							X_diag_Ks[i], X_diag_Ks[j],
																							X_diag_gap_grads[i], X_diag_match_grads[i],X_diag_coef_grads[i],
																							X_diag_gap_grads[j], X_diag_match_grads[j],X_diag_coef_grads[j])
						else:
							k_result_norm, gap_grad_norm, match_grad_norm, coef_grad_norm = _normalize(k_result, gap_grad, match_grad,coef_grad,
																							X_diag_Ks[i], X2_diag_Ks[j],
																							X_diag_gap_grads[i], X_diag_match_grads[i],X_diag_coef_grads[i],
																							X2_diag_gap_grads[j], X2_diag_match_grads[j],X2_diag_coef_grads[j])
						k_results[i, j] = k_result_norm
						gap_grads[i, j] = gap_grad_norm
						match_grads[i, j] = match_grad_norm
						coef_grads[i, j] = np.array(coef_grad_norm) 
					else:
						k_results[i, j] = k_result
						gap_grads[i, j] = gap_grad
						match_grads[i, j] = match_grad
						coef_grads[i, j] = np.array(coef_grad)    
		return k_results, gap_grads, match_grads, coef_grads


	def _diag_calculations(self, X):
		"""
		Calculate the K(x,x) values first because
		they are used in normalization.
		This is pre-normalization (otherwise diag is just ones)
		This function is not to be called directly, as requires preprocessing on X
		"""
		# initialize return values
		k_result = np.zeros(shape=(len(X)))
		gap_grads = np.zeros(shape=(len(X)))
		match_grads = np.zeros(shape=(len(X)))
		coef_grads = np.zeros(shape=(len(X), len(self._order_coefs)))
		
		# All set up. Proceed with kernel matrix calculations
		for i, x1 in enumerate(X):
			result = _k(x1[0],x1[0],self.embs, self.maxlen, self._gap_decay, self._match_decay, np.array(self._order_coefs),self.D,self.dD_dgap)  
			k_result[i] = result[0]
			gap_grads[i] = result[1]
			match_grads[i] = result[2]
			coef_grads[i] = np.array(result[3])
		return (k_result,gap_grads,match_grads,coef_grads)



	def _precalc(self):
		# pecalc the quantities required for every kernel calc, D and dD_dgaps
		# following notation from Beck (2017)

		# Make D: a upper triangular matrix over decay powers.


		tril = np.tril(np.ones((self.maxlen,self.maxlen)))
		power = [[0]*i+list(range(0,self.maxlen-i)) for i in range(1,self.maxlen)]+[[0]*self.maxlen]
		power = np.array(power).reshape(self.maxlen,self.maxlen) + tril
		#tril = np.transpose(tril-np.tril(np.ones((self.maxlen,self.maxlen)),-len(self._order_coefs))) - np.eye(self.maxlen)
		tril = np.transpose(tril) - np.eye(self.maxlen)
		gaps = np.ones([self.maxlen, self.maxlen])*self._gap_decay
		D = (gaps * tril) ** power
		dD_dgap = ((gaps * tril) ** (power - 1.0)) * tril * power
		return D, dD_dgap

def _normalize(K_result, gap_grads, match_grads, coef_grads,diag_Ks_i,
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



def _k(  s1,  s2,  embs,  maxlen,  _gap_decay,  _match_decay, _order_coefs,D,dD_dgap):
	"""
	TF code for vecotrized kernel calc.
	Following notation from Beck (2017), i.e have tensors S,D,Kpp,Kp
	
	Input is two np lists (of length maxlen) of integers represents each character in the alphabet 
	calc kernel between these two string representations. 
	"""
	S = embs[s1] @ embs[s2].T
	# Main loop, where Kp, Kpp values are calculated.
	Kp = np.ones(shape=(len(_order_coefs), maxlen, maxlen))
	dKp_dgap = np.zeros(shape=(len(_order_coefs), maxlen, maxlen))
	dKp_dmatch = np.zeros(shape=(len(_order_coefs), maxlen, maxlen))
	match_sq = _match_decay * _match_decay


	for i in range(len(_order_coefs)-1):
		aux1 = S * Kp[i]
		aux2 = np.dot(aux1,D)
		Kpp = match_sq * aux2
		Kp[i + 1] = Kpp.T.dot(D).T

		daux1_dgap = S * dKp_dgap[i]
		daux2_dgap = daux1_dgap.dot(D) + aux1.dot(dD_dgap)
		dKpp_dgap = match_sq * daux2_dgap
		dKp_dgap[i + 1] = dKpp_dgap.T.dot(D).T + Kpp.T.dot(dD_dgap).T

		daux1_dmatch = S * dKp_dmatch[i]
		daux2_dmatch = daux1_dmatch.dot(D)
		dKpp_dmatch = (match_sq * daux2_dmatch) + (2 * _match_decay * aux2)
		dKp_dmatch[i + 1] = dKpp_dmatch.T.dot(D).T
			

			
	#Final calculation
	final_aux1 = S * Kp
	final_aux2 = np.sum(final_aux1, axis=1)
	final_aux3 = np.sum(final_aux2, axis=1)
	Ki = match_sq * final_aux3
	k = Ki.dot(_order_coefs)

	final_daux1_dgap = S * dKp_dgap
	final_daux2_dgap = np.sum(final_daux1_dgap, axis=1)
	final_daux3_dgap = np.sum(final_daux2_dgap, axis=1)
	dKi_dgap = match_sq * final_daux3_dgap
	dk_dgap = dKi_dgap.dot(_order_coefs)
	
	final_daux1_dmatch = S * dKp_dmatch
	final_daux2_dmatch = np.sum(final_daux1_dmatch, axis=1)
	final_daux3_dmatch = np.sum(final_daux2_dmatch, axis=1)
	dKi_dmatch = match_sq * final_daux3_dmatch + (2 * _match_decay *final_aux3)
	dk_dmatch = dKi_dmatch.dot(_order_coefs)

	dk_dcoefs = Ki
	return k, dk_dgap, dk_dmatch, dk_dcoefs   
		



# helper functions to perform operations on input strings

def pad(s, length):
	#pad out input strings to our maxlen
	new_s = np.zeros(length)
	new_s[:len(s)] = s
	return new_s



def encode_string(s, index):
	"""
	Transform a string in a list of integers.
	The ints correspond to indices in an
	embeddings matrix.
	"""
	return [index[symbol] for symbol in s]

def build_one_hot(alphabet):
	"""
	Build one-hot encodings for a given alphabet.
	"""
	dim = len(alphabet)
	embs = np.zeros((dim+1, dim))
	index = {}
	for i, symbol in enumerate(alphabet):
		embs[i+1, i] = 1.0
		index[symbol] = i+1
	return embs, index