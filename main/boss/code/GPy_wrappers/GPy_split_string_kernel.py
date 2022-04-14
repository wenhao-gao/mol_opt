from GPy.kern import Kern
from GPy.core.parameterization import Param
import numpy as np
import sys
from paramz.transformations import Logexp
from .GPy_string_kernel import StringKernel

class SplitStringKernel(Kern):
	"""
	The split string kernel described by Moss (2020), where seperate string kernels
	are applied to each of m partitions of the string. Greatly improves scalability.
    kernel hyperparameters:
	1) match_decay float
		decrease the contribution of long subsequences
	2) gap_decay float
		decrease the contribtuion of subsequences with large gaps (penalize non-contiguous)
	3) order_coefs list(floats) 
		n-gram weights to help tune the signal coming from different sub-sequence lengths
	4) num_splits int
		parition string into this many roughly equal sized chunks

	We calculate gradients w.r.t kernel hyperparameters following Beck (2017)

	We recommend normalize = True to allow meaningful comparrison of strings of different length
	X is a numpy array of size (n,1) where each element is a string with characters seperated by spaces
	"""

	def __init__(self,gap_decay=1.0, match_decay=2.0, order_coefs=[1.0],
		alphabet = [], maxlen=0, num_splits=1,normalize = True):
		super(SplitStringKernel, self).__init__(1,None, "sk")
		self._name = "sk"
		self.num_splits = num_splits
		self.gap_decay = Param('Gap_decay', gap_decay,Logexp())
		self.match_decay = Param('Match_decay', match_decay,Logexp())
		self.order_coefs = Param('Order_coefs',order_coefs, Logexp())
		self.link_parameters(self.gap_decay, self.match_decay,self.order_coefs)
		
		self.alphabet = alphabet
		self.maxlen = maxlen
		self.normalize = normalize

		# make new kernels for each section
		self.kernels = []
		for i in range(0,num_splits-1):
			self.kernels.append(StringKernel(gap_decay=gap_decay,match_decay=match_decay, order_coefs=order_coefs,
				 alphabet = alphabet, maxlen=int((self.maxlen/self.num_splits)),normalize=normalize))
		# final kernel might be operating on slightly loinger string if maxlen/num_splits % !=0
		self.kernels.append(StringKernel(gap_decay=gap_decay,match_decay=match_decay, order_coefs=order_coefs,
			 alphabet = alphabet, maxlen=int((self.maxlen/self.num_splits)) + self.maxlen - self.num_splits*int((self.maxlen/self.num_splits)),normalize=normalize))
		#tie the params across the kernels 
		for kern in self.kernels:
			kern.unlink_parameter(kern.gap_decay)
			kern.gap_decay = self.gap_decay
			kern.unlink_parameter(kern.match_decay)
			kern.match_decay = self.match_decay
			kern.unlink_parameter(kern.order_coefs)
			kern.order_coefs = self.order_coefs



	def K(self, X, X2=None):
		"""
		split data and Add all kernels together.
		"""
		X_split = splitter(X,self.num_splits) 
		# update params stored in each kernel
		# store sum of k and grads
		if X2 is not None:
			X2_split = splitter(X2,self.num_splits)
			self.kernels[0].kernel._gap_decay = self.gap_decay[0]
			self.kernels[0].kernel._match_decay = self.match_decay[0]
			self.kernels[0].kernel._order_coefs = list(self.order_coefs.values)
			k, gap_grads, match_grads, coef_grads = self.kernels[0].kernel.K(X_split[0], X2_split[0])
			for i in range(1,len(self.kernels)):
				self.kernels[i].kernel._gap_decay = self.gap_decay[0]
				self.kernels[i].kernel._match_decay = self.match_decay[0]
				self.kernels[i].kernel._order_coefs = list(self.order_coefs.values)
				temp_k, temp_gap_grads, temp_match_grads, temp_coef_grads = self.kernels[i].kernel.K(X_split[i], X2_split[i])
				k+=temp_k
				gap_grads+=temp_gap_grads
				match_grads+=temp_match_grads
				coef_grads+=temp_coef_grads
		else:
			self.kernels[0].kernel._gap_decay = self.gap_decay[0]
			self.kernels[0].kernel._match_decay = self.match_decay[0]
			self.kernels[0].kernel._order_coefs = list(self.order_coefs.values)
			k, gap_grads, match_grads, coef_grads = self.kernels[0].kernel.K(X_split[0], None)
			for i in range(1,len(self.kernels)):
				self.kernels[i].kernel._gap_decay = self.gap_decay[0]
				self.kernels[i].kernel._match_decay = self.match_decay[0]
				self.kernels[i].kernel._order_coefs = list(self.order_coefs.values)
				temp_k, temp_gap_grads, temp_match_grads, temp_coef_grads = self.kernels[i].kernel.K(X_split[i],None)
				k+=temp_k
				gap_grads+=temp_gap_grads
				match_grads+=temp_match_grads
				coef_grads+=temp_coef_grads
		self.gap_grads = gap_grads
		self.match_grads = match_grads
		self.coef_grads = coef_grads
		return k


	def Kdiag(self, X):
		# Calc just the diagonal elements of a kernel matrix
		# need to update the TF stored hyper-parameters
		X_split = splitter(X,self.num_splits)
		self.kernels[0].kernel._gap_decay = self.gap_decay[0]
		self.kernels[0].kernel._match_decay = self.match_decay[0]
		self.kernels[0].kernel._order_coefs = list(self.order_coefs.values)
		k_diag = self.kernels[0].kernel.Kdiag(X_split[0])
		for i in range(1,len(self.kernels)):
			self.kernels[i].kernel._gap_decay = self.gap_decay[0]
			self.kernels[i].kernel._match_decay = self.match_decay[0]
			self.kernels[i].kernel._order_coefs = list(self.order_coefs.values)
			k_diag += self.kernels[i].kernel.Kdiag(X_split[i])
		return k_diag


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




#helper function to split long stings into n equal parts
def splitter(X,n):
	# split each input into n equal strings
	split_string = X[0][0].split(" ")
	length = len(split_string)
	chunk_size = int(length/n)
	split_data = []
	for i in range(0,n):
		split_data.append(np.zeros((X.shape[0],1),dtype=object))
	for i in range(0,X.shape[0]):
		split_string = X[i][0].split(" ")
		for j in range(n-1):
			section = split_string[j*chunk_size:(j+1)*chunk_size]
			split_data[j][i]=" ".join(section)
		# add all remaininh in last block
		section = split_string[(n-1)*chunk_size:]
		split_data[n-1][i]=" ".join(section)
	return split_data



