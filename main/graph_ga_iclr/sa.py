import numpy as np 
from tdc import Oracle
raw_sa = Oracle(name = 'SA')
sa_mu = 2.230044
sa_sigma = 0.6526308

def sa(smiles):
	sa_score = raw_sa(smiles)
	mod_score = np.maximum(sa_score, sa_mu)
	return np.exp(-0.5 * np.power((mod_score - sa_mu) / sa_sigma, 2.)) 

