from tqdm import tqdm 
import numpy as np 
from tdc import Oracle 
from random import shuffle 
data_file = "data/zinc_clean.txt"
labelled_file = "data/zinc_label.txt"
data_size = 10000 
"""
6 columns 
`smiles`, `qed`, `sa`, `jnk`, `gsk`, `logp`

"""

with open(data_file) as fin:
	lines = fin.readlines()[1:]
	smiles_lst = [line.strip().strip('"') for line in lines]

oracle_names = ['qed', 'JNK3', 'GSK3B', 'LogP']
oracle_list = [Oracle(name) for name in oracle_names]

qed = Oracle('qed')
jnk = Oracle('JNK3')
gsk = Oracle('GSK3B')
logp = Oracle('LogP')

sa = Oracle(name = 'SA')
mu = 2.230044
sigma = 0.6526308
def sa_oracle(smiles):
	sa_score = sa(smiles)
	mod_score = np.maximum(sa_score, mu)
	return np.exp(-0.5 * np.power((mod_score - mu) / sigma, 2.)) 
def sa_oracle_lst(smiles_lst):
	return [sa_oracle(smiles) for smiles in smiles_lst]

with open(data_file, 'r') as fin:
	lines = fin.readlines()
smiles_lst = [line.strip() for line in lines]
shuffle(smiles_lst)
smiles_lst = smiles_lst[:data_size]
batch_size = 10
num_of_batch = int(np.ceil(len(smiles_lst) / batch_size))

with open(labelled_file, 'w') as fout:
	for i in tqdm(range(num_of_batch)):
		start_idx = i*batch_size 
		end_idx = i*batch_size + batch_size 
		sub_smiles_lst = smiles_lst[start_idx:end_idx]
		qed_scores = qed(sub_smiles_lst)
		sa_scores = sa_oracle_lst(sub_smiles_lst)
		jnk_scores = jnk(sub_smiles_lst)
		gsk_scores = gsk(sub_smiles_lst)
		logp_scores = logp(sub_smiles_lst) 
		for smiles,s1,s2,s3,s4,s5 in zip(sub_smiles_lst, qed_scores, sa_scores, jnk_scores, gsk_scores, logp_scores):
			fout.write(smiles + '\t' + str(s1) + '\t' + str(s2) + '\t' + str(s3) + '\t' + str(s4) + '\t' + str(s5) + '\n')




'''

ZINC 250K 

  - QED  6 min 

  - LogP <1.5hours 

  - JNK3 10 hours 
	- 0.15 second/mol

  - GSK 10 hours   
	- 0.15 second/mol

'''

