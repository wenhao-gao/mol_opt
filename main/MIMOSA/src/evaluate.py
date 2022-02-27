import os, sys 
import numpy as np 
from time import time
from tqdm import tqdm 
from matplotlib import pyplot as plt
import pickle 
from random import shuffle 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tdc import Oracle
torch.manual_seed(1)
np.random.seed(2)
from tdc import Evaluator

from chemutils import * 
## 2. data and oracle 
# qed = Oracle(name = 'qed')
# logp = Oracle(name = 'logp')
# jnk = Oracle(name = 'JNK3')
# gsk = Oracle(name = 'GSK3B')
# def foracle(smiles):
# 	return logp(smiles)

oracle_name = sys.argv[1]
# 'jnkgsk', 'qedsajnkgsk', 'qed', 'jnk', 'gsk'


diversity = Evaluator(name = 'Diversity')
novelty = Evaluator(name = 'Novelty')


file = "data/zinc_clean.txt"
with open(file, 'r') as fin:
	lines = fin.readlines() 
train_smiles_lst = [line.strip().split()[0] for line in lines][:1000] 


## 5. run 
if __name__ == "__main__":

	# result_file = "result/denovo_from_" + start_smiles_lst[0] + "_generation_" + str(generations) + "_population_" + str(population_size) + ".pkl"
	# result_pkl = "result/ablation_dmg_topo_dmg_substr.pkl"
	# pkl_file = "result/denovo_qedlogpjnkgsk_start_ncncccn.pkl"
	pkl_file = "result/"+oracle_name+".pkl"
	idx_2_smiles2f, trace_dict = pickle.load(open(pkl_file, 'rb'))
	# bestvalue, best_smiles = 0, ''
	topk = 100
	whole_smiles2f = dict()
	for idx, (smiles2f,current_set) in tqdm(idx_2_smiles2f.items()):
		whole_smiles2f.update(smiles2f)
		# for smiles,f in smiles2f.items():
		# 	if f > bestvalue:
		# 		bestvalue = f
		# 		print("best", f)
		# 		best_smiles = smiles 

	smiles_f_lst = [(smiles,f) for smiles,f in whole_smiles2f.items()]
	smiles_f_lst.sort(key=lambda x:x[1], reverse=True)
	best_smiles_lst = [smiles for smiles,f in smiles_f_lst[:topk]]
	best_f_lst = [f for smiles,f in smiles_f_lst[:topk]]
	avg, std = np.mean(best_f_lst), np.std(best_f_lst)
	print('average of top-'+str(topk), str(avg)[:5], str(std)[:5])
	#### evaluate novelty 
	t1 = time()
	nov = novelty(best_smiles_lst, train_smiles_lst)
	t2 = time()
	print("novelty", nov, "takes", str(int(t2-t1)), 'seconds')

	### evaluate diversity 
	t1 = time()
	div = diversity(best_smiles_lst)
	t2 = time()
	print("diversity", div, 'takes', str(int(t2-t1)), 'seconds')


	# ### evaluate mean of property 
	# for oracle_name in oracle_lst:
	# 	oracle = Oracle(name = oracle_name)
	# 	scores = oracle(best_smiles_lst)
	# 	avg = np.mean(scores)
	# 	std = np.std(scores)
	# 	print(oracle_name, str(avg)[:7], str(std)[:7])

	# for ii,smiles in enumerate(best_smiles_lst[:20]):
	# 	print(smiles, str(gsk(smiles)))
	# 	draw_smiles(smiles, "figure/best_"+oracle_name+"_"+str(ii)+'.png')





