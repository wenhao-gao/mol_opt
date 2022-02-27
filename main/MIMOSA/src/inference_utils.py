
### 1. import
import numpy as np 
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
import random 
random.seed(1)
from chemutils import * 
'''
optimize_single_molecule_one_iterate
gnn_prediction_of_single_smiles
oracle_screening
gnn_screening
optimize_single_molecule_all_generations
similarity_matrix(smiles_lst)
'''
from dpp import DPPModel



def gnn_prediction_of_single_smiles(smiles, gnn):
	if not is_valid(smiles):
		return 0
	return gnn.smiles2pred(smiles)
	# idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)
	# idx_vec = torch.LongTensor(idx_lst)
	# node_mat = torch.FloatTensor(node_mat)
	# adjacency_matrix = torch.FloatTensor(adjacency_matrix)
	# weight = torch.ones_like(idx_vec)
	# logits = gnn(node_mat = node_mat, adj = adjacency_matrix, weight = weight)
	# logits = logits.item() 
	# print("gnn prediction", logits)
	# return logits 


def oracle_screening(smiles_set, oracle):
	smiles_score_lst = []
	for smiles in smiles_set:
		score = oracle(smiles)
		smiles_score_lst.append((smiles, score))
	smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
	return smiles_score_lst 

def dpp(smiles_score_lst, num_return, lamb):
	smiles_lst = [i[0] for i in smiles_score_lst]
	if len(smiles_lst) <= num_return:
		return smiles_lst, None, None 
	score_arr = np.array([i[1] for i in smiles_score_lst])
	sim_mat = similarity_matrix(smiles_lst)
	dpp_model = DPPModel(smiles_lst = smiles_lst, sim_matrix = sim_mat, f_scores = score_arr, top_k = num_return, lamb = lamb)
	smiles_lst, log_det_V, log_det_S = dpp_model.dpp()
	return smiles_lst, log_det_V, log_det_S 


def gnn_screening(smiles_set, gnn):
	smiles_score_lst = []
	for smiles in smiles_set:
		score = gnn_prediction_of_single_smiles(smiles, gnn)
		smiles_score_lst.append((smiles, score))
	smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
	return smiles_score_lst
	# smiles_lst = [i[0] for i in smiles_score_lst]
	# return smiles_lst

def optimize_single_node(smiles):
	assert substr_num(smiles)==1 
	vocabulary = load_vocabulary()
	atoms = ['N', 'C']

# bondtype_list = [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE] ### chemutils 

def optimize_single_molecule_one_iterate(smiles, gnn):
	target_ = torch.LongTensor([0]).view(-1)
	if smiles == None:
		return set() 
	if not is_valid(smiles):
		return set()
	origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(smiles))
	new_smiles_set = set() 
	jj=-100

	origin_idx_lst, origin_node_mat, origin_substructure_lst, \
	origin_atomidx_2substridx, origin_adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)

	feature_lst = smiles2expandfeature(smiles)
	for node_mat, adj_mat, mask_idx in feature_lst:
		node_mat = torch.FloatTensor(node_mat) 
		adj_mat = torch.FloatTensor(adj_mat)
		N = adj_mat.shape[0]
		for jj in range(N):
			if adj_mat[jj,N-1]==1:
				break 

		_, prediction = gnn.infer(node_mat, adj_mat, mask_idx, target_)
		top_idxs = prediction.reshape(-1).argsort().tolist()[::-1][:3]
		top_words = [vocabulary[ii] for ii in top_idxs]
		for substru_idx, word in zip(top_idxs, top_words):
			leaf_atom_idx_lst = origin_substructure_lst[jj]

			if type(leaf_atom_idx_lst)==int:  ### int: single atom;   else: list of integer
				leaf_atom_idx_lst = [leaf_atom_idx_lst]
			for leaf_atom_idx in leaf_atom_idx_lst:
				for new_bond in bondtype_list:
					if ith_substructure_is_atom(substru_idx):
						new_smiles = add_atom_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                          new_atom = word, new_bond = new_bond)
						new_smiles_set.add(new_smiles)
					else:
						new_smiles_batch = add_fragment_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                                    fragment = word , new_bond = new_bond)
						new_smiles_set = new_smiles_set.union(new_smiles_batch)

	new_smiles_set = set([new_smiles for new_smiles in new_smiles_set if new_smiles != None])
	return new_smiles_set




def optimize_single_molecule_all_generations(input_smiles, gnn, oracle, generations, population_size, lamb):
	smiles2f = dict() 
	traceback_dict = dict() 
	input_smiles = canonical(input_smiles)
	input_score = oracle(input_smiles)
	best_mol_score_list = []
	existing_set = set([input_smiles])
	current_mol_score_list = [(input_smiles, input_score)]
	for it in tqdm(range(generations)):
		new_smiles_set = set()
		#### optimize each single smiles
		for smiles,score in current_mol_score_list:
			# proposal_smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)
			proposal_smiles_set = optimize_single_molecule_one_iterate_v2(smiles, gnn)
			proposal_smiles_set = proposal_smiles_set.difference(set([input_smiles]))
			for new_smiles in proposal_smiles_set:
				if new_smiles not in traceback_dict:
					traceback_dict[new_smiles] = smiles 
			new_smiles_set = new_smiles_set.union(proposal_smiles_set)

		### remove the repetition
		# new_smiles_set = new_smiles_set.difference(existing_set)

		### add smiles into existing_set 
		existing_set = existing_set.union(new_smiles_set)

		### scoring new smiles 
		####### I:GNN & oracle scoring
		# gnn_smiles_lst = gnn_screening(new_smiles_set, gnn)
		# gnn_smiles_lst = gnn_smiles_lst[:population_size*3]
		# mol_score_list = oracle_screening(gnn_smiles_lst, oracle)
		############ oracle call <= generations * population_size * 3 + 1 

		####### II: only oracle scoring
		mol_score_list = oracle_screening(new_smiles_set, oracle)
		############ oracle call: unbounded, with better performance 
		for smiles, score in mol_score_list:
			if score > 0.50:
				print('example', smiles, score)


		### save results 
		best_mol_score_list.extend(mol_score_list)


		### only keep top-k 
		# mol_score_list = mol_score_list[:population_size] 
		### dpp(smiles_score_lst, num_return, lamb)
		smiles_lst = dpp(mol_score_list, num_return = population_size, lamb = lamb)


		### for next generation
		# current_mol_score_list = mol_score_list
		current_mol_score_list = [(smiles,0.0) for smiles in smiles_lst]

	### endfor 

	best_mol_score_list.sort(key=lambda x:x[1], reverse=True) 
	return best_mol_score_list, input_score, traceback_dict 



def calculate_results(input_smiles, input_score, best_mol_score_list):
	if best_mol_score_list == []:
		with open(result_file, 'a') as fout:
			fout.write("fail to optimize" + input_smiles + '\n')
		return None 
	output_scores = [i[1] for i in best_mol_score_list]
	smiles_lst = [i[0] for i in best_mol_score_list]
	with open(result_file, 'a') as fout:
		fout.write(str(input_score) + '\t' + str(output_scores[0]) + '\t' + str(np.mean(output_scores[:3]))
				 + '\t' + input_smiles + '\t' + ' '.join(smiles_lst[:3]) + '\n')
	return input_score, output_scores[0]

def inference_single_molecule(input_smiles, gnn, result_file, generations, population_size):
	best_mol_score_list, input_score, traceback_dict = optimize_single_molecule_all_generations(input_smiles, gnn, oracle, generations, population_size)
	return calculate_results(input_smiles, input_score, result_file, best_mol_score_list, oracle)




def inference_molecule_set(input_smiles_lst, gnn, result_file, generations, population_size):
	score_lst = []
	for input_smiles in tqdm(input_smiles_lst):
		if not is_valid(input_smiles):
			continue 
		result = inference_single_molecule(input_smiles, gnn, result_file, generations, population_size)
		if result is None:
			score_lst.append(None)
		else:
			input_score, output_score = result
			score_lst.append((input_score, output_score))
	return score_lst








