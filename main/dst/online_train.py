import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm 
from matplotlib import pyplot as plt
import pickle 
from random import shuffle 
torch.manual_seed(4) 
np.random.seed(2)
import os
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
from random import random 

from module import GCN 
from chemutils import smiles2graph, vocabulary, is_valid
# def is_valid(smiles):  return True/False

from utils import Molecule_Dataset 

def collate_fn(batch_lst):
	return [element[0] for element in batch_lst], [element[1] for element in batch_lst]


def train_gnn(data, gnn, device = torch.device('cpu')):
	"""
		data: List [(smiles, value), (smiles, value), ....]
		gnn: torch.nn.Module 
	"""
	# device = 'cpu'
	shuffle(data)
	train_data = data[:int(len(data)*0.8)]
	valid_data = data[int(len(data)*0.8):]
	training_set = Molecule_Dataset(train_data)
	valid_set = Molecule_Dataset(valid_data)
	params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1}
	train_generator = torch.utils.data.DataLoader(training_set, collate_fn = collate_fn, **params)
	valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn = collate_fn, **params)

	valid_loss_lst = []
	epoch = 5 
	every_k_iters = 5000
	save_folder = "save_model/"  
	for ep in tqdm(range(epoch)):
		for i, (smiles, score) in tqdm(enumerate(train_generator)):
			### 1. training
			gnn.train()
			smiles = smiles[0]
			y = torch.FloatTensor(score)
			graph_feature = smiles2graph(smiles)
			if graph_feature is None:
				continue 
			idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = graph_feature
			idx_vec = torch.LongTensor(idx_lst).to(device)
			node_mat = torch.FloatTensor(node_mat).to(device)
			adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
			weight = torch.ones_like(idx_vec).to(device)
			
			cost = gnn.learn(node_mat, adjacency_matrix, weight, y)

			#### 2. validation 
			if i % every_k_iters == 0:
				gnn.eval()
				valid_loss, valid_num = 0,0 
				for smiles,score in valid_generator:
					smiles = smiles[0]
					y = torch.FloatTensor(score).to(device)
					graph_feature = smiles2graph(smiles)
					if graph_feature is None:
						continue 
					idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = graph_feature 
					idx_vec = torch.LongTensor(idx_lst).to(device)
					node_mat = torch.FloatTensor(node_mat).to(device)
					adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
					weight = torch.ones_like(idx_vec).to(device)
					cost, _ = gnn.valid(node_mat, adjacency_matrix, weight, y)
					valid_loss += cost
					valid_num += 1 
				valid_loss = valid_loss / valid_num
				if valid_loss_lst != [] and valid_loss > valid_loss_lst[-1]:
					gnn.train()
					return gnn 
				valid_loss_lst.append(valid_loss)
				# file_name = save_folder + str(ep) + "_iter_" + str(i) + "_validloss_" + str(valid_loss)[:7] + ".ckpt"
				# torch.save(gnn, file_name)
				gnn.train()
	gnn.train() 
	return gnn 


"""

oracle = sys.argv[1]
oracle_num = int(sys.argv[2])
oracle2labelidx = {'jnkgsk': [3,4], 'qedsajnkgsk':[1,2,3,4], 'qed':[1], 'jnk':[3], 'gsk':[4]}
labelidx = oracle2labelidx[oracle]
device = 'cpu'
data_file = "data/zinc_label.txt"
with open(data_file, 'r') as fin:
	lines = fin.readlines() 

shuffle(lines)
lines = lines[:oracle_num]
lines = [(line.split()[0], np.mean([float(line.split()[i]) for i in labelidx])) for line in lines]
N = int(len(lines) * 0.9)
train_data = lines[:N]
valid_data = lines[N:]


training_set = Molecule_Dataset(train_data)
valid_set = Molecule_Dataset(valid_data)



train_generator = torch.utils.data.DataLoader(training_set, collate_fn = collate_fn, **params)
valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn = collate_fn, **params)
print('data loader is built!')

gnn = GCN(nfeat = 50, nhid = 100, n_out = 1, num_layer = 3).to(device)
print('GNN is built!')


cost_lst = []
valid_loss_lst = []
epoch = 5 
every_k_iters = 5000
save_folder = "save_model/" + oracle + "_epoch_" 
for ep in tqdm(range(epoch)):
	for i, (smiles, score) in tqdm(enumerate(train_generator)):
		### 1. training
		smiles = smiles[0]
		y = torch.FloatTensor(score)
		idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)
		idx_vec = torch.LongTensor(idx_lst).to(device)
		node_mat = torch.FloatTensor(node_mat).to(device)
		adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
		weight = torch.ones_like(idx_vec).to(device)
		
		cost = gnn.learn(node_mat, adjacency_matrix, weight, y)
		cost_lst.append(cost)

		#### 2. validation 
		if i % every_k_iters == 0:
			gnn.eval()
			valid_loss, valid_num = 0,0 
			for smiles,score in valid_generator:
				smiles = smiles[0]
				y = torch.FloatTensor(score).to(device)
				idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)
				idx_vec = torch.LongTensor(idx_lst).to(device)
				node_mat = torch.FloatTensor(node_mat).to(device)
				adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
				weight = torch.ones_like(idx_vec).to(device)
				cost, _ = gnn.valid(node_mat, adjacency_matrix, weight, y)
				valid_loss += cost
				valid_num += 1 
			valid_loss = valid_loss / valid_num
			valid_loss_lst.append(valid_loss)
			file_name = save_folder + str(ep) + "_iter_" + str(i) + "_validloss_" + str(valid_loss)[:7] + ".ckpt"
			torch.save(gnn, file_name)
			gnn.train()

"""






