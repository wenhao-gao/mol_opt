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
from module import GCN 
from chemutils import smiles2graph, vocabulary, smiles2feature  
from utils import Molecule_Dataset 


device = 'cpu'
data_file = "data/zinc_clean.txt"
with open(data_file, 'r') as fin:
	lines = fin.readlines()

shuffle(lines)
lines = [line.strip() for line in lines]
N = int(len(lines) * 0.9)
train_data = lines[:N]
valid_data = lines[N:]



training_set = Molecule_Dataset(train_data)
valid_set = Molecule_Dataset(valid_data)
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1}
# exit() 


def collate_fn(batch_lst):
	return batch_lst

train_generator = torch.utils.data.DataLoader(training_set, collate_fn = collate_fn, **params)
valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn = collate_fn, **params)

gnn = GCN(nfeat = 50, nhid = 100, num_layer = 3).to(device)
print('GNN is built!')
# exit() 

cost_lst = []
valid_loss_lst = []
epoch = 5 
every_k_iters = 5000
save_folder = "save_model/GNN_epoch_" 
for ep in tqdm(range(epoch)):
	for i, smiles in tqdm(enumerate(train_generator)):
		### 1. training
		smiles = smiles[0]
		node_mat, adjacency_matrix, idx, label = smiles2feature(smiles)  ### smiles2feature: only mask leaf node    
		# idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)
		node_mat = torch.FloatTensor(node_mat).to(device)
		adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
		label = torch.LongTensor([label]).view(-1).to(device)
		# print('label', label)
		cost = gnn.learn(node_mat, adjacency_matrix, idx, label)
		cost_lst.append(cost)

		#### 2. validation 
		if i % every_k_iters == 0:
			gnn.eval()
			valid_loss, valid_num = 0,0 
			for smiles in valid_generator:
				smiles = smiles[0]
				node_mat, adjacency_matrix, idx, label = smiles2feature(smiles)  
				node_mat = torch.FloatTensor(node_mat).to(device)
				adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
				label = torch.LongTensor([label]).view(-1).to(device)
				cost, _ = gnn.infer(node_mat, adjacency_matrix, idx, label)
				valid_loss += cost
				valid_num += 1 
			valid_loss = valid_loss / valid_num
			valid_loss_lst.append(valid_loss)
			file_name = save_folder + str(ep) + "_validloss_" + str(valid_loss)[:7] + ".ckpt"
			torch.save(gnn, file_name)
			gnn.train()




