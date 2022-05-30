import os, torch
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from tqdm import tqdm 
from random import shuffle 
from module import GCN 
from chemutils import smiles2feature  
from utils import Molecule_Dataset 
device = 'cpu'

def train_gnn(data, gnn, epoch=5): 
	"""
		data is smiles list 
	"""
	shuffle(data)
	training_set = Molecule_Dataset(data)

	params = {'batch_size': 1,
	          'shuffle': True,
	          'num_workers': 1}
	def collate_fn(batch_lst):
		return batch_lst
	train_generator = torch.utils.data.DataLoader(training_set, collate_fn = collate_fn, **params)

	cost_lst = []

	for ep in tqdm(range(epoch)):
		for i, smiles in tqdm(enumerate(train_generator)):
			### 1. training
			smiles = smiles[0][0]
			try:
				graph_feature = smiles2feature(smiles)  ### smiles2feature: **randomly** mask one leaf node  
				if graph_feature is None:
					# print('graph feature is none')
					continue 
				node_mat, adjacency_matrix, idx, label = graph_feature 
			except:
				# print('some problems happening')
				continue 
			# idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)
			node_mat = torch.FloatTensor(node_mat).to(device)
			adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
			label = torch.LongTensor([label]).view(-1).to(device)
			cost = gnn.learn(node_mat, adjacency_matrix, idx, label)
			cost_lst.append(cost)

	return gnn 


