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

def train_gnn(data, gnn): 
	"""
		data is smiles list 
	"""
	shuffle(data)
	N = int(len(data) * 0.9)
	train_data = data[:N]
	valid_data = data[N:]

	training_set = Molecule_Dataset(train_data)
	valid_set = Molecule_Dataset(valid_data)
	params = {'batch_size': 1,
	          'shuffle': True,
	          'num_workers': 1}
	def collate_fn(batch_lst):
		return batch_lst
	train_generator = torch.utils.data.DataLoader(training_set, collate_fn = collate_fn, **params)
	valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn = collate_fn, **params)

	cost_lst = []
	valid_loss_lst = []
	epoch = 5 
	every_k_iters = 5000
	# save_folder = "save_model/GNN_epoch_" 
	for ep in tqdm(range(epoch)):
		for i, smiles in tqdm(enumerate(train_generator)):
			### 1. training
			smiles = smiles[0]
			try:
				graph_feature = smiles2feature(smiles)  ### smiles2feature: **randomly** mask one leaf node  
				if graph_feature is None:
					continue 
				node_mat, adjacency_matrix, idx, label = graph_feature   
			except:
				continue 
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
					try:
						graph_feature = smiles2feature(smiles)  
						if graph_feature is None:
							continue 
						node_mat, adjacency_matrix, idx, label = graph_feature 
					except:
						continue 
					node_mat = torch.FloatTensor(node_mat).to(device)
					adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
					label = torch.LongTensor([label]).view(-1).to(device)
					cost, _ = gnn.infer(node_mat, adjacency_matrix, idx, label)
					valid_loss += cost
					valid_num += 1 
				valid_loss = valid_loss / valid_num
				valid_loss_lst.append(valid_loss)
				# file_name = save_folder + str(ep) + "_validloss_" + str(valid_loss)[:7] + ".ckpt"
				# torch.save(gnn, file_name)
				gnn.train()
	return gnn 


