import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy 
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import SequentialSampler
import matplotlib.pyplot as plt
import numpy as np 
sigmoid = torch.nn.Sigmoid() 
from tqdm import tqdm 

from gnn_layer import GraphConvolution, GraphAttention
from chemutils import smiles2graph, vocabulary 

torch.manual_seed(4) 
np.random.seed(1)

# def sigmoid(x):
#     return 1/(1+np.exp(-x))
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, num_layer):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features = nfeat, out_features = nhid)
        self.gcs = [GraphConvolution(in_features = nhid, out_features = nhid) for i in range(num_layer)]
        # self.dropout = dropout
        from chemutils import vocabulary 
        self.vocabulary_size = len(vocabulary) 
        self.out_fc = nn.Linear(nhid, self.vocabulary_size)
        self.nfeat = nfeat 
        self.nhid = nhid 
        self.num_layer = num_layer 
        # self.embedding = nn.Embedding(self.vocabulary_size, nfeat)
        self.embedding = nn.Linear(self.vocabulary_size + 1, nfeat)
        self.criteria = torch.nn.CrossEntropyLoss() 
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.99))
        self.device = device 
        self = self.to(device) 

    def switch_device(self, device):
        self.device = device 
        self = self.to(device)

    def forward(self, node_mat, adj, idx):
        ''' N: # substructure  &  d: vocabulary size
        Input: 
            node_mat:  [N,d]     row sum is 1.
            adj:       [N,N]    
            idx:     integer 

        Output:
            scalar   prediction before sigmoid           [-inf, inf]
        '''
        node_mat, adj = node_mat.to(self.device), adj.to(self.device)
        x = self.embedding(node_mat)
        x = F.relu(self.gc1(x,adj))
        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        x = x[idx].view(1,-1)
        logits = self.out_fc(x)
        return logits 
        ## without sigmoid 

    def smiles2embed(self, smiles):
        idx_lst, node_mat, substructure_lst, atomidx_2substridx, adj, leaf_extend_idx_pair = smiles2graph(smiles)
        idx_vec = torch.LongTensor(idx_lst).to(device)
        node_mat = torch.FloatTensor(node_mat).to(device)
        adj = torch.FloatTensor(adj).to(device)
        weight = torch.ones_like(idx_vec).to(device)
        
        ### forward 
        node_mat, adj, weight = node_mat.to(self.device), adj.to(self.device), weight.to(self.device)
        x = self.embedding(node_mat) ## bug 
        x = F.relu(self.gc1(x,adj))
        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        return torch.mean(x, 0)


    def smiles2pred(self, smiles):
        idx_lst, node_mat, substructure_lst, atomidx_2substridx, adj, leaf_extend_idx_pair = smiles2graph(smiles)
        idx_vec = torch.LongTensor(idx_lst).to(device)
        node_mat = torch.FloatTensor(node_mat).to(device)
        adj = torch.FloatTensor(adj).to(device)
        weight = torch.ones_like(idx_vec).to(device)
        logits = self.forward(node_mat, adj, weight)
        pred = torch.sigmoid(logits) 
        return pred.item() 

    def learn(self, node_mat, adj, idx, label):
        pred_y = self.forward(node_mat, adj, idx)
        pred_y = pred_y.view(1,-1)
        # print(pred_y, pred_y.shape, label, label.shape) 
        cost = self.criteria(pred_y, label) 
        self.opt.zero_grad() 
        cost.backward() 
        self.opt.step() 
        return cost.data.numpy(), pred_y.data.numpy() 

    def infer(self, node_mat, adj, idx, target):
        pred_y = self.forward(node_mat, adj, idx)
        pred_y = pred_y.view(1,-1)
        cost = self.criteria(pred_y, target)
        return cost.data.numpy(), pred_y.data.numpy() 


if __name__ == "__main__":
    gnn = GCN(nfeat = 50, nhid = 100, num_layer = 2)













