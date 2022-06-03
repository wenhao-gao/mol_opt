import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy 
from torch.autograd import Variable
from gnn_layer import GraphConvolution
from chemutils import smiles2graph 
import numpy as np 
sigmoid = torch.nn.Sigmoid() 
device = 'cpu'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, n_out, num_layer):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features = nfeat, out_features = nhid)
        self.gcs = nn.ModuleList([GraphConvolution(in_features = nhid, out_features = nhid) for i in range(num_layer)])
        self.gc2 = GraphConvolution(in_features = nhid, out_features = n_out)
        # self.dropout = dropout
        from chemutils import vocabulary 
        self.vocabulary_size = len(vocabulary) 
        self.nfeat = nfeat 
        self.nhid = nhid 
        self.n_out = n_out 
        self.num_layer = num_layer 
        # self.embedding = nn.Embedding(self.vocabulary_size, nfeat)
        self.embedding = nn.Linear(self.vocabulary_size, nfeat)
        self.criteria = torch.nn.BCEWithLogitsLoss() 
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.99))
        self.device = device 
        self = self.to(device) 

    def switch_device(self, device):
        self.device = device 
        self = self.to(device)

    def forward(self, node_mat, adj, weight):
        '''
            N: # substructure  &  d: vocabulary size

        Input: 
            node_mat:  
                [N,d]     row sum is 1.
            adj:    
                [N,N]
            weight:
                [N]  

        Output:
            scalar   prediction before sigmoid           [-inf, inf]
        '''
        node_mat, adj, weight = node_mat.to(self.device), adj.to(self.device), weight.to(self.device)
        x = self.embedding(node_mat)
        x = F.relu(self.gc1(x,adj))
        # print(x.device, adj.device, next(self.gcs[0].parameters()).device)

        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        x = self.gc2(x, adj)
        logits = torch.sum(x * weight.view(-1,1)) / torch.sum(weight)
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

    def smiles2twodim(self, smiles):
        embed = self.smiles2embed(smiles)
          

    def smiles2pred(self, smiles):
        idx_lst, node_mat, substructure_lst, atomidx_2substridx, adj, leaf_extend_idx_pair = smiles2graph(smiles)
        idx_vec = torch.LongTensor(idx_lst).to(device)
        node_mat = torch.FloatTensor(node_mat).to(device)
        adj = torch.FloatTensor(adj).to(device)
        weight = torch.ones_like(idx_vec).to(device)
        logits = self.forward(node_mat, adj, weight)
        pred = torch.sigmoid(logits) 
        return pred.item() 


    # def update_molecule(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np):
    #     node_mask = torch.BoolTensor(node_mask_np).to(self.device)
    #     node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

    #     pred_lst = []
    #     # for i in tqdm(range(5000)): ### 5k 10k
    #     for i in range(5000): ### 5k 10k

    #         # node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
    #         # adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
    #         # print("node_indicator", node_indicator.shape, adjacency_weight.shape)
    #         node_indicator = Variable(torch.FloatTensor(node_indicator_np2)).to(self.device)
    #         adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2)).to(self.device)
    #         adjacency_weight.requires_grad = True 
    #         node_indicator.requires_grad = True
    #         opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

    #         normalized_node_mat = torch.softmax(node_indicator, 1)
    #         normalized_adjacency_weight = torch.sigmoid(adjacency_weight)
    #         node_weight = torch.sum(normalized_adjacency_weight, 1)
    #         node_weight = torch.clamp(node_weight, max=1) 
    #         node_weight[node_mask] = 1 
    #         pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)

    #         # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
    #         target_y = Variable(torch.Tensor([1.0])[0]).to(self.device)
    #         cost = self.criteria(pred_y, target_y)
    #         opt_mol.zero_grad()
    #         cost.backward()
    #         opt_mol.step()


    #         node_indicator_np2, adjacency_weight_np2 = node_indicator.cpu().detach().numpy(), adjacency_weight.cpu().detach().numpy()
    #         node_indicator_np2[node_mask_np,:] = node_indicator_np[node_mask_np,:]
    #         adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

    #         if i%500==0:
    #             pred_lst.append(pred_y.item())

    #     # print('prediction', pred_lst)
    #     # return node_indicator, adjacency_weight  ### torch.FloatTensor 
    #     return node_indicator_np2, adjacency_weight_np2  #### np.array 


    def update_molecule(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np):
        node_mask = torch.BoolTensor(node_mask_np).to(self.device)
        adjacency_mask = torch.BoolTensor(adjacency_mask_np).to(self.device)
        node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

        pred_lst = []

        node_indicator_fix = Variable(torch.FloatTensor(node_indicator_np2)).to(self.device)
        # node_indicator_fix.requires_grad = True
        adjacency_weight_fix = Variable(torch.FloatTensor(adjacency_weight_np2)).to(self.device)
        # adjacency_weight_fix.requires_grad = True 


        node_indicator = Variable(torch.FloatTensor(node_indicator_np2)).to(self.device)
        adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2)).to(self.device)
        adjacency_weight.requires_grad = True 
        node_indicator.requires_grad = True
        opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

        # for i in tqdm(range(5000)): ### 5k 10k
        for i in range(5000): ### 5k 10k

            # node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
            # adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
            # print("node_indicator", node_indicator.shape, adjacency_weight.shape)

            normalized_node_mat = torch.softmax(node_indicator, 1)
            normalized_adjacency_weight = torch.sigmoid(adjacency_weight)
            node_weight = torch.sum(normalized_adjacency_weight, 1)
            node_weight = torch.clamp(node_weight, max=1) 
            node_weight[node_mask] = 1 
            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)

            # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
            target_y = Variable(torch.Tensor([1.0])[0]).to(self.device)
            cost = self.criteria(pred_y, target_y)
            opt_mol.zero_grad()
            cost.backward()
            opt_mol.step()

            node_indicator.requires_grad = False
            node_indicator[node_mask,:] = node_indicator_fix[node_mask,:]
            node_indicator.requires_grad = True
            adjacency_weight.requires_grad = False
            adjacency_weight[adjacency_mask] = adjacency_weight_fix[adjacency_mask]
            adjacency_weight.requires_grad = True

            if i%500==0:
                pred_lst.append(pred_y.item())


        node_indicator_np2, adjacency_weight_np2 = node_indicator.cpu().detach().numpy(), adjacency_weight.cpu().detach().numpy()
        node_indicator_np2[node_mask_np,:] = node_indicator_np[node_mask_np,:]
        adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

        # print('prediction', pred_lst)
        # return node_indicator, adjacency_weight  ### torch.FloatTensor 
        return node_indicator_np2, adjacency_weight_np2  #### np.array 


    # def update_molecule_interpret(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np):
    #     node_mask = torch.BoolTensor(node_mask_np).to(self.device)
    #     node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

    #     pred_lst = []
    #     # for i in tqdm(range(5000)): ### 5k 10k
    #     for i in range(5000): ### 5k 10k

    #         # node_indicator = Variable(torch.FloatTensor(node_indicator_np2), requires_grad = True).to(self.device)
    #         # adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2), requires_grad = True).to(self.device)
    #         # print("node_indicator", node_indicator.shape, adjacency_weight.shape)
    #         node_indicator = Variable(torch.FloatTensor(node_indicator_np2)).to(self.device)
    #         adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2)).to(self.device)
    #         adjacency_weight.requires_grad = True 
    #         node_indicator.requires_grad = True
    #         opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

    #         normalized_node_mat = torch.softmax(node_indicator, 1)
    #         normalized_adjacency_weight = torch.sigmoid(adjacency_weight)
    #         node_weight = torch.sum(normalized_adjacency_weight, 1)
    #         node_weight = torch.clamp(node_weight, max=1) 
    #         node_weight[node_mask] = 1 
    #         pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)

    #         # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
    #         target_y = Variable(torch.Tensor([1.0])[0]).to(self.device)
    #         cost = self.criteria(pred_y, target_y)
    #         opt_mol.zero_grad()
    #         cost.backward()
    #         opt_mol.step()

    #         if i==0:
    #             node_indicator_grad = node_indicator.grad.detach().numpy()
    #             adjacency_weight_grad = adjacency_weight.grad.detach().numpy() 
    #         # print(node_indicator.grad.shape)
    #         # print(adjacency_weight.grad.shape)

    #         node_indicator_np2, adjacency_weight_np2 = node_indicator.cpu().detach().numpy(), adjacency_weight.cpu().detach().numpy()
    #         node_indicator_np2[node_mask_np,:] = node_indicator_np[node_mask_np,:]
    #         adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

    #         if i%500==0:
    #             pred_lst.append(pred_y.item())

    #     # print('prediction', pred_lst)
    #     # return node_indicator, adjacency_weight  ### torch.FloatTensor 
    #     return node_indicator_np2, adjacency_weight_np2, node_indicator_grad, adjacency_weight_grad  #### np.array 


    # def update_molecule_v2(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np, 
    #                              leaf_extend_idx_pair, leaf_nonleaf_lst):
    #     (is_nonleaf_np, is_leaf_np, is_extend_np) = node_mask_np
    #     is_nonleaf = torch.BoolTensor(is_nonleaf_np).to(self.device)
    #     is_leaf = torch.BoolTensor(is_leaf_np).to(self.device)
    #     is_extend = torch.BoolTensor(is_extend_np).to(self.device)
    #     node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

    #     pred_lst = []
    #     # for i in tqdm(range(5000)): ### 5k 10k
    #     for i in range(5000): ### 5k 10k

    #         node_indicator = Variable(torch.FloatTensor(node_indicator_np2)).to(self.device)
    #         adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2)).to(self.device)
    #         adjacency_weight.requires_grad = True 
    #         node_indicator.requires_grad = True
    #         # print("node_indicator", node_indicator.shape, adjacency_weight.shape)
    #         opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

    #         normalized_node_mat = torch.softmax(node_indicator, 1)
    #         normalized_adjacency_weight = torch.sigmoid(adjacency_weight)  ### [0,1]
    #         node_weight = torch.sum(normalized_adjacency_weight, 1)
    #         node_weight = torch.clamp(node_weight, max=1)
    #         ### support shrink 
    #         node_weight[is_nonleaf] = 1 
    #         node_weight[is_leaf] = torch.cat([normalized_adjacency_weight[x,y].unsqueeze(0) for x,y in leaf_nonleaf_lst])
    #         node_weight[is_extend] *= node_weight[is_leaf]

    #         pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)

    #         # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
    #         target_y = Variable(torch.Tensor([1.0])[0]).to(self.device)
    #         cost = self.criteria(pred_y, target_y)
    #         opt_mol.zero_grad()
    #         cost.backward()
    #         opt_mol.step()

    #         node_indicator_np2, adjacency_weight_np2 = node_indicator.cpu().detach().numpy(), adjacency_weight.cpu().detach().numpy()
    #         node_indicator_np2[is_nonleaf_np,:] = node_indicator_np[is_nonleaf_np,:]
    #         adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

    #         ###### early stop 
    #         pred_lst.append(pred_y.item())
    #         if i%20==0 and len(pred_lst) > 500 and np.mean(pred_lst[-100:]) < np.mean(pred_lst[-200:-100]): 
    #             print("... early stop when optimizing DST for a smiles ...")
    #             break 
    #     # return node_indicator, adjacency_weight  ### torch.FloatTensor 
    #     return node_indicator_np2, adjacency_weight_np2  #### np.array 


    def update_molecule_v2(self, node_mask_np, node_indicator_np, adjacency_mask_np, adjacency_weight_np, 
                                 leaf_extend_idx_pair, leaf_nonleaf_lst):
        (is_nonleaf_np, is_leaf_np, is_extend_np) = node_mask_np
        is_nonleaf = torch.BoolTensor(is_nonleaf_np).to(self.device)
        is_leaf = torch.BoolTensor(is_leaf_np).to(self.device)
        is_extend = torch.BoolTensor(is_extend_np).to(self.device)
        node_indicator_np2, adjacency_weight_np2 = deepcopy(node_indicator_np), deepcopy(adjacency_weight_np)

        pred_lst = []

        adjacency_mask = torch.BoolTensor(adjacency_mask_np).to(self.device)
        node_indicator_fix = Variable(torch.FloatTensor(node_indicator_np2)).to(self.device)
        adjacency_weight_fix = Variable(torch.FloatTensor(adjacency_weight_np2)).to(self.device)

        node_indicator = Variable(torch.FloatTensor(node_indicator_np2)).to(self.device)
        adjacency_weight = Variable(torch.FloatTensor(adjacency_weight_np2)).to(self.device)
        adjacency_weight.requires_grad = True 
        node_indicator.requires_grad = True
        opt_mol = torch.optim.Adam([node_indicator, adjacency_weight], lr=1e-3, betas=(0.9, 0.99))

        # for i in tqdm(range(5000)): ### 5k 10k
        for i in range(5000): ### 5k 10k

            normalized_node_mat = torch.softmax(node_indicator, 1)
            normalized_adjacency_weight = torch.sigmoid(adjacency_weight)  ### [0,1]
            node_weight = torch.sum(normalized_adjacency_weight, 1)
            node_weight = torch.clamp(node_weight, max=1)
            ### support shrink 
            node_weight[is_nonleaf] = 1 
            node_weight[is_leaf] = torch.cat([normalized_adjacency_weight[x,y].unsqueeze(0) for x,y in leaf_nonleaf_lst])
            node_weight[is_extend] *= node_weight[is_leaf]

            pred_y = self.forward(normalized_node_mat, normalized_adjacency_weight, node_weight)

            # target_y = Variable(torch.Tensor([max(sigmoid(pred_y.item()) + 0.05, 1.0)])[0], requires_grad=True)
            target_y = Variable(torch.Tensor([1.0])[0]).to(self.device)
            cost = self.criteria(pred_y, target_y)
            opt_mol.zero_grad()
            cost.backward()
            opt_mol.step()

            node_indicator.requires_grad = False
            node_indicator[is_nonleaf,:] = node_indicator_fix[is_nonleaf,:]
            node_indicator.requires_grad = True 
            adjacency_weight.requires_grad = False
            adjacency_weight[adjacency_mask] = adjacency_weight_fix[adjacency_mask]
            adjacency_weight.requires_grad = True 

            ###### early stop 
            pred_lst.append(pred_y.item())
            if i%20==0 and len(pred_lst) > 500 and np.mean(pred_lst[-100:]) < np.mean(pred_lst[-200:-100]): 
                print("... early stop when optimizing DST for a smiles ...")
                break 

        node_indicator_np2, adjacency_weight_np2 = node_indicator.cpu().detach().numpy(), adjacency_weight.cpu().detach().numpy()
        node_indicator_np2[is_nonleaf_np,:] = node_indicator_np[is_nonleaf_np,:]
        adjacency_weight_np2[adjacency_mask_np] = adjacency_weight_np[adjacency_mask_np]

        # return node_indicator, adjacency_weight  ### torch.FloatTensor 
        return node_indicator_np2, adjacency_weight_np2  #### np.array 







    def learn(self, node_mat, adj, weight, target):
        pred_y = self.forward(node_mat, adj, weight)
        pred_y = pred_y.view(-1)
        target = target.to(self.device)
        cost = self.criteria(pred_y, target)
        self.opt.zero_grad() 
        cost.backward() 
        self.opt.step() 
        return cost.cpu().data.numpy(), pred_y.cpu().data.numpy() 

    def valid(self, node_mat, adj, weight, target):
        pred_y = self.forward(node_mat, adj, weight)
        pred_y = pred_y.view(-1)
        target = target.to(self.device)
        cost = self.criteria(pred_y, target)
        return cost.cpu().data.numpy(), pred_y.cpu().data.numpy() 

    
class GCNSum(GCN): 
    def forward(self, node_mat, adj, weight):
        node_mat, adj, weight = node_mat.to(self.device), adj.to(self.device), weight.to(self.device)
        x = self.embedding(node_mat)
        x = F.relu(self.gc1(x,adj))
        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        x = self.gc2(x, adj)
        logits = torch.sum(x * weight.view(-1,1))
        return logits 
        ## without sigmoid 

class GCNRegress(GCN):
    def __init__(self, nfeat, nhid, n_out, num_layer):
        super(GCNRegress, self).__init__(nfeat, nhid, n_out, num_layer)
        self.criteria = torch.nn.MSELoss() 

    def forward(self, node_mat, adj, weight):
        node_mat, adj, weight = node_mat.to(self.device), adj.to(self.device), weight.to(self.device)
        x = self.embedding(node_mat)
        x = F.relu(self.gc1(x,adj))
        for gc in self.gcs:
            x = F.relu(gc(x,adj))
        x = self.gc2(x, adj)
        pred = torch.sum(x * weight.view(-1,1))
        return pred  
        ## without sigmoid     


    def smiles2pred(self, smiles):
        idx_lst, node_mat, substructure_lst, atomidx_2substridx, adj, leaf_extend_idx_pair = smiles2graph(smiles)
        idx_vec = torch.LongTensor(idx_lst).to(device)
        node_mat = torch.FloatTensor(node_mat).to(device)
        adj = torch.FloatTensor(adj).to(device)
        weight = torch.ones_like(idx_vec).to(device)
        pred = self.forward(node_mat, adj, weight)
        return pred.item() 

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, init):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid, init=init)
#         self.gc2 = GraphConvolution(nhid, nclass, init=init)
#         self.dropout = dropout

#     def bottleneck(self, path1, path2, path3, adj, in_x):
#         return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

#     def forward(self, x, adj):
#         x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return x 
#         # return F.log_softmax(x, dim=1)




# class GCN_drop_in(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, init):
#         super(GCN_drop_in, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid, init=init)
#         self.gc2 = GraphConvolution(nhid, nclass, init=init)
#         self.dropout = dropout

#     def bottleneck(self, path1, path2, path3, adj, in_x):
#         return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
#         x = self.gc2(x, adj)

#         return F.log_softmax(x, dim=1)

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         super(GAT, self).__init__()
#         self.dropout = dropout

#         self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)




if __name__ == "__main__":

    rawdata_file = "raw_data/zinc.tab"

    with open(rawdata_file) as fin:
        lines = fin.readlines()[1:]

    gnn = GCN(nfeat = 50, nhid = 100, n_out = 1, num_layer = 2)













