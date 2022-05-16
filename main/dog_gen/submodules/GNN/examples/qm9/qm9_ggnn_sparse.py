
import numpy as np
import torch
from torch import nn

import graph_neural_networks.sparse_pattern.ggnn_sparse
from graph_neural_networks.datasets import qm9
from graph_neural_networks.core import utils
from graph_neural_networks.core import data_types
from graph_neural_networks.sparse_pattern import ggnn_sparse
from graph_neural_networks.sparse_pattern import graph_as_adj_list
from graph_neural_networks.ggnn_general import ggnn_base
from graph_neural_networks.ggnn_general import graph_tops
from graph_neural_networks.core import mlp
from graph_neural_networks.example_trainers import qm9_regression


class GGNNModel(nn.Module):
    def __init__(self, hidden_layer_size, edge_names, T):
        super().__init__()
        self.ggnn = ggnn_sparse.GGNNSparse(
            ggnn_base.GGNNParams(hidden_layer_size, edge_names, T))

        mlp_project_up = mlp.get_mlp(mlp.MlpParams(hidden_layer_size, 1, []))
        mlp_gate = mlp.get_mlp(mlp.MlpParams(hidden_layer_size, 1, []))
        mlp_down = lambda x: x

        self.ggnn_top = graph_neural_networks.sparse_pattern.ggnn_sparse.GraphFeaturesStackIndexAdd(mlp_project_up, mlp_gate, mlp_down)

    def forward(self, g_adjlist: graph_as_adj_list.DirectedGraphAsAdjList):
        g_adjlist: graph_as_adj_list.DirectedGraphAsAdjList = self.ggnn(g_adjlist)
        graph_feats = self.ggnn_top(g_adjlist.node_features, g_adjlist.node_to_graph_id)
        return graph_feats


class DatasetTransform(object):
    def __init__(self, hidden_layer_size, edge_types_as_ints):
        self.e_to_adjlistdict = qm9.EdgeListToAdjListUndirected(edge_types_as_ints)
        self.nf_em = qm9.NodeFeaturesEmbedder(hidden_layer_size)

    def __call__(self, edge, node_features):
        edge_type_as_int_to_adjacency_list_map = self.e_to_adjlistdict(edge)
        edge_type_to_adjacency_list_map = {f"edge_{k}": np.array(v).T for k, v in
                                           edge_type_as_int_to_adjacency_list_map.items()}

        node_features = self.nf_em(node_features)

        g_adjlist = graph_as_adj_list.DirectedGraphAsAdjList(node_features, edge_type_to_adjacency_list_map,
                                                             np.zeros(node_features.shape[0], dtype=data_types.NP_LONG))
        return g_adjlist


def collate_function(batch):
    #todo: will not be able to pin memory at the moment.

    graphs_as_adjlist = [elem[0] for elem in batch]
    targets = [elem[1] for elem in batch]

    graphs_as_adjlist_catted = graphs_as_adjlist[0].concatenate(graphs_as_adjlist)

    graphs_as_adjlist_catted.inplace_from_np_to_torch()
    targets = torch.from_numpy(np.array(targets))

    return graphs_as_adjlist_catted, targets


class SparseParts(qm9_regression.ExperimentParts):
    def create_model(self):
        return GGNNModel(self.exp_params.hidden_layer_size, self.exp_params.edge_names, self.exp_params.T)

    def create_transform(self):
        return DatasetTransform(self.exp_params.hidden_layer_size, self.exp_params.edge_names_as_ints)

    def create_collate_function(self):
        return collate_function

    def data_split_and_cudify_func(self, data):
        graphs, targets = data
        if torch.cuda.is_available():
            graphs.inplace_torch_to('cuda')
            targets = targets.cuda()
        return (graphs,), targets


def main():
    exp_params = qm9_regression.ExperimentParams()
    exp_parts = SparseParts(exp_params)
    qm9_regression.main_runner(exp_parts)


if __name__ == '__main__':
    print("Starting...")
    main()
    print('Completed!')
