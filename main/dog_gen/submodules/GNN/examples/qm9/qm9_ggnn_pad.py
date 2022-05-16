
import numpy as np
import torch
from torch.utils import data
from torch import nn

from graph_neural_networks.datasets import qm9
from graph_neural_networks.pad_pattern import ggnn_pad
from graph_neural_networks.ggnn_general import ggnn_base
from graph_neural_networks.core import mlp
from graph_neural_networks.example_trainers import qm9_regression


class GGNNModel(nn.Module):
    def __init__(self, hidden_layer_size, edge_names, T):
        super().__init__()
        self.ggnn = ggnn_pad.GGNNPad(
            ggnn_base.GGNNParams(hidden_layer_size, edge_names, T))

        mlp_project_up = mlp.get_mlp(mlp.MlpParams(hidden_layer_size, 1, []))
        mlp_gate = mlp.get_mlp(mlp.MlpParams(hidden_layer_size, 1, []))
        mlp_down = lambda x: x

        self.ggnn_top = ggnn_pad.GraphFeatureTopOnly(mlp_project_up, mlp_gate, mlp_down)

    def forward(self, feats, adj_mat, node_mask):
        node_feats = self.ggnn(feats, adj_mat, node_mask)
        graph_feats = self.ggnn_top(node_feats)
        return graph_feats


class DatasetTransform(object):
    def __init__(self, hidden_layer_size):
        self.e_to_am = qm9.EdgeListToAdjMatUndirected()
        self.nf_em = qm9.NodeFeaturesEmbedder(hidden_layer_size)

    def __call__(self, edge, node_features):
        return (self.e_to_am(edge), self.nf_em(node_features))


def collate_fn(batch):
    adjacancy_matrices = [elem[0][0] for elem in batch]
    node_features = [elem[0][1] for elem in batch]
    targets = [elem[1] for elem in batch]

    max_hw = max([am.shape[1] for am in adjacancy_matrices])

    node_mask = []
    for adj_ in adjacancy_matrices:
        mask_ = np.zeros(max_hw, dtype=np.bool)
        mask_[:adj_.shape[0]] = True
        node_mask.append(mask_)

    padded_adj_mats = [
        np.pad(arr, [(0, max_hw - arr.shape[0]), (0, max_hw - arr.shape[1]), (0, 0)], mode='constant', constant_values=0)
        for arr in adjacancy_matrices]

    padded_node_feats = [np.pad(arr, [(0, max_hw - arr.shape[0]), (0, 0)], mode='constant', constant_values=0)
                         for arr in node_features]

    padded_args = tuple(zip(padded_node_feats, padded_adj_mats, node_mask, targets))

    batch_collated = data.dataloader.default_collate(padded_args)
    return batch_collated


class PadParts(qm9_regression.ExperimentParts):
    def create_model(self):
        return GGNNModel(self.exp_params.hidden_layer_size, self.exp_params.edge_names,
                          self.exp_params.T)

    def create_transform(self):
        return DatasetTransform(self.exp_params.hidden_layer_size)

    def create_collate_function(self):
        return collate_fn

    def data_split_and_cudify_func(self, data):
        node_feats, adj_mat, node_mask, targets = data
        if torch.cuda.is_available():
            node_feats, adj_mat, node_mask, targets = [elem.cuda() for elem in
                                              (node_feats, adj_mat, node_mask, targets)]
        return (node_feats, adj_mat, node_mask), targets


def main():
    exp_params = qm9_regression.ExperimentParams()
    exp_parts = PadParts(exp_params)
    qm9_regression.main_runner(exp_parts)


if __name__ == '__main__':
    print("Starting...")
    main()
    print('Completed!')
