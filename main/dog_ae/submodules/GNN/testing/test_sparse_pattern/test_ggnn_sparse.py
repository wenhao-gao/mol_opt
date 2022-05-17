
import numpy as np
import torch
from torch.nn import functional as F

from graph_neural_networks.ggnn_general import graph_tops
from graph_neural_networks.core import data_types
from graph_neural_networks.core import mlp
from graph_neural_networks.sparse_pattern import ggnn_sparse


class GraphFeaturesStackCSRefImplementation(graph_tops.GraphFeaturesFromStackedNodeFeaturesBase):
    """
    Do the sum via a cumsum and then indexing and then doing difference.

    THE NODE FEATURES MUST COME IN ORDER (IE ALL NODES OF A GRAPH ARE GROUPED TOGETHER)
    """
    def forward(self, node_features,  node_grp_start_with_end):
        """
        :param node_features: [v*, h]
        :param node_grp_start_with_end: [g +1] index into node features for where each new graph starts
         with concatenated on the end the index of the next imaginary graph ie  node_features.shape[0]
         so that when you take off one you index the last node
        """
        proj_up = self.mlp_project_up(node_features)  # [v*, j]
        gate_logit = self.mlp_gate(node_features)  # [v*, 1]
        gate = torch.sigmoid(gate_logit)  # [v*, j]

        weighted_sum = torch.cumsum(gate * proj_up, dim=0)   # [v*, j]
        weighted_sum = F.pad(weighted_sum, (0,0,1,0), mode='constant', value=0)

        indx_before = node_grp_start_with_end[:-1]
        indx_after = node_grp_start_with_end[1:]

        graph_sums = weighted_sum[indx_after, :] - weighted_sum[indx_before, :]  # [g, j]

        result = self.mlp_func(graph_sums)  # [g, q]
        return result


class GraphFeaturesStackPadRefImplementation(graph_tops.GraphFeaturesFromStackedNodeFeaturesBase):
    """
    Do the sum by putting everything into a padded structure and then summing over one of the dimensions

    THE NODE FEATURES MUST COME IN ORDER (IE ALL NODES OF A GRAPH ARE GROUPED TOGETHER)
    """

    def forward(self, node_features, node_grp_start_with_end, max_size):
        """
        :param node_features: [v*, h]
        :param node_grp_start_with_end: [g +1] index into node features for where each new graph starts
         with concatenated on the end the index of the next imaginary graph ie  node_features.shape[0]
         so that when you take off one you index the last node
        :param max_size: the largest number of nodes in a graph.
        """
        proj_up = self.mlp_project_up(node_features)  # [v*, j]
        gate_logit = self.mlp_gate(node_features)  # [v*, 1]
        gate = torch.sigmoid(gate_logit)  # [v*, j]
        gated_vals = gate * proj_up

        padded = torch.zeros(node_grp_start_with_end.shape[0] - 1, max_size, *proj_up.shape[1:],
                             device=str(node_features.device), dtype=data_types.TORCH_FLT)

        for i, (start, end) in enumerate(zip(node_grp_start_with_end[:-1],
                                             node_grp_start_with_end[1:])):
            padded[i, :end-start, ...] = gated_vals[start:end, ...]

        graph_sums = torch.sum(padded, dim=1)  # [g,j]

        result = self.mlp_func(graph_sums)  # [g, q]
        return result



def test_graph_feats_stack_index_add_vs_stack_pad():
    torch.manual_seed(0)

    rng = np.random.RandomState(1010)
    num_nodes = 16
    graph_dim = 52
    out_dim = 7
    x_feats = torch.from_numpy(rng.randn(num_nodes, graph_dim).astype(data_types.NP_FLT))

    node_to_graph_id = torch.tensor([0]* 5 + [1]*7 + [2]*4)
    node_grp_start_with_end = torch.tensor([0] + [5] + [12] + [16])
    max_size = 11

    mlp_project_up = mlp.get_mlp(mlp.MlpParams(graph_dim, out_dim, [17]))
    mlp_gate = mlp.get_mlp(mlp.MlpParams(graph_dim, 1, [28]))
    mlp_func = mlp.get_mlp(mlp.MlpParams(out_dim, 9, [14]))

    graph_top1 = ggnn_sparse.GraphFeaturesStackIndexAdd(mlp_project_up, mlp_gate, mlp_func)
    out_g1 = graph_top1(x_feats, node_to_graph_id).detach().cpu().numpy()
    assert out_g1.shape == (3, 9)

    graph_top2 = GraphFeaturesStackPadRefImplementation(mlp_project_up, mlp_gate, mlp_func)
    out_g2 = graph_top2(x_feats, node_grp_start_with_end, max_size).detach().cpu().numpy()

    np.testing.assert_array_almost_equal(out_g1, out_g2)


def test_graph_feats_stack_index_add_vs_cumsum():
    torch.manual_seed(0)

    rng = np.random.RandomState(1010)
    num_nodes = 16
    graph_dim = 52
    out_dim = 7
    x_feats = torch.from_numpy(rng.randn(num_nodes, graph_dim).astype(data_types.NP_FLT))

    node_to_graph_id = torch.tensor([0]* 5 + [1]*7 + [2]*4)
    node_grp_start_with_end = torch.tensor([0] + [5] + [12] + [16])
    max_size = 11

    mlp_project_up = mlp.get_mlp(mlp.MlpParams(graph_dim, out_dim, [17]))
    mlp_gate = mlp.get_mlp(mlp.MlpParams(graph_dim, 1, [28]))
    mlp_func = mlp.get_mlp(mlp.MlpParams(out_dim, 9, [14]))

    graph_top1 = ggnn_sparse.GraphFeaturesStackIndexAdd(mlp_project_up, mlp_gate, mlp_func)
    out_g1 = graph_top1(x_feats, node_to_graph_id).detach().cpu().numpy()
    assert out_g1.shape == (3, 9)

    graph_top2 = GraphFeaturesStackCSRefImplementation(mlp_project_up, mlp_gate, mlp_func)
    out_g2 = graph_top2(x_feats, node_grp_start_with_end).detach().cpu().numpy()

    np.testing.assert_array_almost_equal(out_g1, out_g2)




