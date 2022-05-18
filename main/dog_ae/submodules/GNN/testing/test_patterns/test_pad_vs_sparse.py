
import numpy as np
import torch

from graph_neural_networks.sparse_pattern import graph_as_adj_list
from graph_neural_networks.sparse_pattern import ggnn_sparse
from graph_neural_networks.pad_pattern import ggnn_pad
from graph_neural_networks.ggnn_general import ggnn_base


def test_sparse_vs_pad():
    node_feats = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [9, 10, 11], [11, 11.1, 12.4], [18, 11.1, 22.4], [24, 15.31, 18.4],
         [16, 10.1, 17.4]])
    graph_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    edges = {'asingle': torch.tensor([[0, 1, 7, 6, 3, 4],
                                      [1, 2, 6, 5, 4, 3]]),
             'bdouble': torch.tensor([[2, 1],
                                      [0, 0]
                                      ])
             }
    graph = graph_as_adj_list.DirectedGraphAsAdjList(node_feats, edges, graph_ids)
    atom_feats, adj_mat, *_ = graph.return_padded_repr()
    nodes_on = torch.any(atom_feats != 0, dim=-1)

    net_params = ggnn_base.GGNNParams(3, ['asingle', 'bdouble'], 3)

    torch.manual_seed(43)
    ggnn_sparse_net = ggnn_sparse.GGNNSparse(net_params)
    out_sparse = ggnn_sparse_net(graph)
    node_feats_out_sparse, *_ = out_sparse.return_padded_repr()

    torch.manual_seed(43)
    ggnn_padded_net = ggnn_pad.GGNNPad(net_params)
    node_feats_out_pad = ggnn_padded_net(atom_feats, adj_mat, nodes_on)

    np.testing.assert_array_almost_equal(node_feats_out_sparse.cpu().detach().numpy(),
                                         node_feats_out_pad.cpu().detach().numpy())



