

import numpy as np
import torch

from graph_neural_networks.sparse_pattern import graph_as_adj_list


def test_return_padded_repr():
    node_feats = np.array([[1,2,3],[4,5,6],[7,8,9],[9,10,11],[11,11.1,12.4],[18,11.1,22.4],[24,15.31,18.4],[16,10.1,17.4]])
    graph_ids = np.array([0,0,0,1,1,2,2,2])
    edges = {'asingle': np.array([[0,1,7,6,3,4],
                                 [1,2,6,5,4,3]]),
             'bdouble': np.array([[2,1],
                                 [0, 0]
                                 ])
             }

    expected_node_feats_padded = np.array([ [[1.,2,3],[4.,5,6],[7.,8,9]],
                                            [
                                             [9.,10,11],[11,11.1,12.4], [0.,0.,0.]],
                                            [[18,11.1,22.4],[24,15.31,18.4],[16,10.1,17.4]]])
    expected_adj_mats = np.array([[[[0,0],[1.,0],[0,0]],[[0,1],[0,0],[1.,0]],[[0,1],[0,0],[0,0]]],
                                  [[[0,0],[1.,0],[0,0]],[[1.,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]],
                                  [[[0,0],[0,0],[0,0]],[[1.,0],[0,0],[0,0]],[[0,0],[1.,0],[0,0]]]
                                  ])

    graph = graph_as_adj_list.DirectedGraphAsAdjList(node_feats, edges, graph_ids)
    node_feats_out, adj_mat_out, num_nodes_out, edges = graph.return_padded_repr()

    num_nodes_out_expected = np.array([3,2,3])

    np.testing.assert_array_almost_equal(node_feats_out, expected_node_feats_padded)
    np.testing.assert_array_almost_equal(adj_mat_out, expected_adj_mats)
    np.testing.assert_array_almost_equal(num_nodes_out, num_nodes_out_expected)




def test_return_padded_repr_torch():
    node_feats = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[9,10,11],[11,11.1,12.4],[18,11.1,22.4],[24,15.31,18.4],[16,10.1,17.4]])
    graph_ids = torch.tensor([0,0,0,1,1,2,2,2])
    edges = {'asingle': torch.tensor([[0,1,7,6,3,4],
                                 [1,2,6,5,4,3]]),
             'bdouble': torch.tensor([[2,1],
                                 [0, 0]
                                 ])
             }

    expected_node_feats_padded = np.array([ [[1.,2,3],[4.,5,6],[7.,8,9]],
                                            [
                                             [9.,10,11],[11,11.1,12.4], [0.,0.,0.]],
                                            [[18,11.1,22.4],[24,15.31,18.4],[16,10.1,17.4]]])
    expected_adj_mats = np.array([[[[0,0],[1.,0],[0,0]],[[0,1],[0,0],[1.,0]],[[0,1],[0,0],[0,0]]],
                                  [[[0,0],[1.,0],[0,0]],[[1.,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]],
                                  [[[0,0],[0,0],[0,0]],[[1.,0],[0,0],[0,0]],[[0,0],[1.,0],[0,0]]]
                                  ])

    graph = graph_as_adj_list.DirectedGraphAsAdjList(node_feats, edges, graph_ids)
    node_feats_out, adj_mat_out, num_nodes_out, edges = graph.return_padded_repr()

    num_nodes_out_expected = np.array([3,2,3])

    np.testing.assert_array_almost_equal(node_feats_out.detach().numpy(), expected_node_feats_padded)
    np.testing.assert_array_almost_equal(adj_mat_out.detach().numpy(), expected_adj_mats)
    np.testing.assert_array_almost_equal(num_nodes_out.detach().numpy(), num_nodes_out_expected)

def test_concat_np():
    graphs = [
        graph_as_adj_list.DirectedGraphAsAdjList(
            np.array([[1., 2, 3], [4, 5, 6], [7, 8, 9]]),
            {'asingle': np.array([[0, 1],
                                      [1, 2]]),
             'bdouble': np.array([[2, 1],
                                      [0, 0]
                                      ])
             },
            np.array([0, 0, 0])
        ),
        graph_as_adj_list.DirectedGraphAsAdjList(
            np.array([[41., 32, 53], [14, 15, 36], [47, 58, 19], [45, 10, 11]]),
            {'asingle': np.array([[1, 0,2,3],
                                      [0, 1,3,2]]),
             'bdouble': np.array([[3, 0],
                                      [0, 3]
                                      ])
             },
            np.array([0, 0, 0,0])
        ),
        graph_as_adj_list.DirectedGraphAsAdjList(
            np.array([[11., 22, 33], [42, 25, 61]]),
            {'asingle': np.array([[1, 0],
                                      [0, 1]]),
             'bdouble': np.array([[0, 1],
                                      [1, 0]
                                      ])
             },
            np.array([0, 0])
        )
    ]
    expected_node_np = np.array([[1., 2, 3], [4, 5, 6], [7, 8, 9],[41., 32, 53], [14, 15, 36],
                                 [47, 58, 19], [45, 10, 11], [11., 22, 33], [42, 25, 61]])
    expect_graph_id_np = np.array([0, 0, 0,1,1,1,1,2,2])
    expected_edges = {
        'asingle': np.array([[0, 1,4,3,5,6,8,7],
                                 [1, 2,3,4,6,5,7,8]]),
        'bdouble': np.array([[2, 1,6,3,7,8],
                                 [0, 0,3,6,8,7]
                                 ])
    }

    graphs_concat = graph_as_adj_list.DirectedGraphAsAdjList.concatenate(graphs)

    np.testing.assert_array_almost_equal(graphs_concat.node_features, expected_node_np)
    np.testing.assert_array_almost_equal(graphs_concat.node_to_graph_id,
                                         expect_graph_id_np)
    for key, val in graphs_concat.edge_type_to_adjacency_list_map.items():
        expected = expected_edges[key]
        np.testing.assert_array_almost_equal(val,
                                             expected, err_msg=f"failed on {key}")



def test_concat_torch():
    graphs = [
        graph_as_adj_list.DirectedGraphAsAdjList(
            torch.tensor([[1., 2, 3], [4, 5, 6], [7, 8, 9]]),
            {'asingle': torch.tensor([[0, 1],
                                      [1, 2]]),
             'bdouble': torch.tensor([[2, 1],
                                      [0, 0]
                                      ])
             },
            torch.tensor([0, 0, 0])
        ),
        graph_as_adj_list.DirectedGraphAsAdjList(
            torch.tensor([[41., 32, 53], [14, 15, 36], [47, 58, 19], [45, 10, 11]]),
            {'asingle': torch.tensor([[1, 0,2,3],
                                      [0, 1,3,2]]),
             'bdouble': torch.tensor([[3, 0],
                                      [0, 3]
                                      ])
             },
            torch.tensor([0, 0, 0,0])
        ),
        graph_as_adj_list.DirectedGraphAsAdjList(
            torch.tensor([[11., 22, 33], [42, 25, 61]]),
            {'asingle': torch.tensor([[1, 0],
                                      [0, 1]]),
             'bdouble': torch.tensor([[0, 1],
                                      [1, 0]
                                      ])
             },
            torch.tensor([0, 0])
        )
    ]
    expected_node_np = np.array([[1., 2, 3], [4, 5, 6], [7, 8, 9],[41., 32, 53], [14, 15, 36],
                                 [47, 58, 19], [45, 10, 11], [11., 22, 33], [42, 25, 61]])
    expect_graph_id_np = np.array([0, 0, 0,1,1,1,1,2,2])
    expected_edges = {
        'asingle': np.array([[0, 1,4,3,5,6,8,7],
                                 [1, 2,3,4,6,5,7,8]]),
        'bdouble': np.array([[2, 1,6,3,7,8],
                                 [0, 0,3,6,8,7]
                                 ])
    }

    graphs_concat = graph_as_adj_list.DirectedGraphAsAdjList.concatenate(graphs)

    np.testing.assert_array_almost_equal(graphs_concat.node_features.cpu().detach().numpy(), expected_node_np)
    np.testing.assert_array_almost_equal(graphs_concat.node_to_graph_id.cpu().detach().numpy(),
                                         expect_graph_id_np)
    for key, val in graphs_concat.edge_type_to_adjacency_list_map.items():
        expected = expected_edges[key]
        np.testing.assert_array_almost_equal(val.cpu().detach().numpy(),
                                             expected, err_msg=f"failed on {key}")



