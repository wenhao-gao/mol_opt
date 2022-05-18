

import typing

import numpy as np
import torch

from ..core import nd_ten_ops
from ..core import utils


class DirectedGraphAsAdjList(object):
    def __init__(self, node_features: nd_ten_ops.Nd_Ten,
                 edge_type_to_adjacency_list_map: typing.Dict[str, nd_ten_ops.Nd_Ten],
                 node_to_graph_id: nd_ten_ops.Nd_Ten):
        """
        :param node_features: [v*, h]. Matrix of node features
        :param edge_type_to_adjacency_list_map: dictionary which for every edge type (represented by keys) we have an
        [2, E*] matrix, each row in here represents the bond. Indexing into the node features matrix
        Edges go from [1,;] index to [0,:]
        :param node_to_graph_id: [v*] (should be in order ie nodes in the same graph should be consecutive).
         We have one for each node.For each node it indexes into the node for which they belong.
        """
        self.node_features = node_features
        self.edge_type_to_adjacency_list_map = edge_type_to_adjacency_list_map
        # ^ in Numpy mode then we have zero dim arrays. If in Torch mode then we will have None instead.
        self.node_to_graph_id = node_to_graph_id
        self.max_num_graphs = self.node_to_graph_id.max() + 1  # plus one to deal with fact that index from zero.

    @property
    def variant(self) -> nd_ten_ops.NdTensor:
        """
        Works out whether underlying stored in Pytorch Tensors or Numpy arrays.
        :return:
        """
        return nd_ten_ops.work_out_nd_or_tensor(self.node_features)

    def return_padded_repr(self):
        num_nodes_per_graph = nd_ten_ops.bincount(self.node_to_graph_id)
        max_graph_size = num_nodes_per_graph.max()

        # Work out where the start indcs of each graph is.
        start_indcs = nd_ten_ops.to_np(num_nodes_per_graph.cumsum(0)).tolist()
        start_indcs = [0] + start_indcs[:-1]

        num_adj_mats = len(self.edge_type_to_adjacency_list_map)
        if self.variant is nd_ten_ops.NdTensor.NUMPY:
            adj_mat_out = np.zeros((self.max_num_graphs, max_graph_size, max_graph_size, num_adj_mats))
            node_feats_out = np.zeros((self.max_num_graphs, max_graph_size, self.node_features.shape[1]))

        else:
            adj_mat_out = torch.zeros(*(self.max_num_graphs, max_graph_size, max_graph_size, num_adj_mats),
                                      dtype=self.node_features.dtype,
                                      device=str(self.node_features.device)
                                      )
            node_feats_out = torch.zeros((self.max_num_graphs, max_graph_size, self.node_features.shape[1]),
                                         dtype=self.node_features.dtype,
                                         device=str(self.node_features.device)
                                         )


        edge_order = []
        for edge_idx, (edge_name, edge_details) in enumerate(sorted(self.edge_type_to_adjacency_list_map.items(),
                                                                    key=lambda x: x[0])):
            edge_order.append(edge_name)
            for e_into, e_from in edge_details.transpose(1,0):
                graph_id = int(self.node_to_graph_id[e_into])
                assert graph_id == int(self.node_to_graph_id[e_from])

                offset = start_indcs[graph_id]
                adj_mat_out[graph_id, int(e_into)-offset, int(e_from)-offset, edge_idx] = 1.

        last_node_id = -1
        idx_in = 0
        for i, node_feats in enumerate(self.node_features):
            graph_id = int(self.node_to_graph_id[i])
            if graph_id != last_node_id:
                last_node_id = graph_id
                idx_in = 0
            node_feats_out[graph_id, idx_in, :] = node_feats
            idx_in += 1

        return node_feats_out, adj_mat_out, num_nodes_per_graph, edge_order


    @classmethod
    def concatenate(cls, grps):

        # Set up the lists will be used to store the new components
        node_features_new = []

        all_keys = set([frozenset(g.edge_type_to_adjacency_list_map.keys()) for g in grps])
        assert len(all_keys) == 1, "inconsistent edges among graph groups"
        adjacency_list_for_all_edges_new = {k: [] for k in all_keys.pop()}
        node_to_graph_id_new = []

        # Now go through and add the respective matrices nodes to these groups
        max_node_index_so_far = 0
        max_num_grps_so_far = 0
        # ^ the indices will all need to be shifted up as add more graphs this variable will record by how much
        for g in grps:

            node_features_new.append(g.node_features)

            for k, v in g.edge_type_to_adjacency_list_map.items():
                if v.shape[0] == 0:
                    continue
                    # sometimes it is empty (ie not every graph has to contain every edge type) so skip these
                adjacency_list_for_all_edges_new[k].append(v + max_node_index_so_far)
            node_to_graph_id_new.append(g.node_to_graph_id + max_num_grps_so_far)

            max_num_grps_so_far += g.max_num_graphs
            max_node_index_so_far += g.node_features.shape[0]


        # Now concatenate together
        node_features_new = nd_ten_ops.concatenate(node_features_new, axis=0)
        def cat_or_set_zero(v):
            if len(v) == 0:
                return np.array([[], []], dtype=np.int64)
            else:
                return nd_ten_ops.concatenate(v, axis=1)
        adjacency_list_for_all_edges_new = {k: cat_or_set_zero(v) for k, v in
                                            adjacency_list_for_all_edges_new.items()}
        node_to_graph_id_new = nd_ten_ops.concatenate(node_to_graph_id_new)

        return DirectedGraphAsAdjList(node_features_new, adjacency_list_for_all_edges_new, node_to_graph_id_new)

    def inplace_from_np_to_torch(self, torch_device=None):
        def func_to_map(x):
            return None if (x is None or x.size == 0) else torch.from_numpy(x).to(device=torch_device)
        self._map_all_props(func_to_map)
        return self

    def inplace_torch_to(self, *args, **kwargs):
        def func_to_map(x):
            return x if x is None else x.to(*args, **kwargs)
        self._map_all_props(func_to_map)
        return self

    def _map_all_props(self, func):
        self.node_features = func(self.node_features)
        self.node_to_graph_id = func(self.node_to_graph_id)
        self.edge_type_to_adjacency_list_map = {k: func(v) for k, v in self.edge_type_to_adjacency_list_map.items()}
