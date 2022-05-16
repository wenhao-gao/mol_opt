
import torch

from graph_neural_networks.core import data_types
from graph_neural_networks.ggnn_general.graph_tops import GraphFeaturesFromStackedNodeFeaturesBase

from ..core import data_types
from ..ggnn_general import ggnn_base
from . import graph_as_adj_list


class GGNNSparse(ggnn_base.GGNNBase):
    def forward(self, graphs: graph_as_adj_list.DirectedGraphAsAdjList):

        hidden = graphs.node_features
        num_nodes = hidden.shape[0]

        for t in range(self.params.num_layers):
            message = torch.zeros(num_nodes, self.params.hlayer_size,
                                                    device=str(hidden.device),
                                                    dtype=data_types.TORCH_FLT)

            for edge_name, projection in self.get_edge_names_and_projections():
                adj_list = graphs.edge_type_to_adjacency_list_map[edge_name]
                if adj_list is None:
                    continue  # no edges of this type
                projected_feats = projection(hidden)
                #todo: potentially wasteful doing this projection on all nodes (ie many may not
                # be connected by all kinds of edge) -- less concerned about this for molecules
                message.index_add_(0, adj_list[0], projected_feats.index_select(0, adj_list[1]))

            hidden = self.GRU_hidden(message, hidden)

        return graph_as_adj_list.DirectedGraphAsAdjList(hidden, graphs.edge_type_to_adjacency_list_map, graphs.node_to_graph_id)



class GraphFeaturesStackIndexAdd(GraphFeaturesFromStackedNodeFeaturesBase):
    """
    Do the sum by Pytorch's index_add method.
    """
    def forward(self, node_features, node_to_graph_id):
        """
        :param node_features: [v*, h]
        :param node_to_graph_id:  for each node index the graph it belongs to [v*]
        """

        proj_up = self.mlp_project_up(node_features)  # [v*, j]
        gate_logit = self.mlp_gate(node_features)  # [v*, _]
        gate = torch.sigmoid(gate_logit)  # [v*, _]
        gated_vals = gate * proj_up

        num_graphs = node_to_graph_id.max() + 1
        graph_sums = torch.zeros(num_graphs, gated_vals.shape[1],
                                                    device=str(node_features.device),
                                                    dtype=data_types.TORCH_FLT)  # [g, j]
        graph_sums.index_add_(0, node_to_graph_id, gated_vals)

        result = self.mlp_func(graph_sums)  # [g, q]
        return result