

from torch import nn

from graph_neural_networks.core import mlp
from graph_neural_networks.sparse_pattern import ggnn_sparse
from graph_neural_networks.sparse_pattern import graph_as_adj_list
from graph_neural_networks.ggnn_general import ggnn_base


class GraphEmbedder(nn.Module):
    def __init__(self, hidden_layer_size, edge_names, embedding_dim, num_layers):
        super().__init__()
        self.ggnn = ggnn_sparse.GGNNSparse(
            ggnn_base.GGNNParams(hidden_layer_size, edge_names, num_layers))

        mlp_project_up = mlp.get_mlp(mlp.MlpParams(hidden_layer_size, embedding_dim, []))
        mlp_gate = mlp.get_mlp(mlp.MlpParams(hidden_layer_size, embedding_dim, []))
        mlp_down = lambda x: x

        self.embedding_dim = embedding_dim
        self.ggnn_top = ggnn_sparse.GraphFeaturesStackIndexAdd(mlp_project_up, mlp_gate, mlp_down)

    def forward(self, g_adjlist: graph_as_adj_list.DirectedGraphAsAdjList):
        g_adjlist: graph_as_adj_list.DirectedGraphAsAdjList = self.ggnn(g_adjlist)
        graph_feats = self.ggnn_top(g_adjlist.node_features, g_adjlist.node_to_graph_id)
        return graph_feats  # [b, embedding_dim]