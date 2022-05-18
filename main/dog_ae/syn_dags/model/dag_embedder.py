
import warnings
import enum

from torch import nn

from graph_neural_networks.sparse_pattern import ggnn_sparse
from graph_neural_networks.sparse_pattern import graph_as_adj_list
from graph_neural_networks.core import mlp


from .. data import synthesis_trees
from .. utils import settings


class AggrType(enum.Enum):
    FINAL_NODE = "final_node"   # take the representation from the final node in the DAG as final representation of DAG


class DAGEmbedder(nn.Module):
    def __init__(self, dag_gnn: ggnn_sparse.GGNNSparse, aggr_type: AggrType, final_dim):
        super().__init__()
        self.gnn = dag_gnn
        self.aggr_type = aggr_type
        self.final_mlp = mlp.get_mlp(mlp.MlpParams(dag_gnn.params.hlayer_size, final_dim, []))


    def forward(self, x_data: synthesis_trees.PredOutBatch):
        dog_in = x_data.dags_for_inputs

        new_features: graph_as_adj_list.DirectedGraphAsAdjList = self.compute_dog_node_feats(dog_in)

        if self.aggr_type is AggrType.FINAL_NODE:
            out_feats = new_features.node_features[x_data.final_molecule_indcs, ...]
        else:
            raise NotImplementedError

        out_feats = self.final_mlp(out_feats)

        return out_feats

    def compute_dog_node_feats(self, dog_in: graph_as_adj_list.DirectedGraphAsAdjList):
        if dog_in.node_features.shape[1] == 1 or dog_in.node_features.dtype == settings.TORCH_INT:
            warnings.warn("Single dimensional or integer features being used for nodes in DAG -- is this meant?")
        new_features: graph_as_adj_list.DirectedGraphAsAdjList = self.gnn(dog_in)
        return new_features

    @property
    def embedding_dim(self):
        return self.gnn.params.hlayer_size
