
import typing

import torch
import torch.nn as nn

from graph_neural_networks.core import utils


class GGNNParams(typing.NamedTuple):
    hlayer_size: int
    edge_names: typing.List[str]  # string of the node groups which are associated with different relationships.
    # SHOULD BE ORDERED THE SAME WAY AS THE ADJACENCY MATRIX FOURTH DIM
    num_layers: int  # this is the number of time steps to do message passing. often denoted T in papers.


APPENDER_TO_HIDDEN_NAMES = '_bond_proj_'
# ^ to avoid name clashes with other attributes.


class GGNNBase(nn.Module):
    """
    Gated Graph Neural Network (node features)

    Li, Y., Tarlow, D., Brockschmidt, M. and Zemel, R., 2015. Gated graph sequence neural networks.
    arXiv preprint arXiv:1511.05493.

    see also
    Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O. and Dahl, G.E., 2017.
    Neural message passing for quantum chemistry. arXiv preprint arXiv:1704.01212.
    """

    def __init__(self, params: GGNNParams):
        super().__init__()
        self.params = params

        self.GRU_hidden = nn.GRUCell(self.params.hlayer_size, self.params.hlayer_size)

        self.A_hidden = nn.ModuleDict(
            {k + APPENDER_TO_HIDDEN_NAMES: nn.Linear(self.params.hlayer_size, self.params.hlayer_size) for k in self.params.edge_names}
        )

    def get_edge_names_and_projections(self):
        return ((k[:-len(APPENDER_TO_HIDDEN_NAMES)], v) for k,v in self.A_hidden.items())

    def forward(self, *input):
        raise NotImplementedError

