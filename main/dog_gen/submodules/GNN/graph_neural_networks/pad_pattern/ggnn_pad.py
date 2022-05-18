


import typing

import torch
import torch.nn as nn

from ..core import utils
from ..core import data_types
from ..ggnn_general import ggnn_base


class GGNNPad(ggnn_base.GGNNBase):
    def forward(self, atom_feat: torch.FloatTensor, adj: torch.FloatTensor, nodes_on_mask):
        """
        :param atom_feat: tensor of starting atom features, shape is [b, v, h]
        :param adj: the adjacency matrix, shape is [b, v, v, e]
        :param nodes_on_mask: Boolean array indicating which nodes actual exist, shape is [b,v]
        :return: the computed features for each node, shape is [b, v, h]
        """

        batch_size, num_nodes, num_hidden = atom_feat.shape

        hidden = atom_feat.view(-1, num_hidden)  # [b*v, h]
        for t in range(self.params.num_layers):
            message = torch.zeros(batch_size, num_nodes, self.params.hlayer_size,
                                                    device=str(hidden.device),
                                                    dtype=data_types.TORCH_FLT)  # [b, v, h]
            for e, edge_type in enumerate(self.params.edge_names):
                M_t = self.A_hidden[edge_type + ggnn_base.APPENDER_TO_HIDDEN_NAMES](hidden)  # [b*v, h]
                M_t = M_t.view(-1, num_nodes, num_hidden)  # [b, v, h]
                message = message + torch.bmm(adj[:, :, :, e], M_t)  # [b, v, v] @ [b, v, h] = [b, v, h]

            message_unrolled = message.view(-1, num_hidden)  # [b*v, h]
            hidden = self.GRU_hidden(message_unrolled, hidden)  # [b*v, h]

        hidden = hidden.view(batch_size, num_nodes, num_hidden)  # [b, v, h]
        hidden = hidden * nodes_on_mask[:, :, None].to(hidden.dtype)
        # ^ the nodes which dont exist may have picked up non zero values (from bias etc from GRU)
        # -- therefore turn these back off
        return hidden


class GraphFeatureTopOnly(nn.Module):
    """
    This runs on node features and computes graph level feature(s).
    """
    #todo: add mask so do not do the gated sum over invalid items.
    def __init__(self, mlp_project_up, mlp_gate, mlp_func):
        super().__init__()
        self.mlp_project_up = mlp_project_up  # net that goes from [None, h'] to [None, j] with j>h usually
        self.mlp_gate = mlp_gate  # net that goes from [None, h'] to [None, 1]
        self.mlp_func = mlp_func  # net that goes from [None, j] to [None, q]

    def forward(self, node_features):
        """
        :param node_features: shape is [b, v, h']
        :return: shape [b,q]
        """
        b, v, h = node_features.shape
        proj_up = self.mlp_project_up(node_features.view(-1, h)).view(b,v,-1)  # [b,v,j]
        gate_logit = self.mlp_gate(node_features.view(-1, h)).view(b,v,1)   # [b,v,1]
        gate = torch.sigmoid(gate_logit)

        gated_sum = torch.sum(proj_up * gate, dim=1)  # [b,j]
        result = self.mlp_func(gated_sum)
        return result
