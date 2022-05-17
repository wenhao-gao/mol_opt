
import typing

import torch
from torch import nn as nn


class GraphFeaturesFromStackedNodeFeaturesBase(nn.Module):
    """
    Attention weighted sum using the computed features.
    The trickiness in performing these operations is that we need to do a sum over nodes. For different graphs we
    have different numbers of nodes and so batching is difficult. The children of this class try doing this in different
    ways.

    Base class for modules that take in the stacked node feature matrix [v*, h] and produce embeddings of graphs
     [g, h']. These are called aggregation functions by Johnson (2017).

    Johnson DD (2017) Learning Graphical State Transitions. In: ICLR, 2017.

    Li Y, Vinyals O, Dyer C, et al. (2018) Learning Deep Generative Models of Graphs.
    arXiv [cs.LG]. Available at: http://arxiv.org/abs/1803.03324.
    """
    def __init__(self, mlp_project_up: typing.Callable[[torch.Tensor], torch.Tensor],
                 mlp_gate: typing.Callable[[torch.Tensor], torch.Tensor],
                 mlp_func: typing.Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.mlp_project_up = mlp_project_up  # net that goes from [None, h'] to [None, j] with j>h usually
        self.mlp_gate = mlp_gate  # net that goes from [None, h'] to [None, 1 or h']
        self.mlp_func = mlp_func  # net that goes from [None, j] to [None, q]

