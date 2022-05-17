
import typing

from dataclasses import dataclass

import torch.nn as nn

from . import data_types

@dataclass
class MlpParams:
    input_dim: int
    output_dim: int
    hidden_sizes: typing.List[int]
    dropout: float = 0.
    batchnorm: bool = False
    activation = nn.ReLU


def get_mlp(params: MlpParams):
    params = params

    layer_sizes = [params.input_dim] + params.hidden_sizes + [params.output_dim]
    layer_dims = list(zip(layer_sizes[:-1], layer_sizes[1:]))

    layers = []
    for i, (input_dim, output_dim) in enumerate(layer_dims):
        if i != 0:
            if params.batchnorm:
                layers.append(nn.BatchNorm1d(input_dim, affine=False))
            if params.dropout != 0.:
                layers.append(nn.Dropout(p=params.dropout))

        layers.append(nn.Linear(input_dim, output_dim))

        last_layer_flag = i == len(layer_dims) - 1
        if not last_layer_flag:
            layers.append(params.activation())

    out = nn.Sequential(*layers)
    out = out.to(dtype=data_types.TORCH_FLT)
    return out

