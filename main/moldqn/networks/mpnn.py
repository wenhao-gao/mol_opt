"""
Code to define the directed message passing network
Slightly modified from https://github.com/swansonk14/chemprop
"""
from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from networks.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from networks.nn_utils import index_select_ND, get_activation_function


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int,
                 bond_fdim: int,
                 device: str = 'cpu'):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.atom_messages = args.atom_messages
        self.use_input_features = args.use_input_features
        self.args = args
        self.device = device

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None,
                duplicate: bool = True) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :param duplicate: Whether to duplicate the feature given.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            features_batch.to(self.device)

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        f_atoms.to(self.device)
        f_bonds.to(self.device)
        a2b.to(self.device)
        b2a.to(self.device)
        b2revb.to(self.device)
        if self.atom_messages:
            a2a.to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds),
                                        dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        if self.use_input_features:
            if duplicate:
                features_batch = torch.Tensor(list(features_batch.data.numpy()) * len(mol_vecs))
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([features_batch.shape[0], 1])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 device: str = 'cpu',
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.device = device
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim, self.device)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(batch, features_batch)

        return output


class MessagePassingNetwork(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: Namespace, device='cpu'):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MessagePassingNetwork, self).__init__()
        self.args = args
        self.args.output_size = args.num_bootstrap_heads
        self.device = device
        self.encoder = MPN(self.args, self.device)

        first_linear_dim = self.args.hidden_size
        if self.args.use_input_features:
            first_linear_dim += self.args.features_dim

        dropout = nn.Dropout(self.args.dropout)
        activation = get_activation_function(self.args.activation)

        # Create FFN layers
        if self.args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.args.ffn_hidden_size)
            ]
            for _ in range(self.args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(self.args.ffn_hidden_size, self.args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(self.args.ffn_hidden_size, self.args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output = self.ffn(self.encoder(*input))
        return output
