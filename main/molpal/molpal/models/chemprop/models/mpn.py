from argparse import Namespace
from typing import List, Optional, Tuple, Union
from functools import reduce

from rdkit import Chem
import torch
from torch import FloatTensor, LongTensor
import torch.nn as nn

from ..features import get_atom_fdim, get_bond_fdim, mol2graph
from ..nn_utils import index_select_ND, get_activation_function

class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`Namespace` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size), requires_grad=False
        )

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(
            self.atom_fdim + self.hidden_size, self.hidden_size
        )

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(
                self.hidden_size + self.atom_descriptors_size,
                self.hidden_size + self.atom_descriptors_size
            )

    def forward(self,
                components: Tuple[
                    FloatTensor, FloatTensor, LongTensor, LongTensor, 
                    LongTensor, List[Tuple[int, int]], List[Tuple[int, int]], 
                    Optional[LongTensor], Optional[LongTensor]
                ],
                # mol_graph: BatchMolGraph,
                # atom_descriptors_batch: List[np.ndarray] = None
                ) -> torch.FloatTensor:
        """Encodes a batch of molecular graphs.

        Parameters
        ----------
        components : Tuple
            The components of a batch of molecular graphs returned from a call
            to BatchMolGraph.get_components(). The components are in the 
            following order:
                f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, b2b, a2a 
            NOTE: b2b and a2a are lazily computed in a BatchMolGraph and are 
            None unless preceding calls to BatchMolGraph.get_b2b() and
            BatchMolGraph.get_a2a(), respectively, are performed.
        :param mol_graph: A :class:`~chemprop.features.featurization.
            BatchMolGraph` representing a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing 
            additional atomic descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` 
            containing the encoding of each molecule.
        """
        # if atom_descriptors_batch is not None:
        #     # padding the first with 0 to match the atom_hiddens
        #     atom_descriptors_batch = [
        #         np.zeros([1, atom_descriptors_batch[0].shape[1]])
        #     ] + atom_descriptors_batch
        #     atom_descriptors_batch = torch.from_numpy(
        #         np.concatenate(atom_descriptors_batch, axis=0)
        #     ).float().to(device)

        # components = mol_graph.get_components(atom_messages=self.atom_messages)
        # print(components)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, b2b, a2a = \
            components

        if self.atom_messages and a2a is None:
            raise ValueError(
                'a2a is "None" but atom_messages is True! Reminder: a2a '
                'component of BatchMolGraph is lazily computed and must be '
                'precalculated via a call to BatchMolGraph.get_a2a() before '
                'BatchMolGraph.get_components()'
            )
        if self.atom_messages:
            f_bonds = f_bonds[:, :get_bond_fdim(self.atom_messages)]
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            f_bonds = f_bonds
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        if self.atom_messages:
            for _ in range(self.depth - 1):
                if self.undirected:
                    message = (message + message[b2revb]) / 2

                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim

                message = self.W_h(message)
                message = self.act_func(input + message)  # num_bonds x hidden_size
                message = self.dropout_layer(message)  # num_bonds x hidden
        else:
            for _ in range(self.depth - 1):
                if self.undirected:
                    message = (message + message[b2revb]) / 2

                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message a_message = sum(nei_a_message) rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

                message = self.W_h(message)
                message = self.act_func(input + message)  # num_bonds x hidden_size
                message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        # num_atoms x max_num_bonds x hidden
        nei_a_message = index_select_ND(message, a2x)
        # num_atoms x hidden
        a_message = nei_a_message.sum(dim=1)
        # num_atoms x (atom_fdim + hidden)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        # num_atoms x hidden
        atom_hiddens = self.act_func(self.W_o(a_input))
        # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # concatenate the atom descriptors
        # if atom_descriptors_batch is not None:
        #     # num_atoms x (hidden + descriptor size)
        #     atom_hiddens = torch.cat(
        #         [atom_hiddens, atom_descriptors_batch], dim=1
        #     )
        #     # num_atoms x (hidden + descriptor size
        #     atom_hiddens = self.atom_descriptors_layer(atom_hiddens)
        #     # num_atoms x (hidden + descriptor size)
        #     atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation=='mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation=='sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation=='norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes 
    input as needed."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.Namespace` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super().__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim(
            atom_messages=args.atom_messages
        )

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.atom_descriptors = args.atom_descriptors

        if self.features_only:
            return

        # if args.mpn_shared:
        #     self.encoder = nn.ModuleList(
        #         [MPNEncoder(args, self.atom_fdim, self.bond_fdim)]
        #         * args.number_of_molecules
        #     )
        # else:
            # self.encoder = nn.ModuleList([
            #     MPNEncoder(args, self.atom_fdim, self.bond_fdim)
            #     for _ in range(args.number_of_molecules)]
            # )
        self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batches: Union[
                    List[List[str]], List[List[Chem.Mol]], List[Tuple]
                ],
                #features_batch: List[np.ndarray] = None,
                #atom_descriptors_batch: List[np.ndarray] = None
                ) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit 
            molecules, or a
            :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional 
            features.
        :param atom_descriptors_batch: A list of numpy arrays containing 
            additional atom descriptors.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` 
            containing the encoding of each molecule.
        """
        if all(isinstance(x, (str, Chem.Mol)) for x in batches[0]):
        # if type(batches[0]) != BatchMolGraph:
            # TODO: handle atom_descriptors_batch with multiple molecules per input
            # if self.atom_descriptors == 'feature':
            #     if len(batches[0]) > 1:
            #         raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
            #                                   'per input (i.e., number_of_molecules = 1).')

            #     batches = [
            #         mol2graph(b, atom_descriptors_batch) for b in batches
            #     ]
            # else:
            batches = [mol2graph(b) for b in batches]

        # if self.use_input_features:
        #     features_batch = torch.from_numpy(
        #         np.stack(features_batch)
        #     ).float().to(self.device)

        #     if self.features_only:
        #         return features_batch

        # if self.atom_descriptors == 'descriptor':
        #     if len(batches) > 1:
        #         raise NotImplementedError(
        #             'Atom descriptors are currently only supported with one '
        #             'molecule per input (i.e., number_of_molecules = 1).'
        #         )

        #     encodings = [
        #         self.encoder(b, atom_descriptors_batch) for b in batches
        #         #zip(self.encoder, batch)
        #     ]
        # else:
        encodings = [
            self.encoder(b) for b in batches
            # self.encoder(b, atom_descriptors_batch) for b in batch
            #zip(self.encoder, batch)
        ]
            # encodings = [enc(b) for enc, b in zip(self.encoder, batch)]

        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        # if self.use_input_features:
        #     if len(features_batch.shape) == 1:
        #         features_batch = features_batch.view(1, -1)

        #     output = torch.cat([output, features_batch], dim=1)

        return output
