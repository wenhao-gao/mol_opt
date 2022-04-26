import numpy as np
import functools
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


class MultiLayerNetwork(nn.Module):
    """
    Simple feed forward network
    Encode the molecule with Morgan fingerprint and then forward a MLP network
    """

    def __init__(self, args, device='cpu'):
        super(MultiLayerNetwork, self).__init__()

        self.args = args

        self.encoder = functools.partial(mol2fp, radius=args['fingerprint_radius'], length=args['fingerprint_length'])

        self.device = device
        self.dense = nn.Sequential()

        hparams_layers = [self.args['n_neurons']] * self.args['n_layers']

        # The input length is the size of Morgan fingerprint plus steps
        input_size = self.args['fingerprint_length'] + 1
        output_size = self.args['num_bootstrap_heads']
        hparams_layers = [input_size] + hparams_layers + [output_size]

        for i in range(1, len(hparams_layers)):

            self.dense.add_module('dense_%i' % i, nn.Linear(hparams_layers[i - 1], hparams_layers[i]))

            if i != len(hparams_layers) - 1:
                # if self.args['batch_norm']:
                #     self.dense.add_module('BN_%i' % i, nn.BatchNorm1d(1))
                self.dense.add_module('%s_%i' % (self.args['activation'], i),
                                      getattr(nn, self.args['activation'])())
                self.dense.add_module('Dropout_%i' % i, nn.Dropout(self.args['dropout']))

    def forward(self, x, step=None):
        """
        Value calculation.
        :param x: input shape: (batch_size, 1, input_size)
            input_size = len(fingerprint) + 1 (step)
        :return:
        """
        if step is not None:
            input_tensor = self.encoder(x, step).to(self.device)
            return self.dense(input_tensor)
        else:
            return self.dense(x)


def get_fingerprint(smiles, radius, length):
    """Get Morgan Fingerprint of a specific SMIELS string"""

    if smiles is None:
        return np.zeros((length,))

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return np.zeros((length,))

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, length)

    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def mol2fp(smiles, steps, radius, length):
    """
    Get Morgan fingerprint representation of a list of smiles

    :param smiles: a list of smiles (can be one)
    :param steps: the remaining steps number (can be a list)
    :param radius:
    :param length:
    :return: a tensor, shape=(batch_size, 1, input_size)
        input_size = len(fingerprint) + 1 (step)
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    if isinstance(steps, int):
        steps = [steps] * len(smiles)

    assert len(steps) == len(smiles)

    state_tensor = torch.Tensor(np.vstack([
        np.append(get_fingerprint(act, radius, length), step)
        for act, step in zip(smiles, steps)
    ]))
    state_tensor = torch.unsqueeze(state_tensor, 1)
    return state_tensor
