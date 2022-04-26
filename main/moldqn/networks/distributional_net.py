import numpy as np
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


class DistributionalMultiLayerNetwork(nn.Module):
    """
    Simple feed forward network
    Encode the molecule with Morgan fingerprint and then forward a MLP network
    """

    def __init__(self, args, device='cpu', vmin=0, vmax=1):
        super(DistributionalMultiLayerNetwork, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.args = args

        self.encoder = functools.partial(mol2fp, radius=args['fingerprint_radius'], length=args['fingerprint_length'])

        self.device = device
        self.dense = nn.Sequential()

        hparams_layers = [self.args['n_neurons']] * self.args['n_layers']
        # The input length is the size of Morgan fingerprint plus steps
        input_size = self.args['fingerprint_length'] + 1
        output_size = self.args['num_bootstrap_heads']
        hparams_layers = [input_size] + hparams_layers + [output_size]

        self.dense.add_module('dense_%i' % 1, nn.Linear(hparams_layers[0], hparams_layers[1]))

        for i in range(1, len(hparams_layers) - 1):

            # if self.args['batch_norm']:
            #     self.dense.add_module('BN_%i' % i, nn.BatchNorm1d(1))
            self.dense.add_module('%s_%i' % (self.args['activation'], i),
                                  getattr(nn, self.args['activation'])())
            self.dense.add_module('Dropout_%i' % i, nn.Dropout(self.args['dropout']))

            self.dense.add_module('dense_%i' % (i+1), NoisyLinear(hparams_layers[i], hparams_layers[i+1]))

    def forward(self, x, step=None):
        """
        Value calculation.
        :param x: input shape: (batch_size, 1, input_size)
            input_size = len(fingerprint) + 1 (step)
        :return:
        """
        if step is not None:
            input_tensor = self.encoder(x, step).to(self.device)
            return F.softmax(self.dense(input_tensor).view(-1, self.args['num_bootstrap_heads']))
        else:
            return F.softmax(self.dense(x).view(-1, self.args['num_bootstrap_heads']))

    def reset_noise(self):
        for layer in self.dense:
            try:
                layer.reset_noise()
                # print('noise resetting')
            except:
                continue


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


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
