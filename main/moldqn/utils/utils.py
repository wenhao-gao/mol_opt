# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Tools for manipulating graphs and converting from atom and pair features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import random
import operator
import pickle
import requests
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
# from SA_Score import sascorer


def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


###################################################################################
#                        Molecular Environment Utilities                          #
###################################################################################


# TODO change to multiple valence
def atom_valences(atom_types):
    """Creates a list of valences corresponding to atom_types.
    Note that this is not a count of valence electrons, but a count of the
    maximum number of bonds each element will make. For example, passing
    atom_types ['C', 'H', 'O'] will return [4, 1, 2].
    Args:
    atom_types: List of string atom types, e.g. ['C', 'H', 'O'].
    Returns:
    List of integer atom valences.
    """
    periodic_table = Chem.GetPeriodicTable()
    return [
        min(list(periodic_table.GetValenceList(atom_type)))
        for atom_type in atom_types
    ]


def get_scaffold(mol):
    """Computes the Bemis-Murcko scaffold for a molecule.
    Args:
    mol: RDKit Mol.
    Returns:
    String scaffold SMILES.
    """
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=True)


def contains_scaffold(mol, scaffold):
    """Returns whether mol contains the given scaffold.
    NOTE: This is more advanced than simply computing scaffold equality (i.e.
    scaffold(mol_a) == scaffold(mol_b)). This method allows the target scaffold to
    be a subset of the (possibly larger) scaffold in mol.
    Args:
    mol: RDKit Mol.
    scaffold: String scaffold SMILES.
    Returns:
    Boolean whether scaffold is found in mol.
    """
    pattern = Chem.MolFromSmiles(scaffold)
    matches = mol.GetSubstructMatches(pattern)
    return bool(matches)


def get_largest_ring_size(molecule):
    """Calculates the largest ring size in the molecule.
    Refactored from
    https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
    Args:
    molecule: Chem.Mol. A molecule.
    Returns:
    Integer. The largest ring size.
    """
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


# def penalized_logp(molecule):
#     """Calculates the penalized logP of a molecule.
#     Refactored from
#     https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
#     See Junction Tree Variational Autoencoder for Molecular Graph Generation
#     https://arxiv.org/pdf/1802.04364.pdf
#     Section 3.2
#     Penalized logP is defined as:
#     y(m) = logP(m) - SA(m) - cycle(m)
#     y(m) is the penalized logP,
#     logP(m) is the logP of a molecule,
#     SA(m) is the synthetic accessibility score,
#     cycle(m) is the largest ring size minus by six in the molecule.
#     Args:
#     molecule: Chem.Mol. A molecule.
#     Returns:
#     Float. The penalized logP value.
#     """
#     log_p = Descriptors.MolLogP(molecule)
#     sas_score = sascorer.calculateScore(molecule)
#     largest_ring_size = get_largest_ring_size(molecule)
#     cycle_score = max(largest_ring_size - 6, 0)
#     return log_p - sas_score - cycle_score


# def get_sa_score(molecule):
#     """
#     Get the SA score of the molecule
#     :param molecules:
#     :return:
#     """
#     if isinstance(molecule, str):
#         molecule = Chem.MolFromSmiles(molecule)

#     assert molecule is not None

#     return sascorer.calculateScore(molecule)


# def get_sc_score(molecule):
#     """
#     Get the SC score of the molecule
#     :param molecule:
#     :return:
#     """
#     if isinstance(molecule, str):
#         molecule = Chem.MolFromSmiles(molecule)

#     assert molecule is not None

#     return scscorer.apply(scscorer.smi_to_fp(Chem.MolToSmiles(molecule)))


def get_buyability(molecule):
    """
    Check if the molecule is buyable
    :param molecule: inquiry molecule in format of SMILES
    :return:
    """
    path_to_buyable_data = 'data/pricer_using_reaxys_v2-chemicals_and_reaxys_v2-buyables.pkl'
    with open(path_to_buyable_data, 'rb') as f:
        buyable_molecules = pickle.load(f)

    try:
        return buyable_molecules[molecule]
    except:
        return 0


def get_synthesizability(molecule):
    # Check if the molecule is buyable first
    buyable = get_buyability(molecule)
    if buyable:
        return 1.0
    else:
        # If not buyable, then call the tree builder oracle
        HOST = 'http://askcos3.mit.edu'
        params = {
            'smiles': molecule,  # required
            # optional with defaults shown
            'max_depth': 5,
            'max_branching': 25,
            'expansion_time': 60,
            'max_ppg': 100,
            'template_count': 1000,
            'max_cum_prob': 0.999,
            'chemical_property_logic': 'none',
            'max_chemprop_c': 0,
            'max_chemprop_n': 0,
            'max_chemprop_o': 0,
            'max_chemprop_h': 0,
            'chemical_popularity_logic': 'none',
            'min_chempop_reactants': 5,
            'min_chempop_products': 5,
            'filter_threshold': 0.1,

            'return_first': 'true'  # default is false
        }

        for _ in range(15):
            resp = requests.get(HOST + '/api/treebuilder/', params=params)
            if 'error' not in resp.json().keys():
                break

        if 'error' not in resp.json().keys() or len(resp.json()['trees']) == 0:
            # No retrosynthetic pathway is found
            sa_score = sascorer.calculateScore(molecule)
            return sa_gaussian_wrapper(sa_score)
        else:
            # Retrosynthetic pathway is found
            return synthesizability_wrapper(resp.json())


def synthesizability_wrapper(json):
    num_path, status, depth, p_score, synthesizability, d_p = tree_analysis(json)
    return d_p


def sa_gaussian_wrapper(x, mu=3.2356, sigma=1.0156):
    if x < mu:
        return 1
    else:
        return np.exp(- (x - mu)**2 / (2 * sigma ** 2))


def sc_gaussian_wrapper(x, mu=2.9308, sigma=0.1803):
    if x < mu:
        return 1
    else:
        return np.exp(- (x - mu)**2 / (2 * sigma ** 2))


def smi_wrapper(x, a=0.1, b=70):
    return 1 - 1 / (1 + np.exp(- a * (x - b)))


def tree_analysis(current):
    """
    Analise the result of tree builder
    Calculate: 1. Number of steps 2. \Pi plausibility 3. If find a path
    In case of celery error, all values are -1

    return:
        num_path = number of paths found
        status: Same as implemented in ASKCOS one
        num_step: number of steps
        p_score: \Pi plausibility
        synthesizability: binary code
    """
    if 'error' in current.keys():
        return -1, {}, -1, -1, -1

    num_path = len(current['trees'])
    if num_path != 0:
        current = [current['trees'][0]]
    else:
        current = []
    depth = 0
    p_score = 1
    status = {0: 1}
    while True:
        num_child = 0
        depth += 0.5
        temp = []
        for i, item in enumerate(current):
            num_child += len(item['children'])
            temp = temp + item['children']
        if num_child == 0:
            break

        if depth % 1 != 0:
            for sth in temp:
                p_score = p_score * sth['plausibility']

        status[depth] = num_child
        current = temp

    if len(status) > 1:
        synthesizability = 1
    else:
        synthesizability = 0
    return num_path, status, int(depth - 0.5), p_score * synthesizability, synthesizability, depth * (1 - 0.5 * p_score)


def get_hparams(path_to_conifg=None):
    """Function to read hyper parameters"""
    # Default setting
    hparams = {
        'atom_types': ['C', 'O', 'N'],
        'max_steps_per_episode': 40,
        'allow_removal': True,
        'allow_no_modification': True,
        'allow_bonds_between_rings': False,
        'allowed_ring_sizes': [3, 4, 5, 6],
        'replay_buffer_size': 1000000,
        'learning_rate': 1e-4,
        'learning_rate_decay_steps': 10000,
        'learning_rate_decay_rate': 0.8,
        'num_episodes': 5000,
        'batch_size': 64,
        'learning_frequency': 4,
        'update_frequency': 20,
        'grad_clipping': 10.0,
        'gamma': 0.9,
        'double_q': True,
        'num_bootstrap_heads': 12,
        'prioritized': False,
        'prioritized_alpha': 0.6,
        'prioritized_beta': 0.4,
        'prioritized_epsilon': 1e-6,
        'fingerprint_radius': 3,
        'fingerprint_length': 2048,
        'dense_layers': [1024, 512, 128, 32],
        'activation': 'ReLU',
        'optimizer': 'Adam',
        'batch_norm': True,
        'save_frequency': 1000,
        'max_num_checkpoints': 100,
        'discount_factor': 0.7,
        'dropout': 0.5
    }
    if path_to_conifg is not None:
        with open(path_to_conifg, 'r') as f:
            hparams.update(json.load(f))
    return hparams


def get_fingerprint_with_stpes_left(smiles, steps_left, hparams):
    fingerprint = get_fingerprint(smiles, hparams)
    return np.append(fingerprint, steps_left)


def get_fingerprint(mol, hparams):
    """Get Morgan fingerprint for a specific molecule"""

    length = hparams['fingerprint_length']
    radius = hparams['fingerprint_radius']

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    if mol is None:
        return np.zeros((length, ))

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


###################################################################################
#                     Reinforcement Learning Utilities                            #
###################################################################################


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        return obses_t, actions, rewards, obses_tp1, dones

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meaning that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
