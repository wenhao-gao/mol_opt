# code modified from https://github.com/haroldsultan/MCTS/blob/master/mcts.py
import argparse
import hashlib
import math
import os
import random
import yaml
from time import time

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
rdBase.DisableLog('rdApp.error')
from tdc import Oracle

from stats import Stats, get_stats_from_pickle
from main.optimizer import BaseOptimizer


def run_rxn(rxn_smarts, mol):
    new_mol_list = []
    patt = rxn_smarts.split('>>')[0]
    # work on a copy so an un-kekulized version is returned
    # if the molecule is not changed
    mol_copy = Chem.Mol(mol)
    try:
        Chem.Kekulize(mol_copy)
    except ValueError:
        pass
    if mol_copy.HasSubstructMatch(Chem.MolFromSmarts(patt)):
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        new_mols = rxn.RunReactants((mol_copy,))
        for new_mol in new_mols:
            try:
                Chem.SanitizeMol(new_mol[0])
                new_mol_list.append(new_mol[0])
            except ValueError:
                pass
        if len(new_mol_list) > 0:
            new_mol = random.choice(new_mol_list)
            return new_mol
        else:
            return mol
    else:
        return mol


def add_atom(rdkit_mol, stats: Stats):
    old_mol = Chem.Mol(rdkit_mol)
    if np.random.random() < 0.63:  # probability of adding ring atom
        rxn_smarts = np.random.choice(stats.rxn_smarts_ring_list, p=stats.p_ring)
        if not rdkit_mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4,r5]')) \
                or AllChem.CalcNumAliphaticRings(rdkit_mol) == 0:
            rxn_smarts = np.random.choice(stats.rxn_smarts_make_ring, p=stats.p_ring)
            if np.random.random() < 0.036:  # probability of starting a fused ring
                rxn_smarts = rxn_smarts.replace("!", "")
    else:
        if rdkit_mol.HasSubstructMatch(Chem.MolFromSmarts('[*]1=[*]-[*]=[*]-1')):
            rxn_smarts = '[r4:1][r4:2]>>[*:1]C[*:2]'
        else:
            rxn_smarts = np.random.choice(stats.rxn_smarts_list, p=stats.p)

    rdkit_mol = run_rxn(rxn_smarts, rdkit_mol)
    if valences_not_too_large(rdkit_mol):
        return rdkit_mol
    else:
        return old_mol


def expand_small_rings(rdkit_mol):
    Chem.Kekulize(rdkit_mol, clearAromaticFlags=True)
    rxn_smarts = '[*;r3,r4;!R2:1][*;r3,r4:2]>>[*:1]C[*:2]'
    while rdkit_mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]')):
        rdkit_mol = run_rxn(rxn_smarts, rdkit_mol)
    return rdkit_mol


def valences_not_too_large(rdkit_mol):
    valence_dict = {5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 5, 16: 6, 17: 1, 34: 2, 35: 1, 53: 1}
    atomicNumList = [a.GetAtomicNum() for a in rdkit_mol.GetAtoms()]
    valences = [valence_dict[atomic_num] for atomic_num in atomicNumList]
    BO = Chem.GetAdjacencyMatrix(rdkit_mol, useBO=True)
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False
    return True


class State:

    def __init__(self, oracle, mol, smiles, max_atoms, max_children, stats: Stats, seed):
        self.mol = mol
        self.turn = max_atoms
        self.smiles = smiles
        self.oracle = oracle
        self.score = self.oracle(self.smiles)
        self.max_children = max_children
        self.stats = stats
        self.seed = seed

    def next_state(self):
        smiles = self.smiles
        # TODO: this seems dodgy...
        for i in range(100):
            mol = add_atom(self.mol, self.stats)
            smiles = Chem.MolToSmiles(mol)
            if smiles != self.smiles:
                break
        next_state = State(oracle=self.oracle,
                           mol=mol,
                           smiles=smiles,
                           max_atoms=self.turn - 1,
                           max_children=self.max_children,
                           stats=self.stats,
                           seed=self.seed)
        return next_state

    def terminal(self):
        target_size = self.stats.size_std_dev * np.random.randn() + self.stats.average_size
        if self.mol is None:
            num_atoms = 0
        else:
            num_atoms = self.mol.GetNumAtoms()

        if self.turn == 0 or num_atoms > target_size:
            self.mol = expand_small_rings(self.mol)
            self.smiles = Chem.MolToSmiles(self.mol)
            # print('terminal!', self.score, self.best_score, self.smiles)
            return True

        return False

    def reward(self, best_state):
        if best_state is None or self.score > best_state.score:
            # best_state = self
            return 1.0
        else:
            return 0.0

    def __hash__(self):
        return int(hashlib.md5(str(self.smiles).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        return f'Value: {self.value} | Moves: {self.moves} | Turn {self.turn}'


class Node:
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        if len(self.children) == self.state.max_children:
            return True
        return False

    def __repr__(self):
        s = str(self.state.smiles)
        return s


def tree_policy(node, exploration_coefficient=(1/math.sqrt(2.0))):
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.fully_expanded() and not node.state.oracle.finish:
        node = best_child(node, exploration_coefficient)

    if node.state.terminal():
        return node
    else:
        node = expand_all(node)
        return node


def expand_all(node):
    lcount = 0
    while not node.fully_expanded() and lcount < node.state.max_children and not node.state.oracle.finish:
        lcount += 1
        node = expand(node)
    return node


def expand(node):
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_state()
    lcount = 0
    while new_state in tried_children and lcount < new_state.max_children:
        lcount += 1
        new_state = node.state.next_state()
    node.add_child(new_state)
    return node


# current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def best_child(node, exploration_coefficient):
    bestscore = 0.0
    bestchildren = []
    for c in node.children:
        exploit = c.reward / c.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
        score = exploit + exploration_coefficient * explore
        # print(score, node.state.terminal(), node.state.smiles, bestscore)
        if score == bestscore:
            bestchildren.append(c)
        if score >= bestscore:
            bestchildren = [c]
            bestscore = score
    if len(bestchildren) == 0:
        print("OOPS: no best child found, probably fatal")
        return node
    return random.choice(bestchildren)


def default_policy(state, best_state):
    while not state.terminal() and not state.oracle.finish:
        state = state.next_state()
    reward = state.reward(best_state)
    if reward == 1:
        best_state = state
    return reward, best_state


def backup(node, reward):
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return


class Graph_MCTS_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "graph_mcts"

    def _optimize(self, oracle, config):
        
        self.oracle.assign_evaluator(oracle)

        init_mol = Chem.MolFromSmiles(config["init_smiles"])
        stats = get_stats_from_pickle(self.args.pickle_directory)

        # evolution: go go go!!
        while True:

            # UCB Tree Search
            if self.finish:
                break
            tmp_seed = int(time())
            np.random.seed(tmp_seed)
            best_state = None
            root_node = Node(State(oracle=self.oracle,
                                mol=init_mol,
                                smiles=config["init_smiles"],
                                max_atoms=config["max_atoms"],
                                max_children=config["max_children"],
                                stats=stats, 
                                seed=tmp_seed))

            for _ in range(int(config["num_sims"])):
                front = tree_policy(root_node, exploration_coefficient=config["exploration_coefficient"])
                if self.finish:
                    break
                for child in front.children:
                    reward, best_state = default_policy(child.state, best_state)
                    backup(child, reward)
                    if self.finish:
                        break

