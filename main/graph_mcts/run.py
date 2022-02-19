import argparse
import hashlib
import json
import logging
import math
import os
import random
from time import time
from typing import List, Optional

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from tqdm import tqdm 

# from graph_mcts.stats import Stats, get_stats_from_pickle, get_stats_from_smiles
from stats import Stats, get_stats_from_pickle, get_stats_from_smiles

rdBase.DisableLog('rdApp.error')

from tdc import Oracle
from tdc import Evaluator
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
qed = Oracle(name = 'qed')
from sa import sa
def oracle(smiles):
    try:
        scores = [qed(smiles), sa(smiles), jnk(smiles), gsk(smiles)]
    except:
        # return -np.inf
        return -100.0 
    return np.mean(scores)

global f_cache 
f_cache = dict()
# global f_cache 


def score_mol(smiles, score_fn):
    # smiles = Chem.MolToSmiles(mol)
    global f_cache
    if smiles not in f_cache:
        f_cache[smiles] = score_fn(smiles)
    return f_cache[smiles]

from rdkit import Chem 
def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.
    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543
    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string
    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None




best_state = {}


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


# code modified from https://github.com/haroldsultan/MCTS/blob/master/mcts.py


SCALAR = 1 / math.sqrt(2.0)


class State:

    def __init__(self, scoring_function, mol, smiles, max_atoms, max_children, stats: Stats, seed):
        self.mol = mol
        self.turn = max_atoms
        self.smiles = smiles
        self.scoring_function = scoring_function
        # self.score = self.scoring_function.score(self.smiles)
        self.score = score_mol(self.smiles, scoring_function)
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

        next_state = State(scoring_function=self.scoring_function,
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

    def reward(self):
        global best_state

        if self.seed not in best_state or self.score > best_state[self.seed].score:
            best_state[self.seed] = self
            # print(self.seed, 'new best state', best_state[self.seed].score)

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


def uct_search(budget, root):
    for _ in range(int(budget)):
        front = tree_policy(root)
        for child in front.children:
            reward = default_policy(child.state)
            backup(child, reward)
    return best_child(root, 0)


def tree_policy(node):
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.fully_expanded():
        node = best_child(node, SCALAR)

    if node.state.terminal():
        return node
    else:
        node = expand_all(node)
        return node


def expand_all(node):
    lcount = 0
    while not node.fully_expanded() and lcount < node.state.max_children:
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
def best_child(node, scalar):
    bestscore = 0.0
    bestchildren = []
    for c in node.children:
        exploit = c.reward / c.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
        score = exploit + scalar * explore
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


def default_policy(state):
    while not state.terminal():
        state = state.next_state()
    return state.reward()


def backup(node, reward):
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return

def find_molecule(scoring_function,  mol, smiles, max_atoms, max_children, num_sims, stats):
    seed = int(time())
    np.random.seed(seed)
    root_node = Node(State(scoring_function=scoring_function,
                           mol=mol,
                           smiles=smiles,
                           max_atoms=max_atoms,
                           max_children=max_children,
                           stats=stats, 
                           seed=seed))
    uct_search(num_sims, root_node)

    return best_state[seed].score, best_state[seed].smiles


def sanitize(population):
    new_population = []
    smile_set = set()
    for mol in population:
        score, smile = mol
        if smile is not None and smile not in smile_set:
            smile_set.add(smile)
            new_population.append(mol)
    return new_population


def generate_optimized_molecules(scoring_function,
                                n_jobs, 
                                pickle_directory, 
                                num_sims,
                                max_children,
                                init_smiles,
                                max_atoms,
                                patience_max,
                                generations,
                                population_size,         
                                start_known_smiles, 
                                max_total_func_calls,) -> List[str]:

        pool = joblib.Parallel(n_jobs=n_jobs)
        init_mol = Chem.MolFromSmiles(init_smiles)
        # f_cache = dict(start_known_smiles)
        global f_cache 
        stats = get_stats_from_pickle(pickle_directory)

        # evolution: go go go!!
        t0 = time()
        patience = 0
        population = []
        old_score = 0

        for generation in tqdm(range(generations)):

            job = delayed(find_molecule)(scoring_function,
                                         init_mol,
                                         init_smiles,
                                         max_atoms,
                                         max_children,
                                         num_sims,
                                         stats)

            new_mols = pool(job for _ in range(population_size))

            # stats
            gen_time = time() - t0
            mol_sec = len(new_mols) / gen_time
            t0 = time()

            population += new_mols
            population = sanitize(population)

            population = sorted(population, key=lambda x: x[0], reverse=True)[:population_size]

            population_scores = [p[0] for p in population]
            new_score = sum(population_scores)

            #print([p[1] for p in population])
            print('population size:', len(population))

            # early stopping
            if new_score == old_score:
                patience += 1
                print(f'Failed to progress: {patience}')
                if patience >= patience_max:
                    print(f'No more patience, bailing...')
                    break
            else:
                patience = 0

            old_score = new_score

            print(f'{generation} | '
                  f'max: {np.max(population_scores):.3f} | '
                  f'avg: {np.mean(population_scores):.3f} | '
                  f'min: {np.min(population_scores):.3f} | '
                  f'std: {np.std(population_scores):.3f} | '
                  f'sum: {np.sum(population_scores):.3f} | '
                  f'{gen_time:.2f} sec/gen | '
                  f'{mol_sec:.2f} mol/sec | '
                  f'{len(f_cache):d} oracle calls')

            if len(f_cache) > max_total_func_calls:
                print("Max oracle calls hit, aborting")
                break

        # return f_cache

        # finally
        # return [p[1] for p in population]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics',
                        default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_jobs', type=int, default=1)  ## default -1 
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--population_size', type=int, default=10)
    parser.add_argument('--num_sims', type=int, default=2) #### default 40
    parser.add_argument('--max_children', type=int, default=2) ### default 25
    parser.add_argument('--max_atoms', type=int, default=4) ### default 60  
    parser.add_argument('--init_smiles', type=str, default='CC')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_func_calls', type=int, default=200)
    # parser.add_argument('--suite', default='v2') 
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    if args.pickle_directory is None:
        args.pickle_directory = os.path.dirname(os.path.realpath(__file__))

    np.random.seed(args.seed)


    # save command line args
    with open(os.path.join(args.output_dir, 'goal_directed_params.json'), 'w') as jf:
        json.dump(vars(args), jf, sort_keys=True, indent=4)


    generate_optimized_molecules(
        scoring_function = oracle,
        pickle_directory = args.pickle_directory, 
        n_jobs = args.n_jobs, 
        num_sims=args.num_sims,
        max_children=args.max_children,
        init_smiles=args.init_smiles,
        max_atoms=args.max_atoms,
        patience_max=args.patience,
        generations=args.generations,
        population_size=args.population_size,         
        start_known_smiles = dict(),
        max_total_func_calls=args.max_func_calls, 
    )

    all_func_evals = f_cache

    # Evaluate 
    new_score_tuples = [(v, k) for k, v in all_func_evals.items() if k is not None and k!='']  # scores of new molecules
    new_score_tuples.sort(reverse=True,key=lambda x:x[0])
    top100_mols = [(k, v) for (v, k) in new_score_tuples[:100]]
    diversity = Evaluator(name = 'Diversity')
    div = diversity([t[0] for t in top100_mols])
    output = dict(
        top_mols=top100_mols,
        AST=np.average([t[1] for t in top100_mols]),
        diversity=div,
        all_func_evals=dict(all_func_evals),
    )
    print(output)






if __name__ == "__main__":
    main()






