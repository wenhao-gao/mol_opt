import sys
import argparse
import math
from rdkit import Chem
from functools import partial
from multiprocessing import Pool
from fuseprop import find_clusters, extract_subgraph
from properties import get_scoring_function

MIN_ATOMS = 15
C_PUCT = 10

class MCTSNode():

    def __init__(self, smiles, atoms, W=0, N=0, P=0):
        self.smiles = smiles
        self.atoms = set(atoms)
        self.children = []
        self.W = W
        self.N = N
        self.P = P

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return C_PUCT * self.P * math.sqrt(n) / (1 + self.N)


def mcts_rollout(node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function):
    #print('cur_node', node.smiles, node.P, node.N, node.W)
    cur_atoms = node.atoms
    if len(cur_atoms) <= MIN_ATOMS:
        return node.P

    # Expand if this node has never been visited
    if len(node.children) == 0:
        cur_cls = set( [i for i,x in enumerate(clusters) if x <= cur_atoms] )
        for i in cur_cls:
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]
            if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
                #print('new_smiles', node.smiles, '->', new_smiles)
                if new_smiles in state_map:
                    new_node = state_map[new_smiles] # merge identical states
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0: return node.P  # cannot find leaves

        scores = scoring_function([x.smiles for x in node.children])
        for child, score in zip(node.children, scores):
            child.P = score
        
    sum_count = sum([c.N for c in node.children])
    selected_node = max(node.children, key=lambda x : x.Q() + x.U(sum_count))
    v = mcts_rollout(selected_node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function)
    selected_node.W += v
    selected_node.N += 1
    return v

def mcts(smiles, scoring_function, n_rollout, max_atoms, prop_delta): 
    mol = Chem.MolFromSmiles(smiles)
    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i,cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - set([i])
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])
    
    root = MCTSNode( smiles, set(range(mol.GetNumAtoms())) ) 
    state_map = {smiles : root}
    for _ in range(n_rollout):
        mcts_rollout(root, state_map, smiles, clusters, atom_cls, nei_cls, scoring_function)

    rationales = [node for _,node in state_map.items() if len(node.atoms) <= max_atoms and node.P >= prop_delta]
    return smiles, rationales


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--prop', required=True)
    parser.add_argument('--rollout', type=int, default=20)
    parser.add_argument('--c_puct', type=float, default=10)
    parser.add_argument('--max_atoms', type=int, default=20)
    parser.add_argument('--min_atoms', type=int, default=15)
    parser.add_argument('--prop_delta', type=float, default=0.5)
    parser.add_argument('--ncand', type=int, default=2)
    parser.add_argument('--ncpu', type=int, default=15)
    args = parser.parse_args()

    scoring_function = get_scoring_function(args.prop)
    scoring_function.clf.n_jobs = 1

    C_PUCT = args.c_puct
    MIN_ATOMS = args.min_atoms

    with open(args.data) as f:
        next(f)
        data = [line.split(',')[0] for line in f]

    work_func = partial(mcts, scoring_function=scoring_function, 
                              n_rollout=args.rollout, 
                              max_atoms=args.max_atoms, 
                              prop_delta=args.prop_delta)

    pool = Pool(args.ncpu)
    results = pool.map(work_func, data)

    rset = set()
    for orig_smiles, rationales in results:
        rationales = sorted(rationales, key=lambda x:len(x.atoms))
        for x in rationales[:args.ncand]:
            if x.smiles not in rset:
                print(orig_smiles, x.smiles, len(x.atoms), x.P)
                rset.add(x.smiles)

