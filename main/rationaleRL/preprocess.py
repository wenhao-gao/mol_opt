import math, random, sys
import pickle
import argparse
import torch
import numpy
import rdkit

from collections import deque
from multiprocessing import Pool
from rdkit import Chem
from fuseprop import MolGraph, common_atom_vocab, random_subgraph, dual_random_subgraph

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c,d = tensors
    b = [convert(x) for x in b]
    return a, b, c, d

def get_ratio():
    ratio = random.gauss(0.5, 0.07) 
    ratio = min(ratio, 0.6) # 0.7
    ratio = max(ratio, 0.3)
    return ratio

def get_natoms(total_natoms):
    natoms = total_natoms // 2 - 4
    return min(natoms, 20)

def tensorize(smiles_batch):
    ratio = get_ratio()
    init_atoms = [random_subgraph(Chem.MolFromSmiles(x), ratio) for x in smiles_batch]
    mol_batch = [MolGraph(x, atoms) for x, atoms in zip(smiles_batch, init_atoms)]
    mol_batch = [x for x in mol_batch if len(x.root_atoms) > 0]
    x = MolGraph.tensorize(mol_batch, common_atom_vocab)
    return to_numpy(x)

def zero_tensorize(smiles_batch):
    init_atoms = [set([0]) for _ in smiles_batch]
    mol_batch = [MolGraph(x, atoms) for x, atoms in zip(smiles_batch, init_atoms)]
    mol_batch = [x for x in mol_batch if len(x.root_atoms) > 0]
    x = MolGraph.tensorize(mol_batch, common_atom_vocab)
    return to_numpy(x)

def dual_tensorize(smiles_batch):
    init_atoms = []
    for x in smiles_batch:
        mol = Chem.MolFromSmiles(x)
        selected_atoms = dual_random_subgraph(mol, ratio=0.6)
        init_atoms.append(selected_atoms)

    mol_batch = [MolGraph(x, atoms) for x, atoms in zip(smiles_batch, init_atoms)]
    mol_batch = [x for x in mol_batch if len(x.root_atoms) > 0]
    x = MolGraph.tensorize(mol_batch, common_atom_vocab)
    return to_numpy(x)


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--mode', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--ncpu', type=int, default=15)
    args = parser.parse_args()
    print(args)

    pool = Pool(args.ncpu) 
    random.seed(1)

    #dataset contains single molecules
    with open(args.train) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    random.shuffle(data)

    batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]

    if args.mode == 1:
        all_data = pool.map(tensorize, batches)
    elif args.mode == 2:
        all_data = pool.map(dual_tensorize, batches)
    elif args.mode == 0:
        all_data = pool.map(zero_tensorize, batches)
    else:
        raise ValueError('mode not supported')

    num_splits = len(all_data) // 1000
    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

