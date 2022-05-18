import sys, os
from tqdm import tqdm
import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from util.smiles.dataset import load_dataset
from util.smiles.char_dict import SmilesCharDictionary

if __name__ == "__main__":
    char_dict = SmilesCharDictionary(dataset="zinc", max_smi_len=81)
    dataset = load_dataset(char_dict=char_dict, smi_path="./resource/data/zinc/full.txt")
    log_ps, sa_scores, atomring_cycle_scores, cyclebasis_cycle_scores = [], [], [], []
    for smi in tqdm(dataset):
        mol = Chem.MolFromSmiles(smi)

        log_p = Descriptors.MolLogP(mol)
        sa_score = sascorer.calculateScore(mol)

        cycle_list = mol.GetRingInfo().AtomRings()
        largest_ring_size = max([len(j) for j in cycle_list]) if cycle_list else 0
        atomring_cycle_score = max(largest_ring_size - 6, 0)

        cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
        largest_ring_size = max([len(j) for j in cycle_list]) if cycle_list else 0
        cyclebasis_cycle_score = max(largest_ring_size - 6, 0)

        log_ps.append(log_p)
        sa_scores.append(sa_score)
        atomring_cycle_scores.append(atomring_cycle_score)
        cyclebasis_cycle_scores.append(cyclebasis_cycle_score)

    print(f"LogP stats: {np.mean(log_ps)}, {np.std(log_ps)}")
    print(f"SA stats: {np.mean(sa_scores)}, {np.std(sa_scores)}")
    print(f"AtomRing stats: {np.mean(atomring_cycle_scores)}, {np.std(atomring_cycle_scores)}")
    print(
        f"CycleBasis stats: {np.mean(cyclebasis_cycle_scores)}, {np.std(cyclebasis_cycle_scores)}"
    )
