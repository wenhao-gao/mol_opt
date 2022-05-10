import sys
from rdkit import Chem
from fuseprop import enum_subgraph, extract_subgraph 

ratio_list = [0.3, 0.4, 0.5, 0.6, 0.7]

next(sys.stdin)
for line in sys.stdin:
    smiles = line.strip("\r\n ").split(',')[0]
    mol = Chem.MolFromSmiles(smiles)
    selections = enum_subgraph(mol, ratio_list)

    res = []
    for selected_atoms in selections:
        subgraph, _ = extract_subgraph(smiles, selected_atoms) 
        if subgraph is not None: 
            res.append(subgraph)

    for subgraph in set(res):
        print(smiles, subgraph)
