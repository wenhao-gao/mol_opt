import sys
import numpy as np
from argparse import ArgumentParser
from rdkit import Chem
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument('--sparsity', type=float, default=20)
parser.add_argument('--prop', type=float, default=0.8)
args = parser.parse_args()

stats = []
data = defaultdict(list)

next(sys.stdin)

for line in sys.stdin:
    line = line.strip("\r\n ")
    items = line.split()
    prop = float(items[-1])
    if prop < args.prop: continue

    mol = Chem.MolFromSmiles(items[0])
    rmol = Chem.MolFromSmiles(items[1])

    sparsity = rmol.GetNumAtoms() #/ mol.GetNumAtoms()
    if sparsity <= args.sparsity:
        data[items[0]].append( (items[1], prop, sparsity) )

for mol,cands in data.items():
    rationale, prop, sparsity = min(cands, key=lambda x:x[2])
    print(mol, rationale, prop, sparsity)
    stats.append(sparsity)

stats = np.array(stats)
print(np.mean(stats), np.std(stats), file=sys.stderr)
