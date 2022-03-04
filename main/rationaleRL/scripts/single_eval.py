import sys
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--ref_path', required=True)
args = parser.parse_args()

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

pred_data = [line.split()[1:] for line in sys.stdin]
pred_mols = [mol for mol,x in pred_data if float(x) >= 0.5]

fraction_actives = len(pred_mols) / len(pred_data)
print('fraction actives:', fraction_actives)

with open(args.ref_path) as f:
    next(f)
    true_mols = [line.split(',')[0] for line in f]
print('number of active reference', len(true_mols))

true_mols = [Chem.MolFromSmiles(s) for s in true_mols]
true_mols = [x for x in true_mols if x is not None]
true_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in true_mols]

pred_mols = [Chem.MolFromSmiles(s) for s in pred_mols]
pred_mols = [x for x in pred_mols if x is not None]
pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_mols]

fraction_similar = 0
for i in range(len(pred_fps)):
    sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)
    if max(sims) >= 0.4:
        fraction_similar += 1

print('novelty:', 1 - fraction_similar / len(pred_mols))

similarity = 0
for i in range(len(pred_fps)):
    sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
    similarity += sum(sims)

n = len(pred_fps) 
n_pairs = n * (n - 1) / 2
diversity = 1 - similarity / n_pairs
print('diversity:', diversity)

