import torch
import torch.nn as nn
from torch.autograd import Variable
from optparse import OptionParser

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import sascorer
from tqdm import tqdm 


import numpy as np  
from fast_jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-a", "--data", dest="data_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
# parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-t", "--deptht", dest="depthT", default=20)
parser.add_option("-g", "--depthg", dest="depthG", default=3)
# parser.add_option('--depthT', type=int, default=20)
# parser.add_option('--depthG', type=int, default=3)
opts,args = parser.parse_args()

with open(opts.data_path) as f:
    smiles = f.readlines()

for i in tqdm(range(len(smiles))):
    smiles[ i ] = smiles[ i ].strip()
smiles = smiles[:170]


vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

batch_size = 100
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
# depth = int(opts.depth)
depthT = int(opts.depthT)
depthG = int(opts.depthG)

model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

print('canonical')
smiles_rdkit = []
for i in tqdm(range(len(smiles))):
    smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[ i ]), isomericSmiles=True))


print('logp')
logP_values = []
for i in tqdm(range(len(smiles))):
    logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[ i ])))


print('sa score')
SA_scores = []
for i in tqdm(range(len(smiles))):
    SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[ i ])))

import networkx as nx

print('cycle_scores')
cycle_scores = []
for i in tqdm(range(len(smiles))):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[ i ]))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_scores.append(-cycle_length)

SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)


print('encode latent ')
latent_points = []
for i in tqdm(range(0, len(smiles), batch_size)):
    batch = smiles[i:i+batch_size]
    mol_vec = model.encode_latent_mean(batch)
    latent_points.append(mol_vec.data.cpu().numpy())

# We store the results
latent_points = np.vstack(latent_points)
np.savetxt('latent_features.txt', latent_points)

targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
np.savetxt('targets.txt', targets)
np.savetxt('logP_values.txt', np.array(logP_values))
np.savetxt('SA_scores.txt', np.array(SA_scores))
np.savetxt('cycle_scores.txt', np.array(cycle_scores))








