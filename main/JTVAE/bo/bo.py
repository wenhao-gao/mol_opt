import torch
import torch.nn as nn
from torch.autograd import Variable
from optparse import OptionParser

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
from tqdm import tqdm 
import pickle
import scipy.stats as sps
from random import shuffle 
import numpy as np, networkx as nx
from fast_jtnn import *
from sparse_gp import SparseGP

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

from tdc import Oracle, Evaluator
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
qed = Oracle(name = 'qed')
from sa import sa
def oracle(smiles):
    scores = [qed(smiles), sa(smiles), jnk(smiles), gsk(smiles)]
    return np.mean(scores)

def score_mol(smiles, score_fn, known_value_dict):
    if smiles not in known_value_dict:
        known_value_dict[smiles] = score_fn(smiles)
    return known_value_dict[smiles]

max_func_calls = 250 
f_cache = dict() 
train_size = int(max_func_calls*0.5)

parser = OptionParser()
parser.add_option("-a", "--data", dest="data_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-t", "--deptht", dest="depthT", default=20)
parser.add_option("-g", "--depthg", dest="depthG", default=3)
parser.add_option("-o", "--save_dir", dest="save_dir")
parser.add_option("-r", "--seed", dest="random_seed", default=1)
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

random_seed = int(opts.random_seed)
np.random.seed(random_seed)

with open(opts.data_path) as f:
    smiles = f.readlines()
smiles = [s.strip() for s in smiles]
shuffle(smiles)
smiles = smiles[:train_size]

batch_size = 100
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depthT = int(opts.depthT)
depthG = int(opts.depthG)

model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()


print('generate latent variable')
latent_points = []
for i in tqdm(range(0, len(smiles), batch_size)):
    batch = smiles[i:i+batch_size]
    mol_vec = model.encode_latent_mean(batch)
    latent_points.append(mol_vec.data.cpu().numpy())

# output  X: "N X d" latent embedding;  y: label "N X 1"
X = np.vstack(latent_points) 
y = np.array([-score_mol(s, oracle, f_cache) for s in smiles]).reshape((-1,1)) 

#### permutation & split 
n = X.shape[0]
permutation = np.random.choice(n, n, replace = False)
X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]
y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]


M,K = 100, 30
print('fit GP')
for iteration in tqdm(range(10)):
    np.random.seed(iteration * random_seed) 
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    print('begin train sparse GP')
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 100, learning_rate = 0.001)

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    print('Test RMSE: ', error)
    print('Test ll: ', testll)

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print('Train RMSE: ', error)
    print('Train ll: ', trainll)

    # We pick the next K inputs
    print(np.min(X_train,0)[:10], np.max(X_train, 0)[:10])
    next_inputs = sgp.batched_greedy_ei(K, np.min(X_train, 0), np.max(X_train, 0))
    valid_smiles = []
    new_features = []
    for i in range(K):
        all_vec = next_inputs[i].reshape((1,-1))
        tree_vec,mol_vec = np.hsplit(all_vec, 2)
        tree_vec = create_var(torch.from_numpy(tree_vec).float())
        mol_vec = create_var(torch.from_numpy(mol_vec).float())
        s = model.decode(tree_vec, mol_vec, prob_decode=False)
        if s is not None: 
            valid_smiles.append(s)
            new_features.append(all_vec)

    print(len(valid_smiles), "molecules are found")
    valid_smiles = valid_smiles[:50]
    new_features = next_inputs[:50]
    new_features = np.vstack(new_features)

    scores = [-score_mol(s, oracle, f_cache) for s in valid_smiles]
    print(valid_smiles)
    print(scores)

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
    if len(f_cache) > max_func_calls: 
        print("max oracle hits, exit")
        break 



# Evaluate 
new_score_tuples = [(v, k) for k, v in f_cache.items()]  # scores of new molecules
new_score_tuples.sort(reverse=True)
top100_mols = [(k, v) for (v, k) in new_score_tuples[:100]]
print(top100_mols)
diversity = Evaluator(name = 'Diversity')
div = diversity([t[0] for t in top100_mols])
output = dict(
        top_mols=top100_mols,
        AST=np.average([t[1] for t in top100_mols]),
        diversity=div,
        all_func_evals=dict(f_cache),
)
# with open(args.output_file, "w") as f:
#     json.dump(output, f, indent=4)





