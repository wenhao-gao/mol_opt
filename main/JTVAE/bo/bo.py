import os, pickle, torch, random, argparse
import yaml
import numpy as np 
from tqdm import tqdm 
torch.manual_seed(1)
np.random.seed(2)
random.seed(1)
from tdc import Oracle
import sys
# sys.path.append('../..')
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer



# import torch
import torch.nn as nn
from torch.autograd import Variable
# from optparse import OptionParser

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import scipy.stats as sps
from random import shuffle 
import networkx as nx
from fast_jtnn import *
from sparse_gp import SparseGP

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)


"""
python bo.py --vocab ../data/moses/vocab.txt --save_dir results --data ../data/moses/train_validity_5k.txt --hidden 450 \
             --latent 56 --model ../fast_molvae/vae_model/model.iter-5000 
"""


class JTVAEBOoptimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "JTVAE_BO"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        train_size = 100

        vocab = [x.strip("\r\n ") for x in open(config['vocab_path'])] 
        vocab = Vocab(vocab)

        random_seed = int(config['random_seed'])
        np.random.seed(random_seed)

        with open(config['data_path']) as f:
            smiles = f.readlines()
        smiles = [s.strip() for s in smiles]
        shuffle(smiles)
        smiles = smiles[:train_size]

        batch_size = 100
        hidden_size = int(config['hidden_size'])
        latent_size = int(config['latent_size'])
        depthT = int(config['depthT'])
        depthG = int(config['depthG'])

        model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
        model.load_state_dict(torch.load(config['model_path']))
        model = model.cuda()


        print('generate latent variable')
        latent_points = []
        for i in tqdm(range(0, len(smiles), batch_size)):
            batch = smiles[i:i+batch_size]
            mol_vec = model.encode_latent_mean(batch)
            latent_points.append(mol_vec.data.cpu().numpy())

        # output  X: "N X d" latent embedding;  y: label "N X 1"
        X = np.vstack(latent_points) 
        y = np.array([-self.oracle(s) for s in smiles]).reshape((-1,1)) 

        #### permutation & split 
        n = X.shape[0]
        permutation = np.random.choice(n, n, replace = False)
        X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
        X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]
        y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
        y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

        X_train = X_train[:200]
        X_test = X_test[:50]
        y_train = y_train[:200]
        y_test = y_test[:50]


        M,K = 50, 25
        print('fit GP')
        for iteration in tqdm(range(10)):
            np.random.seed(iteration * random_seed) 
            sgp = SparseGP(X_train, 0 * X_train, y_train, M)
            print('train sparse GP')
            sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 10, learning_rate = 0.001)
            # print('finish train sparse GP')

            pred, uncert = sgp.predict(X_test, 0 * X_test)
            error = np.sqrt(np.mean((pred - y_test)**2))
            testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
            # print('Test RMSE: ', error)
            # print('Test ll: ', testll)

            pred, uncert = sgp.predict(X_train, 0 * X_train)
            error = np.sqrt(np.mean((pred - y_train)**2))
            trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
            # print('Train RMSE: ', error)
            # print('Train ll: ', trainll)

            # We pick the next K inputs
            # print(np.min(X_train,0)[:10], np.max(X_train, 0)[:10])
            print('batched_greedy_ei')
            next_inputs = sgp.batched_greedy_ei(K, np.min(X_train, 0), np.max(X_train, 0))
            # print('end batched_greedy_ei')
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

            scores = [-self.oracle(s) for s in valid_smiles]
            # print(valid_smiles, scores)

            if len(new_features) > 0:
                X_train = np.concatenate([ X_train, new_features ], 0)
                y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
            X_train = X_train[-50:]
            y_train = y_train[-50:]
            if self.oracle.finish: 
                print("max oracle hits, exit")
                break 







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=500)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

    path_here = os.path.dirname(os.path.realpath(__file__))

    if args.output_dir is None:
        args.output_dir = os.path.join(path_here, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    for oracle_name in args.oracles:

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_here, args.config_default)))

        if args.task == "tune":
            try:
                config_tune = yaml.safe_load(open(args.config_tune))
            except:
                config_tune = yaml.safe_load(open(os.path.join(path_here, args.config_tune)))

        oracle = Oracle(name = oracle_name)
        optimizer = JTVAEBOoptimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main() 







