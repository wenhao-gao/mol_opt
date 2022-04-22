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
from random import shuffle, choice
import networkx as nx
from fast_jtnn import *
from sparse_gp import SparseGP

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

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

        ## 0. load vae model 
        vocab = [x.strip("\r\n ") for x in open(config['vocab_path'])] 
        vocab = Vocab(vocab)
        hidden_size = int(config['hidden_size'])
        latent_size = int(config['latent_size'])
        depthT = int(config['depthT'])
        depthG = int(config['depthG'])
        model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
        model.load_state_dict(torch.load(config['model_path']))
        vae_model = model.cuda()
        # vae_model = torch.load(config['save_model'])



        ''' original version
        
        latent_points = []
        for i in tqdm(range(0, len(smiles), batch_size)):
            batch = smiles[i:i+batch_size]
            mol_vec = model.encode_latent_mean(batch)
            latent_points.append(mol_vec.data.cpu().numpy()) 
        # output  X: "N X d" latent embedding;  y: label "N X 1"
        X = np.vstack(latent_points) 
        y = np.array([-self.oracle(s) for s in smiles]).reshape((-1,1)) 
        ''' 

        ## 0.1 training data 
        smiles_lst = self.all_smiles
        shuffle(smiles_lst)
        train_smiles_lst = smiles_lst[:config['train_num']]
        # y = self.oracle(train_smiles_lst)
        y = []
        train_X = []
        for smiles in train_smiles_lst:
            # x = vae_model.string2tensor(smiles)
            # x = x.unsqueeze(0)
            # z, _ = vae_model.forward_encoder(x) 
            batch = [smiles]
            try:
                mol_vec = vae_model.encode_latent_mean(batch)
                z = mol_vec.view(1,-1)
                train_X.append(z)
                yy = self.oracle(smiles)
                y.append(yy)
            except:
                pass 
        train_X = torch.cat(train_X, dim=0)
        train_X = train_X.detach()
        train_Y = torch.FloatTensor(y).view(-1,1)

        patience = 0
        
        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            # 1. Fit a Gaussian Process model to data
            gp = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            # 2. Construct an acquisition function
            UCB = UpperConfidenceBound(gp, beta=0.1) 

            # 3. Optimize the acquisition function 
            for _ in range(config['bo_batch']):
                bounds = torch.stack([torch.min(train_X, 0)[0], torch.max(train_X, 0)[0]])
                z, acq_value = optimize_acqf(
                    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
                )
                #### z: shape [20, d]

                ######## old version (smiles-vae): decode
                # new_smiles = vae_model.decoder_z(z)
                ######## old version (smiles-vae): decode
                ######## new version (jtvae): decode
                new_smiles = []
                for i in range(z.shape[0]):
                    _z = z[i].view(1,-1)
                    tree_vec, mol_vec = torch.hsplit(_z, 2)
                    s = model.decode(tree_vec, mol_vec, prob_decode=False)
                    new_smiles.append(s)
                ######## new version (jtvae): decode


                new_score = self.oracle(new_smiles)
                if new_score == 0:
                    new_smiles = choice(smiles_lst)
                    new_score = self.oracle(new_smiles)             

                new_score = torch.FloatTensor([new_score]).view(-1,1)

                train_X = torch.cat([train_X, z], dim = 0)
                train_Y = torch.cat([train_Y, new_score], dim = 0)
                if train_X.shape[0] > 100:
                    train_X = train_X[-config['train_num']:]
                    train_Y = train_Y[-config['train_num']:]

            # early stopping
            if len(self.oracle) > 100:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience * 100:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0
            
            if self.finish:
                print('max oracle hit, abort ...... ')
                break 





    # def _optimize(self, oracle, config):
    #     """
    #     old version of jtvae that use old BO.  
    #     """
    #     self.oracle.assign_evaluator(oracle)
    #     train_size = 100

    #     vocab = [x.strip("\r\n ") for x in open(config['vocab_path'])] 
    #     vocab = Vocab(vocab)

    #     random_seed = int(config['random_seed'])
    #     np.random.seed(random_seed)

    #     ###### old version: load train data
    #     # with open(config['data_path']) as f:
    #     #     smiles = f.readlines()
    #     # smiles = [s.strip() for s in smiles]
    #     # shuffle(smiles)
    #     # smiles = smiles[:train_size]

    #     ###### new version: load train data
    #     smiles_lst = self.all_smiles
    #     shuffle(smiles_lst)
    #     smiles = smiles_lst[:config['train_num']]


    #     batch_size = 100
    #     hidden_size = int(config['hidden_size'])
    #     latent_size = int(config['latent_size'])
    #     depthT = int(config['depthT'])
    #     depthG = int(config['depthG'])

    #     model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    #     model.load_state_dict(torch.load(config['model_path']))
    #     model = model.cuda()


    #     print('generate latent variable')
    #     latent_points = []
    #     for i in tqdm(range(0, len(smiles), batch_size)):
    #         batch = smiles[i:i+batch_size]
    #         mol_vec = model.encode_latent_mean(batch)
    #         latent_points.append(mol_vec.data.cpu().numpy())

    #     # output  X: "N X d" latent embedding;  y: label "N X 1"
    #     X = np.vstack(latent_points) 
    #     y = np.array([-self.oracle(s) for s in smiles]).reshape((-1,1)) 

    #     #### permutation & split 
    #     n = X.shape[0]
    #     permutation = np.random.choice(n, n, replace = False)
    #     X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
    #     X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]
    #     y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
    #     y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

    #     X_train = X_train[:200]
    #     X_test = X_test[:50]
    #     y_train = y_train[:200]
    #     y_test = y_test[:50]


    #     M,K = 50, 25
    #     print('fit GP')
    #     for iteration in tqdm(range(config['max_iter'])):
    #         np.random.seed(iteration * random_seed) 
    #         sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    #         print('train sparse GP')
    #         sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 10, learning_rate = 0.001)
    #         # print('finish train sparse GP')

    #         pred, uncert = sgp.predict(X_test, 0 * X_test)
    #         error = np.sqrt(np.mean((pred - y_test)**2))
    #         testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    #         # print('Test RMSE: ', error)
    #         # print('Test ll: ', testll)

    #         pred, uncert = sgp.predict(X_train, 0 * X_train)
    #         error = np.sqrt(np.mean((pred - y_train)**2))
    #         trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    #         # print('Train RMSE: ', error)
    #         # print('Train ll: ', trainll)

    #         # We pick the next K inputs
    #         # print(np.min(X_train,0)[:10], np.max(X_train, 0)[:10])
    #         print('batched_greedy_ei')
    #         next_inputs = sgp.batched_greedy_ei(K, np.min(X_train, 0), np.max(X_train, 0))
    #         # print('end batched_greedy_ei')
    #         valid_smiles = []
    #         new_features = []
    #         for i in range(K):
    #             all_vec = next_inputs[i].reshape((1,-1))
    #             tree_vec,mol_vec = np.hsplit(all_vec, 2)
    #             tree_vec = create_var(torch.from_numpy(tree_vec).float())
    #             mol_vec = create_var(torch.from_numpy(mol_vec).float())
    #             s = model.decode(tree_vec, mol_vec, prob_decode=False)
    #             if s is not None: 
    #                 valid_smiles.append(s)
    #                 new_features.append(all_vec)

    #         print(len(valid_smiles), "molecules are found")
    #         valid_smiles = valid_smiles[:50]
    #         new_features = next_inputs[:50]
    #         new_features = np.vstack(new_features)

    #         scores = [-self.oracle(s) for s in valid_smiles]
    #         # print(valid_smiles, scores)

    #         if len(new_features) > 0:
    #             X_train = np.concatenate([ X_train, new_features ], 0)
    #             y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
    #         X_train = X_train[-50:]
    #         y_train = y_train[-50:]
    #         if self.oracle.finish: 
    #             print("max oracle hits, exit")
    #             break 



