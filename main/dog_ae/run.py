from __future__ import print_function

import argparse
import heapq
import yaml
import os
import random
from time import time
from typing import List, Optional

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
from tdc import Oracle
rdBase.DisableLog('rdApp.error')

from os import path
from time import strftime, gmtime
import uuid
import pickle
import csv

import numpy as np
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from docopt import docopt

import os, sys 
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append(os.path.join(path_here, 'submodules/autoencoders'))
sys.path.append(os.path.join(path_here, 'submodules/GNN'))

from syn_dags.script_utils import train_utils
from syn_dags.model import doggen
from syn_dags.script_utils import doggen_utils
from syn_dags.script_utils import opt_utils
from syn_dags.utils import settings

from time import strftime, gmtime
from os import path
import os

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from docopt import docopt

from syn_dags.script_utils import train_utils
from syn_dags.script_utils import dogae_utils
from syn_dags.utils import misc
from syn_dags.data import synthesis_trees

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from main.optimizer import BaseOptimizer, Objdict

TB_LOGS_FILE = 'tb_logs'
HC_RESULTS_FOLDER = 'hc_results'


class Params:
    def __init__(self, weight_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tuple_tree_path = os.path.join(path_here, "scripts/dataset_creation/data/uspto-train-depth_and_tree_tuples.pick")

        self.weight_path = weight_path

        self.num_starting_locations = 50
        self.num_steps_before_filtering = 30
        self.epsilon = 0.01
        self.num_unique_after = 6
        self.include_same_trees_different_order = False

        self.walk_strategy = "random"
        print(self.walk_strategy)

        time_run = strftime("%d-%b-%H-%M", gmtime())
        f_name_weights = path.splitext(path.basename(self.weight_path))[0]
        self.run_name = f"walks_for{f_name_weights}_done_{time_run}_strategy{self.walk_strategy}"
        print(f"Run name is {self.run_name}")
        print(f"Checkpoint name is {self.weight_path}")
        print(f"Tuple tree path (where we pick starting points from)  is {self.tuple_tree_path}")


class DoG_AE_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "dog_ae"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        config = Objdict(config)

        weight_path = os.path.join(path_here, 'scripts/dogae/train/chkpts/dogae_weights.pth.pick')
        params = Params(weight_path)

        randomstate = np.random.randint(1000)
        rng = np.random.RandomState(randomstate)

        # Model!
        log_path = path.join(path_here, "logs")
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_path = path.join(log_path, f"reactions-{params.run_name}.log")
        model, collate_func, *_ = dogae_utils.load_dogae_model(params.device, log_path,
                                                               weight_path=params.weight_path)

        # Some starting locations
        tuple_trees = train_utils.load_tuple_trees(params.tuple_tree_path, rng)
        indices_chosen = rng.choice(len(tuple_trees), params.num_starting_locations, replace=False)
        tuple_trees = [tuple_trees[i] for i in indices_chosen]

        # Get the first embeddings
        pred_batch_largest_first, new_orders = collate_func(tuple_trees)
        pred_batch_largest_first.inplace_to(params.device)

        embedded_graphs = model.mol_embdr(pred_batch_largest_first.molecular_graphs)
        pred_batch_largest_first.molecular_graph_embeddings = embedded_graphs
        new_node_feats_for_dag = pred_batch_largest_first.molecular_graph_embeddings[pred_batch_largest_first.dags_for_inputs.node_features.squeeze(), :]
        pred_batch_largest_first.dags_for_inputs.node_features = new_node_feats_for_dag
        initial_embeddings = model._run_through_to_z(pred_batch_largest_first, sample_z=False)
        print(initial_embeddings.shape) ##### [50, 25]
        last_z = initial_embeddings.detach().clone()
        train_X = last_z 
        smiles_list = [i[0] for i in tuple_trees]
        y = self.oracle(smiles_list)
        train_Y = torch.FloatTensor(y).view(-1,1) 

        patience = 0
    
        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            try:

                # 1. Fit a Gaussian Process model to data
                gp = SingleTaskGP(train_X, train_Y)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)

                # 2. Construct an acquisition function
                UCB = UpperConfidenceBound(gp, beta=0.1) 

                # 3. Optimize the acquisition function 
                for _ in range(config['bo_batch']):
                    bounds = torch.stack([torch.min(train_X, 0)[0], torch.max(train_X, 0)[0]])
                    z, _ = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,)

                    new_out, _ = model.decode_from_z_no_grad(z, sample_x=False) 
                    ### new_out is list of syn_dags.data.synthesis_trees.SynthesisTree 
                    new_smiles = new_out[0].root_smi
                    print(new_smiles)


                    new_score = self.oracle(new_smiles)           
                    new_score = torch.FloatTensor([new_score]).view(-1,1)

                    train_X = torch.cat([train_X, z], dim = 0)
                    train_Y = torch.cat([train_Y, new_score], dim = 0)
                    if train_X.shape[0] > config['train_num']:
                        train_X = train_X[-config['train_num']:]
                        train_Y = train_Y[-config['train_num']:]
            except:
                break 

            # early stopping
            if len(self.oracle) > 100:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0
            
            if self.finish:
                print('max oracle hit, abort ...... ')
                break 


