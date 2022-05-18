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


TB_LOGS_FILE = 'tb_logs'
HC_RESULTS_FOLDER = 'hc_results'


'''
export PYTHONPATH=${PYTHONPATH}:${DIR}
export PYTHONPATH=${PYTHONPATH}:${DIR}/submodules/autoencoders
export PYTHONPATH=${PYTHONPATH}:${DIR}/submodules/GNN
'''


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



def get_walk_direction_function(name):
    # Although we only consider walking randomly we leave it open to consider alternative ways to explore the latent
    # space in the future

    if name == 'random':
        # Set up function that will give direction to walk in
        def get_direction(model, current_z):
            return F.normalize(torch.randn(*current_z.shape, device=current_z.device), dim=1)
    else:
        raise NotImplementedError
    return get_direction



# class Params:
#     def __init__(self, task_name, weight_path: str):
#         self.device = settings.torch_device()

#         self.train_tree_path = os.path.join(path_here, "scripts/dataset_creation/data/uspto-train-depth_and_tree_tuples.pick")
#         self.valid_tree_path = os.path.join(path_here, "scripts/dataset_creation/data/uspto-valid-depth_and_tree_tuples.pick")

#         self.weight_path = weight_path
#         self.num_dataloader_workers = 4

#         self.opt_name = task_name
#         time_run = strftime("%y-%m-%d_%H:%M:%S", gmtime())
#         self.exp_uuid = uuid.uuid4()
#         self.run_name = f"doggen_hillclimbing_{time_run}_{self.exp_uuid}_{self.opt_name}"
#         print(f"Run name is {self.run_name}\n\n")
#         self.property_predictor = opt_utils.get_task('qed')   #### qed, sas, ....

#         self.log_for_reaction_predictor_path = path.join(path_here, "logs", f"reactions-{self.run_name}.log")


from main.optimizer import BaseOptimizer



class DoG_AE_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "dog_ae"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)


        weight_path = os.path.join(path_here, 'scripts/dogae/train/chkpts/dogae_weights.pth.pick')
        params = Params(weight_path)



        rng = np.random.RandomState(564165416)
        torch.manual_seed(6514564)

        # Model!
        log_path = path.join("logs", f"reactions-{params.run_name}.log")
        model, collate_func, *_ = dogae_utils.load_dogae_model(params.device, log_path,
                                                               weight_path=params.weight_path)

        # Some starting locations
        tuple_trees = train_utils.load_tuple_trees(params.tuple_tree_path, rng)
        indices_chosen = rng.choice(len(tuple_trees), params.num_starting_locations, replace=False)
        tuple_trees = [tuple_trees[i] for i in indices_chosen]
        # tuple_trees = tuple_trees[:2000]

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
        # exit() 
        # train_X = torch.cat(train_X, dim=0)
        # train_X = train_X.detach()
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

                # new_smiles = vae_model.decoder_z(z)  ### model 
                # mol = smiles_to_rdkit_mol(new_smiles) 
                # print(z)
                new_out, _ = model.decode_from_z_no_grad(z, sample_x=False) 
                ### new_out is list of syn_dags.data.synthesis_trees.SynthesisTree 
                new_smiles = new_out[0].root_smi
                # print(z, new_out)
                print(new_smiles)
                # exit()



                new_score = self.oracle(new_smiles)

                # if new_score == 0:
                #     # new_smiles = choice(smiles_lst)
                #     new_score = self.oracle(new_smiles)             

                new_score = torch.FloatTensor([new_score]).view(-1,1)

                train_X = torch.cat([train_X, z], dim = 0)
                train_Y = torch.cat([train_Y, new_score], dim = 0)
                if train_X.shape[0] > config['train_num']:
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




        # exit()  

        # # Set up storage
        # samples_out = [[] for _ in range(initial_embeddings.shape[0])]
        # samples_out_set = [set() for _ in range(initial_embeddings.shape[0])]

        # # walk dir
        # get_direction = get_walk_direction_function(params.walk_strategy)

        # # Now loop through
        # last_z = initial_embeddings.detach().clone()
        # while True:

        #     # Run for params.num_steps_before_filtering steps
        #     temp_samples_out_before_filtering = [[] for _ in range(initial_embeddings.shape[0])]
        #     for i in tqdm(range(params.num_steps_before_filtering), desc="Inner Loop..."):

        #         if i > 0:  # first time round just decode.
        #             direction = get_direction(model, last_z)
        #         else:
        #             direction = 0.

        #         with torch.no_grad():
        #             new_z = last_z + params.epsilon * direction
        #             new_out, _ = model.decode_from_z_no_grad(new_z, sample_x=False)
        #         for current_list, new_item in zip(temp_samples_out_before_filtering, new_out):
        #             current_list.append(new_item)
        #         last_z = new_z.detach().clone()

        #     # Work out how many unique we actually got on this time.
        #     for current_samples, current_samples_set, new_possible_samples in zip(samples_out, samples_out_set, temp_samples_out_before_filtering):
        #         for new_sample in new_possible_samples:
        #             new_sample: synthesis_trees.SynthesisTree
        #             immutable_repr = new_sample.immutable_repr(include_order=params.include_same_trees_different_order)
        #             if immutable_repr in current_samples_set:
        #                 pass
        #             else:
        #                 current_samples.append(new_sample)
        #                 current_samples_set.add(immutable_repr)
        #                 if len(current_samples) >= params.num_unique_after:
        #                     break

        #     # If we have got the number we need we can break!
        #     num_that_are_unique = sum([len(el) >= params.num_unique_after for el in samples_out])
        #     tqdm.write(f"## {num_that_are_unique} / {len(samples_out)} have found at least {params.num_unique_after} unique samples")
        #     if num_that_are_unique == len(samples_out):
        #         tqdm.write("completed while loop as collected enough samples for each.")
        #         break
        #     else:
        #         tqdm.write("Starting new loop!.")




        # # Create the results
        # order_change = np.argsort(new_orders)
        # samples_out_reordered = [samples_out[i] for i in order_change ]
        # syn_trees_initial_reordered = [pred_batch_largest_first.syn_trees[i] for i in order_change]

        # results = dict(path_to_starting=params.tuple_tree_path,
        #                indices_chosen=indices_chosen,
        #                tuple_trees_in=tuple_trees,
        #                syn_trees_initial=syn_trees_initial_reordered,
        #                syn_trees_decoded=samples_out_reordered,
        #                walk_strategy=params.walk_strategy)

        # # Save to disk
        # os.makedirs(path.join(OUT_DIR, params.run_name))
        # fname = path.join(OUT_DIR, f"{params.run_name}.pick")
        # tqdm.write(f"Saving results in {fname}")
        # misc.to_pickle(results, fname)





"""
    /project/molecular_data/graphnn/mol_opt/main/dog_gen/syn_dags/script_utils/doggen_utils.py

"""




"""

BO in smiles_vae 

        ## 0. load vae model & get training data
        vae_model = torch.load(config['save_model'])
        smiles_lst = self.all_smiles
        shuffle(smiles_lst)
        train_smiles_lst = smiles_lst[:config['train_num']]
        y = self.oracle(train_smiles_lst)
        train_X = []
        for smiles in train_smiles_lst:
            x = vae_model.string2tensor(smiles)
            x = x.unsqueeze(0)
            z, _ = vae_model.forward_encoder(x) 
            train_X.append(z)
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
                z, _ = optimize_acqf(
                    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
                )

                new_smiles = vae_model.decoder_z(z)
                mol = smiles_to_rdkit_mol(new_smiles)
                
                if mol is None:
                    new_score = 0
                else:
                    new_score = self.oracle(new_smiles)

                if new_score == 0:
                    new_smiles = choice(smiles_lst)
                    new_score = self.oracle(new_smiles)             

                new_score = torch.FloatTensor([new_score]).view(-1,1)

                train_X = torch.cat([train_X, z], dim = 0)
                train_Y = torch.cat([train_Y, new_score], dim = 0)
                if train_X.shape[0] > config['train_num']:
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


"""

