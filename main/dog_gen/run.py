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

import sys 
path_here = os.path.dirname(os.path.realpath(__file__))
path_dogae = '/'.join(path_here.split('/')[:-1])
path_dogae = os.path.join(path_dogae, 'dog_ae')
sys.path.append(path_here)
sys.path.append(os.path.join(path_here, 'submodules/autoencoders'))
sys.path.append(os.path.join(path_here, 'submodules/GNN'))

from os import path
from time import strftime, gmtime
import uuid
import pickle
import csv

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from docopt import docopt

from syn_dags.script_utils import train_utils
from syn_dags.model import doggen
from syn_dags.script_utils import doggen_utils
from syn_dags.script_utils import opt_utils
from syn_dags.utils import settings

from main.optimizer import BaseOptimizer, Objdict

TB_LOGS_FILE = 'tb_logs'
HC_RESULTS_FOLDER = 'hc_results'


class Params:
    def __init__(self, task_name, weight_path: str):
        self.device = settings.torch_device()

        self.train_tree_path = os.path.join(path_dogae, "scripts/dataset_creation/data/uspto-train-depth_and_tree_tuples.pick")
        self.valid_tree_path = os.path.join(path_dogae, "scripts/dataset_creation/data/uspto-valid-depth_and_tree_tuples.pick")

        self.weight_path = weight_path
        self.num_dataloader_workers = 4

        self.opt_name = task_name
        time_run = strftime("%y-%m-%d_%H:%M:%S", gmtime())
        self.exp_uuid = uuid.uuid4()
        self.run_name = f"doggen_hillclimbing_{time_run}_{self.exp_uuid}_{self.opt_name}"
        print(f"Run name is {self.run_name}\n\n")
        self.property_predictor = opt_utils.get_task('qed')   #### qed, sas, ....

        if not path.exists(path.join(path_here, "logs")):
            os.mkdir(path.join(path_here, "logs"))

        self.log_for_reaction_predictor_path = path.join(path_here, "logs", f"reactions-{self.run_name}.log")


class DoG_Gen_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "dog_gen"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        config = Objdict(config)
        rng = np.random.RandomState(5156)

        weight_path = os.path.join(path_here, 'scripts/dog_gen/chkpts/doggen_weights.pth.pick')
        params = Params('optimize', weight_path)

        # # Data
        train_trees = train_utils.load_tuple_trees(params.train_tree_path, rng)
        val_trees = train_utils.load_tuple_trees(params.valid_tree_path, rng)
        train_trees = train_trees + val_trees
        # ^ nb we add the train and valid datasets from ordinary training together now for the optimizing as is done
        # for the baselines.

        # # Model (from chkpt)
        model, collate_func, other_parts = doggen_utils.load_doggen_model(params.device, params.log_for_reaction_predictor_path,
                                                                          weight_path=params.weight_path) ### bugs 

        # # TensorBoard
        tb_summary_writer = SummaryWriter(log_dir=TB_LOGS_FILE)

        # # Setup functions needed for hillclimber
        def loss_fn(model: doggen.DogGen, x, new_order):
            # Outside the model shall compute the embeddings of the graph -- these are needed in both the encoder
            # and decoder so saves compute to just compute them once.
            embedded_graphs = model.mol_embdr(x.molecular_graphs)
            x.molecular_graph_embeddings = embedded_graphs
            new_node_feats_for_dag = x.molecular_graph_embeddings[x.dags_for_inputs.node_features.squeeze(), :]
            x.dags_for_inputs.node_features = new_node_feats_for_dag

            loss = model(x).mean()
            return loss

        def prepare_batch(batch, device):
            x, new_order = batch
            x.inplace_to(device)
            return x, new_order

        def create_dataloader(tuple_trees, batch_size):
            return data.DataLoader(tuple_trees, batch_size=batch_size,
                                   num_workers=params.num_dataloader_workers, collate_fn=collate_func,
                                   shuffle=True)

        # # Now put this together for hillclimber constructor arguments
        hparams = doggen_utils.DogGenHillclimbingParams(
            n_samples_per_round = config.n_samples_per_round, 
            n_samples_to_keep_per_round = config.n_samples_to_keep_per_round,
            n_epochs_for_finetuning = config.n_epochs_for_finetuning,
            batch_size = config.batch_size,
            sample_batch_size = config.sample_batch_size,
            learning_rate = config.learning_rate
        )

        #### oracle is params.property_predictor 
        parts = doggen_utils.DogGenHillclimberParts(model, self.oracle,
                                                    set(other_parts['mol_to_graph_idx_for_reactants'].keys()), rng,
                                                    create_dataloader, prepare_batch, loss_fn, params.device)

        # # Now create hillclimber
        hillclimber = doggen_utils.DogGenHillclimber(parts, hparams)

        # # Run!
        print("Starting hill climber")
        sorted_tts = hillclimber.run_hillclimbing(train_trees, tb_summary_writer)  
        print("Done!")
