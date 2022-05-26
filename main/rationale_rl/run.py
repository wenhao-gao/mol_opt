
import os, torch, random
import numpy as np 
from tqdm import tqdm 
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
import math, random, sys, os

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

from fuseprop import *
def remove_order(s):
    for x in range(15):
        s = s.replace(":%d]" % (x,), ":1]")
    return s


from properties import get_scoring_function
qed_sa_func = lambda x: x[0] >= 0.5 and x[1] >= 0.5 and x[2] >= 0.6 and x[3] <= 4.0
normal_func = lambda x: min(x) >= 0.5


# Decode molecules
def decode_rationales(model, rationale_dataset):
    loader = DataLoader(rationale_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x:x[0])
    model.eval()
    cand_mols = []
    single_block_size = 100 
    #### different from origin setup, save oracle call. so that small oracle can update model. made by tianfan 
    with torch.no_grad():
        for ii, init_smiles in tqdm(enumerate(loader)):
            final_smiles = model.decode(init_smiles)
            mols = [(x,y) for x,y in zip(init_smiles, final_smiles) if y and '.' not in y]
            mols = [(x,y) for x,y in mols if Chem.MolFromSmiles(y).HasSubstructMatch(Chem.MolFromSmiles(x))]
            cand_mols.extend(mols)
            cand_mols = list(set(cand_mols))
            if len(cand_mols) > single_block_size:
                break  
    return cand_mols

# Predict properties and filter 
def property_filter(cand_mols, scoring_function):
    rationales, smiles_list = zip(*cand_mols)
    cand_props = scoring_function(smiles_list)
    new_data = []
    for (init_smiles, final_smiles), prop in zip(cand_mols, cand_props):
        if prop > 0.5:    
            new_data.append((init_smiles, final_smiles))
    return new_data

class Rationale_RL_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "rationale_rl"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        atom_vocab = common_atom_vocab
        config['atom_vocab'] = common_atom_vocab 

        # prop_funcs = [get_scoring_function(prop) for prop in 'qed,sa'.split(',')]
        # scoring_function = lambda x : list( zip(*[func(x) for func in prop_funcs]) ) #### oracle 

        with open(os.path.join(path_here, "data/rationales.txt")) as f:
            rationales = [line.split()[1] for line in f]
            rationales = unique_rationales(rationales)
            rationale_dataset = SubgraphDataset(rationales, atom_vocab, config['decode_batch_size'], config['num_decode'])

        model = AtomVGNN(config).cuda()
        model.load_state_dict(torch.load(config['init_model']))

        print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = lr_scheduler.ExponentialLR(optimizer, config['anneal_rate'])

        param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
        grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

        for epoch in range(config['epoch']):
            print('epoch', epoch)
            cand_mols = decode_rationales(model, rationale_dataset)
            # cand_mols, rationale_dist = property_filter(cand_mols, scoring_function, args)
            ## property_filter 
            rationales, smiles_list = zip(*cand_mols)
            cand_props = self.oracle(list(smiles_list))
            new_data = []
            for (init_smiles, final_smiles), prop in zip(cand_mols, cand_props):
                if prop > 0.5:
                    new_data.append((init_smiles, final_smiles))

            cand_mols = list(set(new_data))
            if len(cand_mols) == 0:
                print("early stopping, the generated molecules are all the same and converge")
                break #### no patience: early stopping, the generated molecules are all the same and converge, made by tianfan  
            random.shuffle(cand_mols)

            # Update model
            dataset = MoleculeDataset(cand_mols, atom_vocab, config['batch_size'])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x:x[0])
            model.train()

            meters = np.zeros(5)
            for total_step, batch in tqdm(enumerate(dataset)):
                if batch is None: continue

                model.zero_grad()
                loss, kl_div, wacc, tacc, sacc = model(*batch, beta=config['beta'])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
                optimizer.step()
                
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])
            print('used oracle call', len(self.oracle))
            if self.finish:
                break  


