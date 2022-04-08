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
# from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
# from guacamol.utils.helpers import setup_default_logger
from smiles_rnn_directed_generator import SmilesRnnDirectedGenerator

'''
default 
    # n_epochs = 20
    # mols_to_sample = 1024
    # keep_top = 512 
    # optimize_n_epochs = 2 
    # max_len = 100 
    # optimize_batch_size = 256 
    # benchmark_num_samples = 4096 
    # random_start = True
    # n_jobs = -1 
'''

class SELFIES_LSTM_HC_Optimizer(BaseOptimizer):

  def __init__(self, args=None):
    super().__init__(args)
    self.model_name = "smiles_lstm_hc"

  def _optimize(self, oracle, config):

    self.oracle.assign_evaluator(oracle)

    model_path = os.path.join(path_here, 'pretrained_model', 'model_final_0.698.pt')

    smiles_file = os.path.join(path_here, 'zinc_500.txt')
    with open(smiles_file, 'r') as fin:
      lines = fin.readlines() 
    start_smiles_lst = [line.strip() for line in lines]

    optimizer = SmilesRnnDirectedGenerator(pretrained_model_path=model_path,
                                           n_epochs=config['n_epochs'],
                                           mols_to_sample=config['mols_to_sample'],
                                           keep_top=config['keep_top'],
                                           optimize_n_epochs=config['optimize_n_epochs'],
                                           max_len=config['max_len'],
                                           optimize_batch_size=config['optimize_batch_size'],
                                           number_final_samples=config['benchmark_num_samples'],
                                           random_start=config['random_start'],
                                           smi_file=smiles_file,
                                           n_jobs=config['n_jobs'])

    result = optimizer.generate_optimized_molecules(self.oracle, number_molecules = 20100,
                                     starting_population = start_smiles_lst)











