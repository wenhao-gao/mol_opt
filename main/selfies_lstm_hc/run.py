import os, pickle, torch, random, argparse
from pathlib import Path
import yaml
import numpy as np 
from tqdm import tqdm 
from tdc import Oracle
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
from rnn_generator import SmilesRnnMoleculeGenerator
from rnn_utils import load_rnn_model

'''
default 
    # n_epochs = 20
    # mols_to_sample = 1024
    # keep_top = 512 
    # optimize_n_epochs = 2 
    # max_len = 100 
    # optimize_batch_size = 256 
    # benchmark_num_samples = 4096 
'''

class SELFIES_LSTM_HC_Optimizer(BaseOptimizer):

  def __init__(self, args=None):
    super().__init__(args)
    self.model_name = "selfies_lstm_hc"

  def _optimize(self, oracle, config):

    self.oracle.assign_evaluator(oracle)

    # <<<<<<< HEAD
    #     model_path = os.path.join(path_here, 'pretrained_model', 'model_final_0.698.pt')

    #     smiles_file = os.path.join(path_here, 'zinc_500.txt')
    #     with open(smiles_file, 'r') as fin:
    #       lines = fin.readlines() 
    #     start_smiles_lst = [line.strip() for line in lines]

    #     optimizer = SmilesRnnDirectedGenerator(pretrained_model_path=model_path,
    #                                            n_epochs=config['n_epochs'],
    #                                            mols_to_sample=config['mols_to_sample'],
    #                                            keep_top=config['keep_top'],
    #                                            optimize_n_epochs=config['optimize_n_epochs'],
    #                                            max_len=config['max_len'],
    #                                            optimize_batch_size=config['optimize_batch_size'],
    #                                            number_final_samples=config['benchmark_num_samples'],
    #                                            random_start=config['random_start'],
    #                                            smi_file=smiles_file,
    #                                            n_jobs=config['n_jobs'])

    #     result, self.oracle = optimizer.generate_optimized_molecules(self.oracle, number_molecules = 20100,
    #                                      starting_population = start_smiles_lst)
    # =======
    pretrained_model_path = os.path.join(path_here, 'pretrained_model', 'model_final_0.698.pt')

    population_size = 500

    if self.smi_file is not None:
        # Exploitation run
        starting_population = self.all_smiles[:population_size]
    else:
        # Exploration run
        starting_population = np.random.choice(self.all_smiles, population_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_def = Path(pretrained_model_path).with_suffix('.json')
    sample_final_model_only = False

    model = load_rnn_model(model_def, pretrained_model_path, device, copy_to_cpu=True)

    generator = SmilesRnnMoleculeGenerator(model=model,
                                            max_len=config['max_len'],
                                            device=device)
    print('build generator')

    molecules = generator.optimise(objective=self.oracle,
                                    start_population=starting_population,
                                    n_epochs=config['n_epochs'],
                                    mols_to_sample=config['mols_to_sample'],
                                    keep_top=config['keep_top'],
                                    optimize_batch_size=config['optimize_batch_size'],
                                    optimize_n_epochs=config['optimize_n_epochs'],
                                    pretrain_n_epochs=0)

    # self.oracle.assign_evaluator(oracle)

    # model_path = os.path.join(path_here, 'pretrained_model', 'model_final_0.698.pt')

    # smiles_file = os.path.join(path_here, 'zinc_500.txt')
    # with open(smiles_file, 'r') as fin:
    #   lines = fin.readlines() 
    # start_smiles_lst = [line.strip() for line in lines]

    # optimizer = SmilesRnnDirectedGenerator(pretrained_model_path=model_path,
    #                                        n_epochs=config['n_epochs'],
    #                                        mols_to_sample=config['mols_to_sample'],
    #                                        keep_top=config['keep_top'],
    #                                        optimize_n_epochs=config['optimize_n_epochs'],
    #                                        max_len=config['max_len'],
    #                                        optimize_batch_size=config['optimize_batch_size'],
    #                                        number_final_samples=config['benchmark_num_samples'],
    #                                        random_start=config['random_start'],
    #                                        smi_file=smiles_file,
    #                                        n_jobs=config['n_jobs'])

    # result = optimizer.generate_optimized_molecules(self.oracle, number_molecules = 20100,
    #                                  starting_population = start_smiles_lst)
# >>>>>>> bbab772b978f46c31f79a8fa49f9d5629b7e7599











