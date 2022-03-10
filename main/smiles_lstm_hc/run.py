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

class LSTM_HC_Optimizer(BaseOptimizer):

  def __init__(self, args=None):
    super().__init__(args)
    self.model_name = "LSTM_HC"

  def _optimize(self, oracle, config):
    self.oracle.assign_evaluator(oracle)
    model_path = os.path.join(path_here, 'pretrained_model', 'model_final_0.473.pt')
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

    result = optimizer.generate_optimized_molecules(self.oracle, number_molecules = 1000,
                                     starting_population = start_smiles_lst)



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


        # max_n_oracles = config['max_n_oracles']
        oracle = Oracle(name = oracle_name)
        optimizer = LSTM_HC_Optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main() 





