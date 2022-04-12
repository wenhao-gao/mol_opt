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

class SMILES_LSTM_HC_Optimizer(BaseOptimizer):

  def __init__(self, args=None):
    super().__init__(args)
    self.model_name = "smiles_lstm_hc"

  def _optimize(self, oracle, config):

    self.oracle.assign_evaluator(oracle)

    pretrained_model_path = os.path.join(path_here, 'pretrained_model', 'model_final_0.473.pt')

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
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
        optimizer = SMILES_LSTM_HC_Optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main() 





