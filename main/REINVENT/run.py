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
import time
from train_agent import train_agent

class REINVENToptimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "REINVENT"

    def _optimize(self, oracle, config):
        pass 


parser = argparse.ArgumentParser(description="Main script for running the model")
parser.add_argument('--scoring-function', action='store', dest='scoring_function',
                    choices=['activity_model', 'tanimoto', 'no_sulphur'],
                    default='tanimoto',
                    help='What type of scoring function to use.')
parser.add_argument('--scoring-function-kwargs', action='store', dest='scoring_function_kwargs',
                    nargs="*",
                    help='Additional arguments for the scoring function. Should be supplied with a '\
                    'list of "keyword_name argument". For pharmacophoric and tanimoto '\
                    'the keyword is "query_structure" and requires a SMILES. ' \
                    'For activity_model it is "clf_path " '\
                    'pointing to a sklearn classifier. '\
                    'For example: "--scoring-function-kwargs query_structure COc1ccccc1".')
parser.add_argument('--learning-rate', action='store', dest='learning_rate',
                    type=float, default=0.0005)
parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                    default=3000)
parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                    default=64)
parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                    default=20)
parser.add_argument('--experience', action='store', dest='experience_replay', type=int,
                    default=0, help='Number of experience sequences to sample each step. '\
                    '0 means no experience replay.')
parser.add_argument('--num-processes', action='store', dest='num_processes',
                    type=int, default=0,
                    help='Number of processes used to run the scoring function. "0" means ' \
                    'that the scoring function will be run in the main process.')
parser.add_argument('--prior', action='store', dest='restore_prior_from',
                    default='data/Prior.ckpt',)
parser.add_argument('--agent', action='store', dest='restore_agent_from',
                    default='data/Prior.ckpt',)
parser.add_argument('--save-dir', action='store', dest='save_dir',)

if __name__ == "__main__":

    arg_dict = vars(parser.parse_args())

    if arg_dict['scoring_function_kwargs']:
        kwarg_list = arg_dict.pop('scoring_function_kwargs')
        if not len(kwarg_list) % 2 == 0:
            raise ValueError("Scoring function kwargs must be given as pairs, "\
                             "but got a list with odd length.")
        kwarg_dict = {i:j for i, j in zip(kwarg_list[::2], kwarg_list[1::2])}
        arg_dict['scoring_function_kwargs'] = kwarg_dict
    else:
        arg_dict['scoring_function_kwargs'] = dict()

    train_agent(**arg_dict)
























def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

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
        optimizer = REINVENToptimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main() 









