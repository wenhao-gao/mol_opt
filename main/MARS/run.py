import os
import rdkit
import torch
import random
import pathlib
import argparse
import numpy as np
import logging as log
from tqdm import tqdm
from tdc import Oracle 
from rdkit import Chem, RDLogger
import sys, yaml 
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
from estimator.estimator import Estimator
from proposal.models.editor_basic import BasicEditor
from proposal.proposal import Proposal_Random, Proposal_Editor, Proposal_Mix
from sampler import Sampler_SA, Sampler_MH, Sampler_Recursive
from datasets.utils import load_mols

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

"""
objectives: 'gsk3b,jnk3,qed,sa'
score_wght: '1.0, 1.0, 1.0, 1.0'
score_succ: '0.5, 0.5, 0.6, .67'
score_clip: '0.6, 0.6, 0.7, 0.7'

# config['objectives'] = config['objectives'].split(',')
# config['score_wght'] = [float(_) for _ in config['score_wght'].split(',')]
# config['score_succ'] = [float(_) for _ in config['score_succ'].split(',')]
# config['score_clip'] = [float(_) for _ in config['score_clip'].split(',')]
# assert len(config['score_wght']) == len(config['objectives'])
# assert len(config['score_succ']) == len(config['objectives'])
# assert len(config['score_clip']) == len(config['objectives'])

estimator is used for scoring, will be replaced by tdc.oracle 

"""


class MARS_Optimizer(BaseOptimizer):
    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "mars"


    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        config['device'] = torch.device(config['device'])
        config['run_dir'] = os.path.join(config['root_dir'], config['run_dir'])
        config['data_dir'] = os.path.join(config['root_dir'], config['data_dir'])
        os.makedirs(config['run_dir'], exist_ok=True)

        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)

        ### estimator
        if config['mols_ref']: 
            config['mols_ref'] = load_mols(config['data_dir'], config['mols_ref'])
        # estimator = Estimator(config)

        run_dir = config['run_dir']

        ### proposal
        editor = BasicEditor(config).to(config['device']) if not config['proposal'] == 'random' else None
        if config['editor_dir'] is not None: # load pre-trained editor
            path = os.path.join(config['root_dir'], config['editor_dir'], 'model_best.pt')
            editor.load_state_dict(torch.load(path, map_location=torch.device(config['device'])))
            print('successfully loaded editor model from %s' % path)
        if config['proposal'] == 'random': proposal = Proposal_Random(config)
        elif config['proposal'] == 'editor': proposal = Proposal_Editor(config, editor)
        elif config['proposal'] == 'mix': proposal = Proposal_Mix(config, editor)

        ### sampler
        if config['sampler'] == 're': sampler = Sampler_Recursive(config, proposal, oracle, config['max_n_oracles']) 
        elif config['sampler'] == 'sa': sampler = Sampler_SA(config, proposal, oracle, config['max_n_oracles'])
        elif config['sampler'] == 'mh': sampler = Sampler_MH(config, proposal, oracle, config['max_n_oracles'])

        ### sampling
        if config['mols_init']:
            mols = load_mols(config['data_dir'], config['mols_init'])
            mols = random.choices(mols, k=config['num_mols'])
            mols_init = mols[:config['num_mols']]
        else: 
            mols_init = [Chem.MolFromSmiles('CC') for _ in range(config['num_mols'])]
        self.mol_buffer = sampler.sample(run_dir, mols_init, self.mol_buffer)



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
        optimizer = MARS_Optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == '__main__':
    main() 

