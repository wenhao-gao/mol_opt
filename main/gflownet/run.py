from __future__ import print_function

import argparse
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import torch

from main.optimizer import BaseOptimizer
from gflownet_tdc import train_model_with_proxy, Dataset, make_model

import os 
path_here = os.path.dirname(os.path.realpath(__file__))

model_parser = argparse.ArgumentParser()
model_parser.add_argument("--learning_rate", default=5e-4, help="Learning rate", type=float)
model_parser.add_argument("--mbsize", default=4, help="Minibatch size", type=int)
model_parser.add_argument("--opt_beta", default=0.9, type=float)
model_parser.add_argument("--opt_beta2", default=0.999, type=float)
model_parser.add_argument("--opt_epsilon", default=1e-8, type=float)
model_parser.add_argument("--nemb", default=256, help="#hidden", type=int)
model_parser.add_argument("--min_blocks", default=2, type=int)
model_parser.add_argument("--max_blocks", default=8, type=int)
model_parser.add_argument("--num_iterations", default=50000, type=int)
model_parser.add_argument("--num_conv_steps", default=10, type=int)
model_parser.add_argument("--log_reg_c", default=2.5e-5, type=float)
model_parser.add_argument("--reward_exp", default=10, type=float)
model_parser.add_argument("--reward_norm", default=8, type=float)
model_parser.add_argument("--sample_prob", default=1, type=float)
model_parser.add_argument("--R_min", default=0.0001, type=float)
model_parser.add_argument("--leaf_coef", default=10, type=float)
model_parser.add_argument("--clip_grad", default=0, type=float)
model_parser.add_argument("--clip_loss", default=0, type=float)
model_parser.add_argument("--replay_mode", default='online', type=str)
model_parser.add_argument("--bootstrap_tau", default=0, type=float)
model_parser.add_argument("--weight_decay", default=0, type=float)
model_parser.add_argument("--random_action_prob", default=0.05, type=float)
model_parser.add_argument("--array", default='')
model_parser.add_argument("--repr_type", default='block_graph')
model_parser.add_argument("--model_version", default='v4')
model_parser.add_argument("--run", default=0, help="run", type=int)
model_parser.add_argument("--save_path", default='results/')
model_parser.add_argument("--proxy_path", default='./data/pretrained_proxy')
model_parser.add_argument("--print_array_length", default=False, action='store_true')
model_parser.add_argument("--progress", default='yes')
model_parser.add_argument("--floatX", default='float64')
model_parser.add_argument("--include_nblocks", default=False)
model_parser.add_argument("--balanced_loss", default=True)
# If True this basically implements Buesing et al's TreeSample Q,
# samples uniformly from it though, no MTCS involved
model_parser.add_argument("--do_wrong_thing", default=False)
model_parser.add_argument("--obj", default='qed')

class GFlowNet_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "gflownet"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        args = model_parser.parse_args()
        
        # bpath = "/home/whgao/mol_opt/main/gflownet/data/blocks_PDB_105.json"
        bpath = os.path.join(path_here, 'data/blocks_PDB_105.json')
        device = torch.device('cuda')

        if args.floatX == 'float32':
            args.floatX = torch.float
        else:
            args.floatX = torch.double
        dataset = Dataset(args, bpath, device, floatX=args.floatX)
        print(args)

        mdp = dataset.mdp

        model = make_model(args, mdp)
        model.to(args.floatX)
        model.to(device)

        train_model_with_proxy(args, model, self.oracle, dataset, do_save=True)
        print('Done.')


class GFlowNet_AL_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "gflownet_al"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        args = model_parser.parse_args()
        
        # bpath = "/home/whgao/mol_opt/main/gflownet/data/blocks_PDB_105.json"
        bpath = os.path.join(path_here, 'data/blocks_PDB_105.json')
        device = torch.device('cuda')

        if args.floatX == 'float32':
            args.floatX = torch.float
        else:
            args.floatX = torch.double
        dataset = Dataset(args, bpath, device, floatX=args.floatX)
        print(args)

        mdp = dataset.mdp

        model = make_model(args, mdp)
        model.to(args.floatX)
        model.to(device)

        train_model_with_proxy(args, model, self.oracle, dataset, do_save=True)
        print('Done.')

