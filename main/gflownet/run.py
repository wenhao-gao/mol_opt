from __future__ import print_function

import argparse
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import torch

import os, sys 
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)

from main.optimizer import BaseOptimizer
from gflownet_tdc import train_model_with_proxy, Dataset, make_model


class Argument:
    def __init__(self):
        pass



class GFlowNet_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "gflownet"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        args = Argument() 

        args.learning_rate = config['learning_rate']
        args.mbsize = config['mbsize']
        args.opt_beta = config['opt_beta']
        args.opt_beta2 = config['opt_beta2']
        args.opt_epsilon = config['opt_epsilon']
        args.nemb = config['nemb']
        args.min_blocks = config['min_blocks']
        args.max_blocks = config['max_blocks']
        args.num_iterations = config['num_iterations']
        args.num_conv_steps = config['num_conv_steps']
        args.log_reg_c = config['log_reg_c']
        args.reward_exp = config['reward_exp']
        args.reward_norm = config['reward_norm']
        args.sample_prob = config['sample_prob']
        args.R_min = config['R_min']
        args.leaf_coef = config['leaf_coef']
        args.clip_grad = config['clip_grad']
        args.clip_loss = config['clip_loss']
        args.replay_mode = config['replay_mode']
        args.bootstrap_tau = config['bootstrap_tau']
        args.weight_decay = config['weight_decay']
        args.random_action_prob = config['random_action_prob']
        args.array = config['array']
        args.repr_type = config['repr_type']
        args.model_version = config['model_version']
        args.run = config['run']
        args.save_path = config['save_path']
        args.proxy_path = config['proxy_path']
        args.print_array_length = config['print_array_length']
        args.progress = config['progress']
        args.floatX = config['floatX']
        args.include_nblocks = config['include_nblocks']
        args.balanced_loss = config['balanced_loss']
        args.do_wrong_thing = config['do_wrong_thing']
        args.obj = config['obj']


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
        args = Argument() 

        args.learning_rate = config['learning_rate']
        args.mbsize = config['mbsize']
        args.opt_beta = config['opt_beta']
        args.opt_beta2 = config['opt_beta2']
        args.opt_epsilon = config['opt_epsilon']
        args.nemb = config['nemb']
        args.min_blocks = config['min_blocks']
        args.max_blocks = config['max_blocks']
        args.num_iterations = config['num_iterations']
        args.num_conv_steps = config['num_conv_steps']
        args.log_reg_c = config['log_reg_c']
        args.reward_exp = config['reward_exp']
        args.reward_norm = config['reward_norm']
        args.sample_prob = config['sample_prob']
        args.R_min = config['R_min']
        args.leaf_coef = config['leaf_coef']
        args.clip_grad = config['clip_grad']
        args.clip_loss = config['clip_loss']
        args.replay_mode = config['replay_mode']
        args.bootstrap_tau = config['bootstrap_tau']
        args.weight_decay = config['weight_decay']
        args.random_action_prob = config['random_action_prob']
        args.array = config['array']
        args.repr_type = config['repr_type']
        args.model_version = config['model_version']
        args.run = config['run']
        args.save_path = config['save_path']
        args.proxy_path = config['proxy_path']
        args.print_array_length = config['print_array_length']
        args.progress = config['progress']
        args.floatX = config['floatX']
        args.include_nblocks = config['include_nblocks']
        args.balanced_loss = config['balanced_loss']
        args.do_wrong_thing = config['do_wrong_thing']
        args.obj = config['obj']



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

