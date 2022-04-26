import os, pickle, torch, random, argparse
import yaml
import numpy as np 
from tqdm import tqdm 
from tdc import Oracle
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tdc import Oracle
from utils.parsing import parse_args
from agents.agent import DQN 

class MolDQN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "molDQN"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        agent = DQN(
            oracle=self.oracle,
            q_fn = config['q_function'], 
            args=config,
        )
        agent.train()




"""
TASK='test'
HPARAMS='agents/configs/test.json'
CUDA_VISIBLE_DEVICES=0 python -u run_dqn.py \
    -t ${TASK} \
    -c ${HPARAMS} \
    -o qed \
    --synthesizability smi \
    --q_function mlp


"""

