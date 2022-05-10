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
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
print("path:", '/'.join(path_here.rstrip('/').split('/')[:-2]))
print(sys.path)
from main.optimizer import BaseOptimizer
import time
from train_agent import train_agent


class REINVENT_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "reinvent"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        restore_prior_from=os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from=restore_prior_from 

        # train_agent(**arg_dict)
        mol_buffer = train_agent(restore_prior_from=restore_prior_from,
                restore_agent_from=restore_agent_from,
                scoring_function=self.oracle,  ### 'tanimoto'
                scoring_function_kwargs=dict(),
                save_dir=None, 
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'], 
                n_steps=config['n_steps'],
                num_processes=0, 
                sigma=config['sigma'],
                experience_replay=0)






