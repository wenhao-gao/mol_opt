import os, pickle, torch, random
import numpy as np 
import argparse
from time import time
from tqdm import tqdm 
from matplotlib import pyplot as plt
from random import shuffle 
import torch.nn as nn
import torch.nn.functional as F
from tdc import Oracle
torch.manual_seed(1)
np.random.seed(2)
random.seed(1)
import sys, yaml 
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
from chemutils import * 
from inference_utils import * 


class MIMOSA_Optimizer(BaseOptimizer):

	def __init__(self, args=None):
		super().__init__(args)
		self.model_name = "MIMOSA"

	def _optimize(self, oracle, config):
		self.oracle.assign_evaluator(oracle)
		max_n_oracles = config["max_n_oracles"]
		max_generations = config["max_generations"]
		population_size = config['population_size']
		lamb = config['lamb']
		topk = config['topk']

		start_smiles_lst = ['C1(N)=NC=CC=N1']  ## 'C1=CC=CC=C1NC2=NC=CC=N2' 
		model_ckpt = os.path.join(path_here, "pretrained_model/GNN.ckpt")
		gnn = torch.load(model_ckpt)
		current_set = set(start_smiles_lst)
		### logging intermediate
		# self.log_intermediate(population_mol, population_scores)

		for i_gen in tqdm(range(max_generations)):
			next_set = set()
			for smiles in current_set:
				smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)
				next_set = next_set.union(smiles_set)

			smiles_lst = list(next_set)
			score_lst = self.oracle(smiles_lst)
			smiles_score_lst = [(smiles, score) for smiles, score in zip(smiles_lst, score_lst)]
			smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
			print(smiles_score_lst[:5], "Oracle num: ", len(self.mol_buffer))
			current_set, _, _ = dpp(smiles_score_lst = smiles_score_lst, num_return = population_size, lamb = lamb) 	# Option II: DPP
			if self.oracle.finish:
				break 



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--smi_file', default=None)
	parser.add_argument('--config_default', default='hparams_default.yaml')
	parser.add_argument('--config_tune', default='hparams_tune.yaml')
	parser.add_argument('--n_jobs', type=int, default=-1)
	parser.add_argument('--output_dir', type=str, default=None)
	parser.add_argument('--patience', type=int, default=5)
	parser.add_argument('--n_runs', type=int, default=5)
	parser.add_argument('--max_oracle_calls', type=int, default=1000)
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

		oracle = Oracle(name = oracle_name)
		optimizer = MIMOSA_Optimizer(args=args)

		if args.task == "simple":
			optimizer.optimize(oracle=oracle, config=config_default)
		elif args.task == "tune":
			optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
		elif args.task == "production":
			optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)
 


if __name__ == "__main__":
	main() 







