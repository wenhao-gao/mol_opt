import os, torch
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
from chemutils import * 
from inference_utils import * 
from online_train import * 
from random import shuffle 
class MIMOSA_Optimizer(BaseOptimizer):

	def __init__(self, args=None):
		super().__init__(args)
		self.model_name = "mimosa"

	def _optimize(self, oracle, config):

		self.oracle.assign_evaluator(oracle)

		population_size = config['population_size']
		lamb = config['lamb']

		start_smiles_lst = ['C1(N)=NC=CC=N1']  ## 'C1=CC=CC=C1NC2=NC=CC=N2' 
		model_ckpt = os.path.join(path_here, "pretrained_model/GNN.ckpt") # mGNN only
		gnn = torch.load(model_ckpt)
		current_set = set(start_smiles_lst)

		patience = 0
		all_smiles_score_list = []  

		while True:

			if len(self.oracle) > 100:
				self.sort_buffer()
				old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
			else:
				old_scores = 0

			next_set = set()
			for smiles in current_set:
				smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)
				next_set = next_set.union(smiles_set)

			smiles_lst = list(next_set)
			shuffle(smiles_lst)
			print(f"Generated molecules: {len(smiles_lst)}")
			smiles_lst = smiles_lst[:config['offspring_size']]

			score_lst = self.oracle(smiles_lst)

			if self.finish:
				print('max oracle hit, abort ...... ')
				break 

			smiles_score_lst = [(smiles, score) for smiles, score in zip(smiles_lst, score_lst)]
			smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
			current_set, _, _ = dpp(smiles_score_lst = smiles_score_lst, num_return = population_size, lamb = lamb) 	# Option II: DPP

			### online train gnn 
			all_smiles_score_list.extend(smiles_score_lst)
			all_smiles_score_list.sort(key=lambda x:x[1], reverse=True)
			# good_smiles_list = [i[0] for i in filter(lambda x:x[1] > 0.5, all_smiles_score_list)]
			# if len(good_smiles_list) < config['train_data_size']:
			# 	good_smiles_list.extend(all_smiles_score_list[:config['train_data_size']])

			# import ipdb; ipdb.set_trace()
			# print(f"Training mol length: {len(good_smiles_list)}")
			good_smiles_list = all_smiles_score_list[:config['train_data_size']]
			train_gnn(good_smiles_list, gnn, epoch=config['train_epoch'])

			# early stopping
			if len(self.oracle) > 2000:
				self.sort_buffer()
				new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
				if new_scores == old_scores:
					patience += 1
					if patience >= self.args.patience:
						self.log_intermediate(finish=True)
						print('convergence criteria met, abort ...... ')
						break
				else:
					patience = 0

