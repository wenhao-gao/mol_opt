import os, torch
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
from chemutils import * 
from inference_utils import * 
from tqdm import tqdm
from chemutils import smiles2graph, vocabulary 
from online_train import train_gnn
from random import shuffle 

class DST_Optimizer(BaseOptimizer):

	def __init__(self, args=None):
		super().__init__(args)
		self.model_name = "dst"

	def _optimize(self, oracle, config):

		self.oracle.assign_evaluator(oracle)

		population_size = config['population_size']
		lamb = config['lamb']
		topk = config['topk']
		epsilon = config['epsilon']

		start_smiles_lst = ['C1(N)=NC=CC=N1', 'C1(C)=NC=CC=N1', 'C1(C)=CC=CC=C1', 'C1(N)=CC=CC=C1', 'CC', 'C1(C)CCCCC1']
		for smiles in start_smiles_lst:
			# print(smiles)
			assert is_valid(smiles)

		model_ckpt = os.path.join(path_here, "pretrained_model/gnn_init.ckpt")
		gnn = torch.load(model_ckpt)

		current_set = set(start_smiles_lst)
		all_smiles_score_list = []
		patience = 0

		while True:

			if len(self.oracle) > 100:
				self.sort_buffer()
				old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
			else:
				old_scores = 0

			next_set = set() 
			print('Sampling from current state')
			for smiles in tqdm(current_set):
				try:
					if substr_num(smiles) < 3: #### short smiles
						smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)  ### optimize_single_molecule_one_iterate_v2
					else:
						smiles_set = optimize_single_molecule_one_iterate_v3(smiles, gnn, topk = topk, epsilon = epsilon)
					next_set = next_set.union(smiles_set)
				except:
					pass 

			smiles_lst = list(next_set)
			shuffle(smiles_lst)
			smiles_lst = smiles_lst[:config['pool_size']] + start_smiles_lst  ### at most XXX mols per generation
			smiles_lst = list(filter(is_valid, smiles_lst))
			score_lst = self.oracle(smiles_lst)

			if self.finish:
				print('max oracle hit, abort ...... ')
				break

			all_smiles_score_list.extend(list(zip(smiles_lst, score_lst)))
			train_gnn(all_smiles_score_list, gnn)
				
			smiles_score_lst = [(smiles, score) for smiles, score in zip(smiles_lst, score_lst)]
			smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
			print(smiles_score_lst[:5], "Oracle num: ", len(self.oracle))

			# current_set = [i[0] for i in smiles_score_lst[:population_size]]  # Option I: top-k 
			print('DPP ing ...')
			current_set, _, _ = dpp(smiles_score_lst = smiles_score_lst, num_return = population_size, lamb = lamb) # Option II: DPP

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
