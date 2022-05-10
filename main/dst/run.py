import os, torch
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
from chemutils import * 
from inference_utils import * 
from tqdm import tqdm


class DSToptimizer(BaseOptimizer):

	def __init__(self, args=None):
		super().__init__(args)
		self.model_name = "dst"

	def _optimize(self, oracle, config):

		self.oracle.assign_evaluator(oracle)

		population_size = config['population_size']
		lamb = config['lamb']
		topk = config['topk']
		epsilon = config['epsilon']

		start_smiles_lst = ['C1(N)=NC=CC=N1']
		model_ckpt = os.path.join(path_here, "pretrained_model/qed_epoch_4_iter_0_validloss_0.57661.ckpt")
		gnn = torch.load(model_ckpt)

		current_set = set(start_smiles_lst)

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
			score_lst = self.oracle(smiles_lst)

			if self.finish:
				print('max oracle hit, abort ...... ')
				break
				
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
					if patience >= self.args.patience * 5:
						self.log_intermediate(finish=True)
						print('convergence criteria met, abort ...... ')
						break
				else:
					patience = 0
