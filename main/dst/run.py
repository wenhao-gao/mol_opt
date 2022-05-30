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

		model_ckpt = os.path.join(path_here, "pretrained_model/gnn_init.ckpt")
		gnn = torch.load(model_ckpt)
		all_smiles_score_list = []

		population_size = config['population_size']
		lamb = config['lamb']
		topk = config['topk']
		epsilon = config['epsilon']

		start_smiles_lst = ['C1(N)=NC=CC=N1', 'C1(C)=NC=CC=N1', 'C1(C)=CC=CC=C1', 'C1(N)=CC=CC=C1', 'CC', 'C1(C)CCCCC1']

		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		random.seed(self.seed)

		#### screening zinc first to obtain warm start 
		shuffle(self.all_smiles)
		warmstart_smiles_lst = self.all_smiles[:2000] 
		warmstart_smiles_score = self.oracle(warmstart_smiles_lst)
		warmstart_smiles_score_lst = list(zip(warmstart_smiles_lst, warmstart_smiles_score))
		warmstart_smiles_score_lst.sort(key=lambda x:x[1], reverse = True)  #### [(smiles1, score1), (smiles2, score2), ... ] 
		all_smiles_score_list.extend(warmstart_smiles_score_lst)
		train_gnn(all_smiles_score_list, gnn)
		print('##### train GNN ######')


		init_smiles_lst = start_smiles_lst + [i[0] for i in warmstart_smiles_score_lst[:50]]
		current_set = set(init_smiles_lst)
		patience = 0

		while True:

			### save results after 5k call to observe results earlier  
			if len(self.oracle) >= 5000:
				self.save_result(self.model_name + "_" + oracle.name + "_" + str(self.seed))

			if len(self.oracle) > 100:
				self.sort_buffer()
				old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
			else:
				old_scores = 0

			next_set = set() 
			print('Sampling from current state') #### most time consuming 
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

			###### train GNN online 
			print('##### train GNN online ######')
			all_smiles_score_list.extend(list(zip(smiles_lst, score_lst)))
			train_gnn(all_smiles_score_list, gnn)
				
			smiles_score_lst = [(smiles, score) for smiles, score in zip(smiles_lst, score_lst)]
			smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
			print(smiles_score_lst[:5], "Oracle num: ", len(self.oracle))

			# current_set = [i[0] for i in smiles_score_lst[:population_size]]  # Option I: top-k 
			print('diversify molecules ...')
			current_set, _, _ = dpp(smiles_score_lst = smiles_score_lst, num_return = population_size, lamb = lamb) # Option II: DPP



			### early stopping
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
