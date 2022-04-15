import os, pickle, torch, random, argparse
import yaml
import numpy as np 
from tqdm import tqdm 
torch.manual_seed(1)
np.random.seed(2)
random.seed(1)
from tdc import Oracle
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from tdc.generation import MolGen
from random import shuffle, choice  

class SMILES_VAEBO_optimizer(BaseOptimizer):

	def __init__(self, args=None):
		super().__init__(args)
		self.model_name = "smiles_vae_bo"

	def _optimize(self, oracle, config):

		self.oracle.assign_evaluator(oracle)

		## 0. load vae model & get training data
		# import ipdb; ipdb.set_trace()
		vae_model = torch.load(config['save_model'])
		data = MolGen(name = 'ZINC')
		smiles_lst = data.get_data()['smiles'].tolist() 
		shuffle(smiles_lst)
		train_smiles_lst = smiles_lst[:config['train_num']]
		y = self.oracle(train_smiles_lst)
		train_X = []
		for smiles in train_smiles_lst:
			x = vae_model.string2tensor(smiles)
			x = x.unsqueeze(0)
			z, _ = vae_model.forward_encoder(x) ### z: (1,d)
			reconstruct_smiles = vae_model.decoder_z(z)
			train_X.append(z)
		train_X = torch.cat(train_X, dim=0)
		# train_X = torch.FloatTensor(train_X)
		train_X = train_X.detach()
		train_Y = torch.FloatTensor(y).view(-1,1)

		for i in tqdm(range(config['iter_num'])):
			# 1. Fit a Gaussian Process model to data
			# print(torch.min(train_Y), torch.max(train_Y))
			gp = SingleTaskGP(train_X, train_Y)
			mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
			fit_gpytorch_model(mll)

			# 2. Construct an acquisition function
			UCB = UpperConfidenceBound(gp, beta=0.1) 

			# 3. Optimize the acquisition function 
			bounds = torch.stack([torch.min(train_X, 0)[0], torch.max(train_X, 0)[0]])
			z, acq_value = optimize_acqf(
		    	UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
			)
			# print(candidate.shape, acq_value.shape)

			new_smiles = vae_model.decoder_z(z)
			new_score = self.oracle(new_smiles)
			if new_score == 0:
				new_smiles = choice(smiles_lst)
				new_score = self.oracle(new_smiles)				

			new_score = torch.FloatTensor([new_score]).view(-1,1)

			train_X = torch.cat([train_X, z], dim = 0)
			train_Y = torch.cat([train_Y, new_score], dim = 0)
			if train_X.shape[0] > 100:
				train_X = train_X[-config['train_num']:]
				train_Y = train_Y[-config['train_num']:]
			if self.finish:
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
    parser.add_argument('--max_oracle_calls', type=int, default=150)
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
        optimizer = smiles_VAEBO_optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
	main() 






# # 0. data 
# train_X = torch.rand(10, 2)
# Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
# Y += 0.1 * torch.rand_like(Y)
# train_Y = (Y - Y.mean()) / Y.std()
# # print(train_X.shape, train_Y.shape, train_X, train_Y)
# print(train_X.shape, train_Y.shape)


# # 1. Fit a Gaussian Process model to data
# gp = SingleTaskGP(train_X, train_Y)
# mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
# fit_gpytorch_model(mll)


# # 2. Construct an acquisition function
# UCB = UpperConfidenceBound(gp, beta=0.1) 


# # 3. Optimize the acquisition function 
# bounds = torch.stack([torch.zeros(2), torch.ones(2)])
# candidate, acq_value = optimize_acqf(
#     UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
# )

# print(candidate.shape, acq_value.shape)

