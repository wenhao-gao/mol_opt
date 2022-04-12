import pickle, re, gzip, os, emukit
import numpy as np, matplotlib.pyplot as plt
from emukit.core.initial_designs import RandomDesign
from emukit.core import ParameterSpace
from emukit.core.optimization import RandomSearchAcquisitionOptimizer
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.loop import FixedIterationsStoppingCondition
import warnings
warnings.filterwarnings('ignore')
from random import shuffle 
from tdc import Oracle
import sys, argparse, yaml   
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer

from boss.code.parameters.candidate_parameter import CandidateStringParameter
from boss.code.optimizers.StringGeneticAlgorithmAcquisitionOptimizer import StringGeneticProgrammingOptimizer
from boss.code.emukit_models.emukit_bow_model import BOW_model
from boss.code.emukit_models.emukit_linear_model import linear_model
from boss.code.emukit_models.emukit_ssk_model import SSK_model



class BOSSoptimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "boss"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)

        def objective(x):
            x = [''.join(i[0].split()) for i in x]
            results = self.oracle(x)
            return - np.array(results).reshape(-1,1)


        # get 250,000 candidate molecules
        # file = gzip.GzipFile(os.path.join(path_here, "./example_data/SMILES/SMILES.gzip"), 'rb')
        # data = file.read()
        # smiles_full = pickle.loads(data)
        # file.close()
        # print(smiles_full[:10], targets_full[:10])
        # for tutorial only keep strings <40 length (for quick SSK)
        # smiles=[]
        # for i in range(0,len(smiles_full)):
        #     if len(smiles_full[i])<40:
        #         smiles.append(smiles_full[i])
        # smiles=np.array(smiles)
        with open(os.path.join(path_here, "smiles.txt"), 'r') as fin:
            lines = fin.readlines()
        all_smiles_lst = [line.strip() for line in lines] 
        batch_size = config['batch_size'] 
        num_of_trials = int(config['max_oracle_calls'] / config['batch_size'])


        for ii in range(num_of_trials):
            shuffle(all_smiles_lst)
            smiles_lst = all_smiles_lst[:config['initial_points_count_single_batch']]
            ss = self.oracle(smiles_lst)
            #seperate all character with blank space
            targets = np.array(ss)
            smiles_lst = np.array([" ".join(list(smile)) for smile in smiles_lst]).reshape(-1,1)
            # print("length of initial data", len(smiles)) 
            # define an objective function (to be minimized) and space 
            # def objective(x):
            #     # return score of the molecules
            #     # *-1 so we can minimize
            #     return -targets[np.where(smiles==x)[0][0]]
            # objective=np.vectorize(objective)

            # define search space
            space = ParameterSpace([CandidateStringParameter("string", smiles_lst)])

            # collect initial design (uniform sample)
            np.random.seed(1234)
            random_design = RandomDesign(space)
            X_init = random_design.get_samples(config['initial_points_count_single_batch'])
            Y_init = objective(X_init)


            # build BO loop
            # fit SSK model
            # just a single restart when fitting kernel params for demo 
            # (we recommend at least 3 for high performance)
            model = SSK_model(space, X_init, Y_init, max_subsequence_length=5, n_restarts=1)
            # Load core elements for Bayesian optimization
            expected_improvement = ExpectedImprovement(model)
            # use random search to optimize acqusition function
            optimizer = RandomSearchAcquisitionOptimizer(space, 100)
            bayesopt_loop_ssk = BayesianOptimizationLoop(model = model, 
                                                     space = space,
                                                     acquisition = expected_improvement,
                                                     acquisition_optimizer = optimizer)
            # add loop summary
            def summary(loop, loop_state):
                print("Performing BO step {}".format(loop.loop_state.iteration))
            bayesopt_loop_ssk.iteration_end_event.append(summary)


            # run BO loop for 35 steps 
            stopping_condition = FixedIterationsStoppingCondition(i_max = config['batch_size'] - config['initial_points_count_single_batch']) 
            bayesopt_loop_ssk.run_loop(objective, stopping_condition)

            # also see performance of random search 
            #(starting from the initialization used by the other approaches)
            # np.random.seed(1234)
            # Y_random=np.vstack([Y_init,objective(random_design.get_samples(35))])

            # obj = bayesopt_loop_ssk.loop_state.Y 
            generated_smiles = bayesopt_loop_ssk.loop_state.X
            generated_smiles = generated_smiles.tolist()
            generated_smiles = [''.join(i[0].split()) for i in generated_smiles]
            print(len(generated_smiles))
            print(len(self.oracle)) 
            values = self.oracle(generated_smiles)
            print(len(self.oracle)) 





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=500)
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
        optimizer = BOSSoptimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)



if __name__ == "__main__":
    main() 







