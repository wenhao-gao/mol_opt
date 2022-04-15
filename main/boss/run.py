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
from main.boss.code.parameters.candidate_parameter import CandidateStringParameter
from main.boss.code.optimizers.StringGeneticAlgorithmAcquisitionOptimizer import StringGeneticProgrammingOptimizer
from main.boss.code.emukit_models.emukit_bow_model import BOW_model
from main.boss.code.emukit_models.emukit_linear_model import linear_model
from main.boss.code.emukit_models.emukit_ssk_model import SSK_model

from main.optimizer import BaseOptimizer


class BOSS_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "boss"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        def objective(x):
            x = [''.join(i[0].split()) for i in x]
            results = self.oracle(x)
            return - np.array(results).reshape(-1,1)

        all_smiles_lst = self.all_smiles

        patience = 0

        while True:

            if len(self.oracle) > 50:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:50]]
            else:
                old_scores = 0
            
            shuffle(all_smiles_lst)
            smiles_lst = all_smiles_lst[:config['initial_points_count_single_batch']]
            ss = self.oracle(smiles_lst)
            #seperate all character with blank space
            targets = np.array(ss)
            smiles_lst = np.array([" ".join(list(smile)) for smile in smiles_lst]).reshape(-1,1)

            # define search space
            space = ParameterSpace([CandidateStringParameter("string", smiles_lst)])

            # collect initial design (uniform sample)
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


            # run BO loop
            stopping_condition = FixedIterationsStoppingCondition(i_max = config['batch_size'] - config['initial_points_count_single_batch']) 
            bayesopt_loop_ssk.run_loop(objective, stopping_condition)

            # also see performance of random search 
            #(starting from the initialization used by the other approaches)

            # obj = bayesopt_loop_ssk.loop_state.Y 
            generated_smiles = bayesopt_loop_ssk.loop_state.X
            generated_smiles = generated_smiles.tolist()
            generated_smiles = [''.join(i[0].split()) for i in generated_smiles]
            values = self.oracle(generated_smiles)

            # early stopping
            if len(self.oracle) > 50:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:50]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        break
                else:
                    patience = 0
            
            if self.finish:
                print('max oracle hit, abort ...... ')
                break 
