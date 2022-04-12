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


import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)

from boss.code.parameters.candidate_parameter import CandidateStringParameter
from boss.code.optimizers.StringGeneticAlgorithmAcquisitionOptimizer import StringGeneticProgrammingOptimizer
from boss.code.emukit_models.emukit_bow_model import BOW_model
from boss.code.emukit_models.emukit_linear_model import linear_model
from boss.code.emukit_models.emukit_ssk_model import SSK_model



# get 250,000 candidate molecules
file = gzip.GzipFile("./example_data/SMILES/SMILES.gzip", 'rb')
data = file.read()
smiles_full = pickle.loads(data)
file.close()
# get their scores
file = gzip.GzipFile("./example_data/SMILES/TARGETS.gzip", 'rb')
data = file.read()
targets_full = pickle.loads(data)
file.close()
# for tutorial only keep strings <40 length (for quick SSK)
smiles=[]
targets=[]
for i in range(0,len(smiles_full)):
    if len(smiles_full[i])<40:
        smiles.append(smiles_full[i])
        targets.append(targets_full[i])
# smiles=np.array(smiles)
# targets=np.array(targets)
smiles = smiles[:25]

property_name = "GSK3B"
from tdc import Oracle 
oracle = Oracle(property_name)
f = property_name + ".txt"
with open(f, 'w') as fout:
    fout.write('0\n')
print(smiles)
ss = oracle(smiles)
#seperate all character with blank space
targets = np.array(ss)
smiles = np.array([" ".join(list(smile)) for smile in smiles]).reshape(-1,1)
print(smiles)
# define an objective function (to be minimized) and space 
# def objective(x):
#     # return score of the molecules
#     # *-1 so we can minimize
#     return -targets[np.where(smiles==x)[0][0]]
# objective=np.vectorize(objective)


smiles_dict = dict() 


def oracle_call(smiles, oracle, smiles_dict):
    if smiles not in smiles_dict:
        smiles_dict[smiles] = oracle(smiles)
    return smiles_dict[smile]

def objective(x):
    x = [''.join(i[0].split()) for i in x]
    with open(f, 'r') as fin:
        line = fin.readlines()[-1] 
        num = int(line.split()[0])
    num += len(x)
    results = oracle(x)
    values = ' '.join([str(i) for i in results])
    smiles = ' '.join(x)
    with open(f, 'a+') as fout:
        fout.write(str(num) + '\t' + values + '\t' + smiles + '\n')
    return - np.array(results).reshape(-1,1)

# define search space
space = ParameterSpace([CandidateStringParameter("string",smiles)])

# collect initial design (uniform sample)
np.random.seed(1234)
random_design = RandomDesign(space)
initial_points_count = 15
X_init = random_design.get_samples(initial_points_count)
Y_init = objective(X_init)




# build BO loop
# fit SSK model
# just a single restart when fitting kernel params for demo 
# (we recommend at least 3 for high performance)
model =SSK_model(space,X_init,Y_init,max_subsequence_length=5,n_restarts=1)
# Load core elements for Bayesian optimization
expected_improvement = ExpectedImprovement(model)
# use random search to optimize acqusition function
optimizer = RandomSearchAcquisitionOptimizer(space,100)
bayesopt_loop_ssk = BayesianOptimizationLoop(model = model, 
                                         space = space,
                                         acquisition = expected_improvement,
                                         acquisition_optimizer = optimizer)
# add loop summary
def summary(loop, loop_state):
    print("Performing BO step {}".format(loop.loop_state.iteration))
bayesopt_loop_ssk.iteration_end_event.append(summary)





# run BO loop for 35 steps 
stopping_condition = FixedIterationsStoppingCondition(i_max = 50000) 
bayesopt_loop_ssk.run_loop(objective, stopping_condition)



# also see performance of random search 
#(starting from the initialization used by the other approaches)
# np.random.seed(1234)
# Y_random=np.vstack([Y_init,objective(random_design.get_samples(35))])

obj = bayesopt_loop_ssk.loop_state.Y 
generated_smiles = bayesopt_loop_ssk.loop_state.X
print(-np.minimum.accumulate(bayesopt_loop_ssk.loop_state.Y))
# print(bayesopt_loop_ssk.loop_state.X)
print(obj[:-50])
pickle.dump((obj, generated_smiles), open(property_name + ".pkl", 'wb'))

# plot results from the two methods
# recall that first 15 points are a random sample shared by all the methods
# plt.plot(-np.minimum.accumulate(bayesopt_loop_ssk.loop_state.Y),label="Split SSk")
# plt.plot(-np.minimum.accumulate(Y_random),label="Random Search")


# plt.ylabel('Current best')
# plt.xlabel('Iteration')
# plt.legend()







