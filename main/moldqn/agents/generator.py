from typing import List, Optional
from agents.agent import DQN

class DQNDirectedGenerator(object):
    """
    The class that wrap the molecular value learning model, with compatibility with Guacamol benchmarks
    """

    def __init__(self,
                 task_name='dqn',
                 q_fn='morgan_mlp',
                 objective=None,
                 hparams='dqn/configs/test.json',
                 args=None,
                 param=None,
                 model_path='./checkpoints',
                 gen_file='./mol_gen.csv',
                 **kwargs):
        super(DQNDirectedGenerator, self).__init__()
        self.task_name = task_name
        self.q_fn = q_fn
        self.objective = objective
        self.args = args
        self.param = param
        self.path = model_path
        self.gen_file = gen_file


    def generate_optimized_molecules_reward(self, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        """
        The one only returns all information.

        :param scoring_function: The objective function
        :param number_molecules: Number of molecules to generate.
        :param starting_population: The initial data set to start with, can be none.
        :return: Molecules generated.
        """
        agent = DQN(
            task=self.task_name,
            q_fn=self.q_fn,
            objective=self.objective,
            score_fn=scoring_function,
            args=self.args,
            keep=number_molecules,
            param=self.param,
            model_path=self.path,
            gen_file=self.gen_file
        )
        generated_mols = agent.train()
        return generated_mols

