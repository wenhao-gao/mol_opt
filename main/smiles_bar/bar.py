import torch
import pandas as pd

from model import Model
from utils import to_tensor

from copy import deepcopy


class BAR:
    def __init__(self,
                 pretrained_model_path: str,
                 batch_size: int=64,
                 sigma: int=1000,
                 alpha: float=0.25,
                 agent_update_frequency: int=5,
                 replay_buffer_size: int=100,
                 replay_number: int=10,
                 learning_rate: float=0.0005,
                 ):
        self.prior = Model.load_from_file(pretrained_model_path)
        self.agent = Model.load_from_file(pretrained_model_path)
        # keep track of the best Agent
        self.best_agent = deepcopy(self.agent)
        self.batch_size = batch_size
        self.sigma = sigma
        self.alpha = alpha
        self.agent_update_frequency = agent_update_frequency
        self.replay_buffer = ExperienceReplay(memory_size=replay_buffer_size,
                                              replay_number=replay_number)
        self.optimizer = torch.optim.Adam(self.agent.get_network_parameters(), lr=learning_rate)


class ExperienceReplay:
    def __init__(self,
                 memory_size: int=100,
                 replay_number: int=10):
        self.buffer = pd.DataFrame(columns=['smiles', 'likelihood', 'scores'])
        self.memory_size = memory_size
        self.replay_number = replay_number

    def add_to_buffer(self, smiles, scores, neg_likelihood):
        """this method adds new SMILES to the experience replay buffer if they are better scoring"""
        df = pd.DataFrame({"smiles": smiles, "likelihood": neg_likelihood.cpu().detach().numpy(), "scores": scores.cpu().detach().numpy()})
        self.buffer = pd.concat([self.buffer, df])
        self.purge_buffer()

    def purge_buffer(self):
        """
        this method slices the experience replay buffer to keep only
        the top memory_size number of best scoring SMILES
        """
        unique_df = self.buffer.drop_duplicates(subset=["smiles"])
        sorted_df = unique_df.sort_values('scores', ascending=False)
        self.buffer = sorted_df.head(self.memory_size)
        self.buffer = self.buffer.loc[self.buffer['scores'] != 0.0]

    def sample_buffer(self):
        """this method randomly samples replay_number of SMILES from the experience replay buffer"""
        sample_size = min(len(self.buffer), self.replay_number)
        if sample_size > 0:
            sampled = self.buffer.sample(sample_size)
            smiles = sampled["smiles"].values
            scores = sampled["scores"].values
            prior_likelihood = to_tensor(sampled["likelihood"].values)
            return smiles, scores, prior_likelihood
        else:
            return [], [], []








