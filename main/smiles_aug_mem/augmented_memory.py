import torch
import pandas as pd
import numpy as np

from model import Model
from utils import get_randomized_smiles, to_tensor


class AugmentedMemory:
    def __init__(self,
                 pretrained_model_path: str,
                 batch_size: int=64,
                 sigma: int=500,
                 replay_buffer_size: int=100,
                 replay_number: int=10,
                 augmented_memory: bool=True,
                 augmentation_rounds: int=2,
                 learning_rate: float=0.0005
                 ):
        self.prior = Model.load_from_file(pretrained_model_path)
        self.agent = Model.load_from_file(pretrained_model_path)
        self.batch_size = batch_size
        self.sigma = sigma
        self.replay_buffer = ExperienceReplay(pretrained_model_path=pretrained_model_path,
                                              memory_size=replay_buffer_size,
                                              replay_number=replay_number)
        self.augmented_memory = augmented_memory
        self.augmentation_rounds = augmentation_rounds
        self.optimizer = torch.optim.Adam(self.agent.get_network_parameters(), lr=learning_rate)


class ExperienceReplay:
    def __init__(self,
                 pretrained_model_path: str,
                 memory_size: int=100,
                 replay_number: int=10):
        self.buffer = pd.DataFrame(columns=['smiles', 'likelihood', 'scores'])
        self.memory_size = memory_size
        self.replay_number = replay_number
        self.prior = Model.load_from_file(pretrained_model_path)

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
        # do not store SMILES with 0 reward
        self.buffer = self.buffer.loc[self.buffer['scores'] != 0.0]

    def augmented_memory_replay(self):
        """this method augments all the SMILES in the replay buffer and returns their likelihoods"""
        if len(self.buffer) > 0:
            smiles = self.buffer["smiles"].values
            # randomize the smiles
            randomized_smiles_list = get_randomized_smiles(smiles, self.prior)
            scores = self.buffer["scores"].values
            prior_likelihood = to_tensor(-self.prior.likelihood_smiles(randomized_smiles_list))
            return randomized_smiles_list, scores, prior_likelihood
        else:
            return [], [], []

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








