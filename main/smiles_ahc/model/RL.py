import os
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from model.model import Model
from model import utils

import pandas as pd


class ReinforcementLearning:
    def __init__(self,
                 device,
                 agent,
                 scoring_function,
                 save_dir,
                 optimizer,
                 learning_rate,
                 is_molscore=True,
                 freeze=None):
        # Device
        self.device = device
        # Load agent
        self.agent = Model.load_from_file(file_path=agent, sampling_mode=False, device=device)
        # Scoring function
        self.scoring_function = scoring_function
        self.molscore = is_molscore
        self.save_dir = save_dir
        # Optimizer
        self.optimizer = optimizer(self.agent.network.parameters(), lr=learning_rate)
        if freeze is not None:
            self._freeze_network(freeze)
        self.record = None

    def train(self, n_steps, save_freq):
        for step in tqdm(range(n_steps), total=n_steps):
            self._train_step(step=step)
            # Save the agent weights every few iterations
            if step % save_freq == 0 and step != 0:
                self.agent.save(os.path.join(self.save_dir, f'Agent_{step}.ckpt'))
        # If the entire training finishes, clean up
        self.agent.save(os.path.join(self.save_dir, f'Agent_{n_steps}.ckpt'))
        return self.record

    def _freeze_network(self, freeze):
        n_freeze = freeze * 4 + 1
        for i, param in enumerate(self.agent.network.parameters()):
            if i < n_freeze:  # Freeze parameter
                param.requires_grad = False

    def _update(self, loss, verbose=False):
        loss = loss.mean()
        if verbose:
            print(f'    Loss: {loss.data:.03f}')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _train_step(self, step):
        raise NotImplementedError

    def _sample_batch(self, batch_size):
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = self.agent.sample_sequences_and_smiles(
            batch_size)
        return seqs, smiles, agent_likelihood, probs, log_probs, critic_values

    def _score(self, smiles, step):
        try:
            scores = self.scoring_function(smiles)
            scores = utils.to_tensor(scores).to(self.device)
        except (Exception, BaseException, SystemExit, KeyboardInterrupt) as e:
            if self.molscore:
                # If anything fails, save smiles, agent, scoring_function etc.
                utils.save_smiles(smiles,
                                  os.path.join(self.save_dir,
                                               f'failed_{self.scoring_function.step}.smi'))
                self.agent.save(os.path.join(self.save_dir, f'Agent_{step}.ckpt'))
                self.scoring_function.write_scores()
                self.scoring_function._write_temp_state(step=self.scoring_function.step)
                self.scoring_function.kill_monitor()
                raise e
            else:
                utils.save_smiles(smiles,
                                  os.path.join(self.save_dir,
                                               f'failed_{step + 1}.smi'))
                self.agent.save(os.path.join(self.save_dir, f'Agent_{step}.ckpt'))
                raise e
        return scores


class AugmentedHillClimb(ReinforcementLearning):
    _short_name = 'AHC'
    def __init__(self, device, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore=True, freeze=None,
                 prior=None, batch_size=64, sigma=60, topk=0.5, **kwargs):
        super().__init__(device, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore, freeze=None)

        # Load prior
        if prior is None:
            self.prior = Model.load_from_file(file_path=agent, sampling_mode=True, device=device)
        else:
            self.prior = Model.load_from_file(file_path=prior, sampling_mode=True, device=device)
        # Parameters
        self.batch_size = batch_size
        self.sigma = sigma
        self.topk = topk
        # Record
        self.record = {'loss': [],
                       'prior_nll': [],
                       'agent_nll': []}

        self.replay_buffer = ExperienceReplay(memory_size=100,
                                              replay_number=10)

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = self._sample_batch(self.batch_size)
        # Score
        scores = self._score(smiles, step)
        # Compute loss
        agent_likelihood = - agent_likelihood
        prior_likelihood = - self.prior.likelihood(seqs)
        augmented_likelihood = prior_likelihood + self.sigma * scores
        sscore, sscore_idxs = scores.sort(descending=True)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        # Update
        self.record['loss'] += list(loss.detach().cpu().numpy())
        self.record['prior_nll'] += list(-prior_likelihood.detach().cpu().numpy())
        self.record['agent_nll'] += list(-agent_likelihood.detach().cpu().numpy())
        loss = loss[sscore_idxs.data[:int(self.batch_size * self.topk)]]
        self._update(loss, verbose=False)

class ExperienceReplay:
    def __init__(self,
                 memory_size: int = 100,
                 replay_number: int = 10):
        self.buffer = pd.DataFrame(columns=['smiles', 'likelihood', 'scores'])
        self.memory_size = memory_size
        self.replay_number = replay_number

    def add_to_buffer(self, smiles, scores, neg_likelihood):
        """this method adds new SMILES to the experience replay buffer if they are better scoring"""
        df = pd.DataFrame({"smiles": smiles, "likelihood": neg_likelihood.cpu().detach().numpy(),
                           "scores": scores.cpu().detach().numpy()})
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
            prior_likelihood = utils.to_tensor(sampled["likelihood"].values)
            return smiles, scores, prior_likelihood
        else:
            return [], [], []



