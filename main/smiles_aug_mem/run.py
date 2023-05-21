import torch
import numpy as np
import os

from main.optimizer import BaseOptimizer

from augmented_memory import AugmentedMemory
from utils import get_randomized_smiles, to_tensor


class AugmentedMemory_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        # Your model name
        self.model_name = 'smiles_aug_mem'

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        # set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = torch.cuda.FloatTensor
        torch.set_default_tensor_type(tensor)

        print('----- Initializing SMILES Augmented Memory Model -----')

        model = AugmentedMemory(pretrained_model_path=os.path.join(os.path.dirname(__file__), 'prior/zinc.prior.ckpt'),
                                batch_size=64,
                                sigma=500,
                                replay_buffer_size=100,
                                augmented_memory=True,
                                augmentation_rounds=2,
                                learning_rate=0.0005)

        patience = 0
        while True:
            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            if self.finish:
                print('max oracle hit')
                break

            # sample SMILES
            seqs, smiles, agent_likelihood = model.agent.sample_sequences_and_smiles(batch_size=model.batch_size)
            # switch signs
            agent_likelihood = -agent_likelihood
            # get prior likelihood
            prior_likelihood = -model.prior.likelihood(seqs)
            # get scores
            scores = to_tensor(np.array(self.oracle(smiles))).to(device)
            # get augmented likelihood
            augmented_likelihood = prior_likelihood + model.sigma * scores
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # add "normal" experience replay
            loss, agent_likelihood = self.add_experience_replay(model=model,
                                                                loss=loss,
                                                                agent_likelihood=agent_likelihood,
                                                                prior_likelihood=prior_likelihood,
                                                                smiles=smiles,
                                                                scores=scores,
                                                                override=True)
            loss = loss.mean()
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            # perform augmented memory
            for _ in range(model.augmentation_rounds):
                # augment the *sampled* SMILES
                randomized_smiles_list = get_randomized_smiles(smiles, model.prior)
                # get prior likelihood of randomized SMILES
                prior_likelihood = -model.prior.likelihood_smiles(randomized_smiles_list)
                # get agent likelihood of randomized SMILES
                agent_likelihood = -model.agent.likelihood_smiles(randomized_smiles_list)
                # compute augmented likelihood with the "new" prior likelihood using randomized SMILES
                augmented_likelihood = prior_likelihood + model.sigma * scores
                # compute loss
                loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
                # add augmented experience replay using randomized SMILES
                loss, agent_likelihood = self.add_experience_replay(model=model,
                                                                    loss=loss,
                                                                    agent_likelihood=agent_likelihood,
                                                                    prior_likelihood=prior_likelihood,
                                                                    smiles=smiles,
                                                                    scores=scores,
                                                                    override=False)
                loss = loss.mean()
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()

            # early stopping
            if len(self.oracle) > 1000:
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

    @staticmethod
    def add_experience_replay(model, loss, agent_likelihood, prior_likelihood, smiles, scores, override):
        # use augmented memory
        if model.augmented_memory and not override:
            if len(model.replay_buffer.buffer) == 0:
                return loss, agent_likelihood
            else:
                exp_smiles, exp_scores, exp_prior_likelihood = model.replay_buffer.augmented_memory_replay()
        # sample normally from the replay buffer
        else:
            exp_smiles, exp_scores, exp_prior_likelihood = model.replay_buffer.sample_buffer()
        # concatenate the loss with experience replay SMILES added
        if len(exp_smiles) > 0:
            exp_agent_likelihood = -model.agent.likelihood_smiles(exp_smiles)
            exp_augmented_likelihood = exp_prior_likelihood + model.sigma * to_tensor(exp_scores)
            exp_loss = torch.pow((to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        model.replay_buffer.add_to_buffer(smiles, scores, prior_likelihood)

        return loss, agent_likelihood

