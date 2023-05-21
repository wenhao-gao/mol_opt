import torch
import numpy as np
import os

from main.optimizer import BaseOptimizer

from bar import BAR
from utils import to_tensor

from copy import deepcopy


class BAR_Optimizer(BaseOptimizer):
    """
    implementation built on https://github.com/MorganCThomas/SMILES-RNN
    original paper: https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00838
    """

    def __init__(self, args=None):
        super().__init__(args)
        # Your model name
        self.model_name = 'smiles_bar'

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        # set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = torch.cuda.FloatTensor
        torch.set_default_tensor_type(tensor)

        print('----- Initializing SMILES Best Agent Reminder (BAR) Model -----')

        model = BAR(pretrained_model_path=os.path.join(os.path.dirname(__file__), 'prior/zinc.prior.ckpt'),
                    batch_size=64,
                    sigma=1000,
                    alpha=0.25,
                    agent_update_frequency=5,
                    replay_buffer_size=100,
                    replay_number=10,
                    learning_rate=0.0005)

        patience = 0
        epoch = 1
        # keep track of the best average score as update criterion for the best Agent
        best_average_score = 0

        while True:
            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            if self.finish:
                print('max oracle hit')
                break

            # sample batch from current Agent
            seqs, smiles, agent_likelihood = model.agent.sample_sequences_and_smiles(batch_size=model.batch_size)

            # sample batch from best Agent
            best_seqs, best_smiles, best_agent_likelihood = model.best_agent.sample_sequences_and_smiles(batch_size=model.batch_size)

            # score current Agent SMILES
            scores = to_tensor(np.array(self.oracle(smiles))).to(device)

            # score best Agent SMILES
            best_scores = to_tensor(np.array(self.oracle(best_smiles))).to(device)

            # compute loss between Prior and current Agent
            agent_likelihood = -agent_likelihood
            prior_likelihood = -model.prior.likelihood(seqs)
            augmented_likelihood = prior_likelihood + model.sigma * to_tensor(scores)
            current_agent_loss = (1 - model.alpha) * torch.pow((augmented_likelihood - agent_likelihood), 2).mean()

            # compute loss between the best Agent and current Agent
            best_agent_likelihood = -best_agent_likelihood
            # this is the likelihood of the SMILES sampled by the *BAR Agent* as computed by the *current* Agent
            current_agent_likelihood = -model.agent.likelihood(best_seqs)
            best_augmented_likelihood = best_agent_likelihood + model.sigma * to_tensor(best_scores)
            best_agent_loss = model.alpha * torch.pow((best_augmented_likelihood - current_agent_likelihood), 2)

            # add experience replay
            # pass the current Agent and current Agent's sampled SMILES because we want to update this Agent
            # the only thing we are passing that belongs to the best Agent is the best Agent loss
            best_agent_loss, best_agent_likelihood = self.add_experience_replay(model=model,
                                                                                loss=best_agent_loss,
                                                                                agent_likelihood=agent_likelihood,
                                                                                prior_likelihood=prior_likelihood,
                                                                                smiles=smiles,
                                                                                scores=scores)

            # add experience replay above before taking the mean
            best_agent_loss = best_agent_loss.mean()

            # compute the BAR loss
            BAR_loss = current_agent_loss + best_agent_loss

            model.optimizer.zero_grad()
            BAR_loss.backward()
            model.optimizer.step()

            if epoch % model.agent_update_frequency == 0:
                current_average_score = scores.mean().detach().cpu().numpy()
                if current_average_score > best_average_score:
                    print('----- New Best Agent -----')
                    best_average_score = current_average_score
                    # new best Agent
                    model.best_agent = deepcopy(model.agent)

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

            epoch += 1

    @staticmethod
    def add_experience_replay(model, loss, agent_likelihood, prior_likelihood, smiles, scores):
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


