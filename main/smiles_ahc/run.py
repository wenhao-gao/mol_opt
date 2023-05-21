import os
import torch

from main.optimizer import BaseOptimizer
from main.utils.chem import canonicalize_list

from model.utils import set_default_device_cuda, to_tensor
from model.RL import AugmentedHillClimb


class AHC_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        # Your model name
        self.model_name = "smiles_ahc"

    def _optimize(self, oracle, config):
        """
        The code for a function to optimize a oracle function with hyper-parameters defined in config
        """

        ## This line is necessary
        self.oracle.assign_evaluator(oracle)

        # Setup device
        device = set_default_device_cuda('cuda')

        # Initalization
        AHC = AugmentedHillClimb(
            device=device,
            agent=os.path.join(os.path.dirname(__file__), 'pretrained_model/Prior_ZINC250k_Epoch-5.ckpt'),
            scoring_function=self.oracle, save_dir=self.args.output_dir,
            optimizer=torch.optim.Adam,
            learning_rate=config['learning_rate'],
            is_molscore=False,
            batch_size=config['batch_size'],
            sigma=config['sigma'],
            topk=config['topk']
        )

        step = 0
        converge = False
        patience = 0
        best_score = 0.
        while (not self.finish) and (not converge):

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            # Sample
            seqs, smiles, agent_likelihood, probs, log_probs, critic_values = AHC._sample_batch(AHC.batch_size)
            # Penalize non-unique
            # non_unique = []
            # for i, smi in enumerate(smiles):
            #    if smi in self.mol_buffer:
            #        non_unique.append(i)
            # Score
            scores = AHC._score(smiles, step).double()
            # scores[non_unique] = 0.
            # Compute loss
            agent_likelihood = - agent_likelihood
            prior_likelihood = - AHC.prior.likelihood(seqs)
            augmented_likelihood = prior_likelihood + AHC.sigma * scores
            sscore, sscore_idxs = scores.sort(descending=True)
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            loss = loss[sscore_idxs.data[:int(AHC.batch_size * AHC.topk)]]

            # add experience replay
            if config['experience_replay']:
                loss, agent_likelihood = self.add_experience_replay(model=AHC,
                                                                    loss=loss,
                                                                    agent_likelihood=agent_likelihood,
                                                                    prior_likelihood=prior_likelihood,
                                                                    smiles=smiles,
                                                                    scores=scores)

            AHC._update(loss, verbose=False)

            # --------------------------------------------------------------------------------------
            # copied REINVENT's early stopping mechanism so Augmented Memory, REINVENT, AHC, and BAR
            # are assessed under the same criterion

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= config['patience']:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

            # --------------------------------------------------------------------------------------

            """
            # Update step
            step += 1
            # Update best score
            if float(scores.mean()) > best_score:
                best_score = float(scores.mean())
            else:
                patience += 1
            # Check convergence
            if patience >= config['patience']:
                self.log_intermediate(finish=True)
                break
            """

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

