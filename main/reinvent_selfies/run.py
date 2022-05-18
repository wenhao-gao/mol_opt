import os
import sys
import numpy as np
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from main.optimizer import BaseOptimizer
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, seq_to_selfies 
from model import RNN
from data_structs import Vocabulary, Experience
import torch

from tdc.chem_utils import MolConvert
selfies2smiles = MolConvert(src = 'SELFIES', dst = 'SMILES')
smiles2selfies = MolConvert(src = 'SMILES', dst = 'SELFIES')


class REINVENT_SELFIES_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "reinvent_selfies"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from=os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from=restore_prior_from 
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        Prior = RNN(voc)
        Agent = RNN(voc)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here,'data/Prior.ckpt')))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from))
        else:
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        experience = Experience(voc)

        print("Model initialized, starting training...")

        step = 0
        patience = 0

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0
            
            # Sample from Agent
            seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'])

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]

            # Get prior likelihood and score
            prior_likelihood, _ = Prior.likelihood(Variable(seqs))
            ##### original 
            # smiles = seq_to_smiles(seqs, voc) #################### matrix (seq) -> smiles_list
            # score = np.array(self.oracle(smiles))
            ##### original 

            ##### new 
            selfies_list = seq_to_selfies(seqs, voc) 
            smiles_list = selfies2smiles(selfies_list)
            score = np.array(self.oracle(smiles_list))
            ##### new 

            if self.finish:
                print('max oracle hit')
                break 

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience*2:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

            # Calculate augmented likelihood
            augmented_likelihood = prior_likelihood.float() + config['sigma'] * Variable(score).float()
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

            # Experience Replay
            # First sample
            if config['experience_replay'] and len(experience)>config['experience_replay']:
                # exp_seqs, exp_score, exp_prior_likelihood = experience.sample(config['experience_replay']) #### old ---- bug  
                exp_seqs, exp_score, exp_prior_likelihood = experience.sample_selfies(config['experience_replay']) #### new ---- 
                exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
                exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
                exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

            # Then add new experience
            prior_likelihood = prior_likelihood.data.cpu().numpy()
            # print(smiles) ## list of smiles 
            # new_experience = zip(smiles, score, prior_likelihood) ## old === bugs: oov string 
            new_experience = zip(selfies_list, score, prior_likelihood) ## new 
            experience.add_experience(new_experience)

            # Calculate loss
            loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = - (1 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p

            # Calculate gradients and make an update to the network weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Convert to numpy arrays so that we can print them
            augmented_likelihood = augmented_likelihood.data.cpu().numpy()
            agent_likelihood = agent_likelihood.data.cpu().numpy()

            step += 1

