import os
import torch

from main.optimizer import BaseOptimizer
from main.utils.chem import canonicalize_list

from model.utils import set_default_device_cuda
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
            # Sample
            seqs, smiles, agent_likelihood, probs, log_probs, critic_values = AHC._sample_batch(AHC.batch_size)
            # Penalize non-unique
            #non_unique = []
            #for i, smi in enumerate(smiles):
            #    if smi in self.mol_buffer:
            #        non_unique.append(i)
            # Score
            scores = AHC._score(smiles, step).double()
            #scores[non_unique] = 0.
            # Compute loss
            agent_likelihood = - agent_likelihood
            prior_likelihood = - AHC.prior.likelihood(seqs)
            augmented_likelihood = prior_likelihood + AHC.sigma * scores
            sscore, sscore_idxs = scores.sort(descending=True)
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            loss = loss[sscore_idxs.data[:int(AHC.batch_size * AHC.topk)]]
            AHC._update(loss, verbose=False)
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

