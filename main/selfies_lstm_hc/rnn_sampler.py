import torch

from action_sampler import ActionSampler
from rnn_model import SmilesRnn
from selfies_char_dict import SelfiesCharDictionary
from tdc.chem_utils import MolConvert
converter = MolConvert(src = 'SMILES', dst = 'SELFIES')
# converter = MolConvert(src = 'SELFIES', dst = 'SMILES')
smiles2selfies = MolConvert(src = 'SMILES', dst = 'SELFIES')
selfies2smiles = MolConvert(src = 'SELFIES', dst = 'SMILES')

import os 
path_here = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(path_here,'Voc'), 'r') as fin:
    word_list = fin.readlines() 
vocab_list = [word.strip() for word in word_list]

class SmilesRnnSampler:
    """
    Samples molecules from an RNN smiles language model
    """

    def __init__(self, device: str, batch_size=64) -> None:
        """
        Args:
            device: cpu | cuda
            batch_size: number of concurrent samples to generate
        """
        self.device = device
        self.batch_size = batch_size
        self.sd = SelfiesCharDictionary()

    def sample(self, model: SmilesRnn, num_to_sample: int, max_seq_len=100):
        """

        Args:
            model: RNN to sample from
            num_to_sample: number of samples to produce
            max_seq_len: maximum length of the samples
            batch_size: number of concurrent samples to generate

        Returns: a list of SMILES string, with no beginning nor end symbols

        """
        sampler = ActionSampler(max_batch_size=self.batch_size, max_seq_length=max_seq_len, device=self.device)

        model.eval()
        with torch.no_grad():
            indices = sampler.sample(model, num_samples=num_to_sample)
            selfies = self.sd.matrix_to_smiles(indices)
            # smiles = selfies2smiles(selfies)
            smiles_list, selfies_list = [], []
            for s in selfies:
                try:
                    ss = selfies2smiles(s)
                except:
                    continue 
                smiles_list.append(ss)
                selfies_list.append(s) 

            return selfies_list, smiles_list ### list of selfies; list of smiles 




