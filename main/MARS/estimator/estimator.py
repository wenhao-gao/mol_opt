import torch
import torch.nn.functional as F
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from torch.utils.data import DataLoader

# from .scorer import chemprop_scorer
from .scorer.scorer import get_scores
from common.chem import mol_to_dgl
from datasets.datasets import GraphDataset


class Estimator():
    def __init__(self, config, mols_ref=None):
        '''
        @params:
            config (dict): configurations
        '''
        # chemprop_scorer.device = config['device']
        self.batch_size = config['batch_size']
        self.objectives = config['objectives']
        self.fps_ref = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) 
                        for x in config['mols_ref']] if config['mols_ref'] else None

    def get_scores(self, mols):
        '''
        @params:
            mols: molecules to estimate score
        @return:
            dicts (list): list of score dictionaries
        '''
        dicts = [{} for _ in mols]
        for obj in self.objectives:
            scores = get_scores(obj, mols)
            for i, mol in enumerate(mols):
                dicts[i][obj] = scores[i]
        return dicts
