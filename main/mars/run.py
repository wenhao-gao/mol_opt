import os
import torch
import random
from rdkit import Chem, RDLogger
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
from proposal.models.editor_basic import BasicEditor
from proposal.proposal import Proposal_Random, Proposal_Editor, Proposal_Mix
from sampler import Sampler_SA, Sampler_MH, Sampler_Recursive
from datasets.utils import load_mols

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

class MARS_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "mars"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        config['device'] = torch.device(config['device'])

        ### estimator
        # if config['mols_ref']: 
        #     config['mols_ref'] = load_mols(config['data_dir'], config['mols_ref'])

        ### proposal
        editor = BasicEditor(config).to(config['device']) if not config['proposal'] == 'random' else None
        if config['editor_dir'] is not None: # load pre-trained editor
            path = os.path.join(config['root_dir'], config['editor_dir'], 'model_best.pt')
            editor.load_state_dict(torch.load(path, map_location=torch.device(config['device'])))
            print('successfully loaded editor model from %s' % path)
        if config['proposal'] == 'random': proposal = Proposal_Random(config)
        elif config['proposal'] == 'editor': proposal = Proposal_Editor(config, editor)
        elif config['proposal'] == 'mix': proposal = Proposal_Mix(config, editor)

        ### sampler
        if config['sampler'] == 're': sampler = Sampler_Recursive(config, proposal, self.oracle) 
        elif config['sampler'] == 'sa': sampler = Sampler_SA(config, proposal, self.oracle)
        elif config['sampler'] == 'mh': sampler = Sampler_MH(config, proposal, self.oracle)

        ### sampling
        if config['mols_init']:
            mols = load_mols(config['data_dir'], config['mols_init'])
            mols = random.choices(mols, k=config['num_mols'])
            mols_init = mols[:config['num_mols']]
        else: 
            # mols_init = [Chem.MolFromSmiles('CC') for _ in range(config['num_mols'])]
            half = int(config["num_mols"]/5)
            mols_init = [Chem.MolFromSmiles(smi) for smi in self.all_smiles[:half]] + [Chem.MolFromSmiles('CC') for _ in range(config['num_mols'] - half)]

        sampler.sample(mols_init)
