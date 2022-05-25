import os
import math
import torch
import random
import logging as log
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from torch.utils import data
# from torch.utils.tensorboard import SummaryWriter

from common.train import train
from common.chem import mol_to_dgl
from common.utils import print_mols
from datasets.utils import load_mols
from datasets.datasets import ImitationDataset

class Sampler():
    def __init__(self, config, proposal, oracle):
        self.proposal = proposal
        self.config = config
        self.oracle = oracle 
        
        self.writer = None
        self.run_dir = None
        
        ### for sampling
        self.step = None
        self.PATIENCE = 100
        self.patience = 100
        self.best_eval_res = 0.
        self.best_avg_score = 0.
        self.last_avg_size = 20
        self.train = config['train']
        self.num_mols = config['num_mols']
        # self.num_step = config['num_step']
        self.batch_size = config['batch_size']
        self.fps_ref = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) 
                        for x in config['mols_ref']] if config['mols_ref'] else None

        ### for training editor
        if self.train:
            self.dataset = None
            self.DATASET_MAX_SIZE = config['dataset_size']
            self.optimizer = torch.optim.Adam(self.proposal.editor.parameters(), lr=config['lr'])


    def scores_from_dicts(self, dicts):
        '''
        @params:
            dicts (list): list of score dictionaries
        @return:
            scores (list): sum of property scores of each molecule after clipping
        '''
        scores = []
        # score_norm = sum(self.score_wght.values())
        for score_dict in dicts:
            score = 0.
            for k, v in score_dict.items():
                score += v 
            # score /= score_norm
            score = max(score, 0.)
            scores.append(score)
        return scores

    def record(self, step, old_mols, old_dicts, acc_rates):
        ### average score and size
        old_scores = self.scores_from_dicts(old_dicts)
        avg_score = 1. * sum(old_scores) / len(old_scores)
        sizes = [mol.GetNumAtoms() for mol in old_mols]
        avg_size = sum(sizes) / len(old_mols)
        self.last_avg_size = avg_size

        ### successful rate and uniqueness
        fps_mols, unique = [], set()
        success_dict = {k: 0. for k in old_dicts[0].keys()}
        success, novelty, diversity = 0., 0., 0.
        for i, score_dict in enumerate(old_dicts):
            all_success = True
            for k, v in score_dict.items():
                if v >= self.score_succ[k]:
                    success_dict[k] += 1.
                else: all_success = False
            success += all_success
            if all_success:
                fps_mols.append(old_mols[i])
                unique.add(Chem.MolToSmiles(old_mols[i]))
        success_dict = {k: v / len(old_mols) for k, v in success_dict.items()}
        success = 1. * success / len(old_mols)
        unique = 1. * len(unique) / (len(fps_mols) + 1e-6)

        ### novelty and diversity
        fps_mols = [AllChem.GetMorganFingerprintAsBitVect(
            x, 3, 2048) for x in fps_mols]
        
        if self.fps_ref:
            n_sim = 0.
            for i in range(len(fps_mols)):
                sims = DataStructs.BulkTanimotoSimilarity(
                    fps_mols[i], self.fps_ref)
                if max(sims) >= 0.4: n_sim += 1
            novelty = 1. - 1. * n_sim / (len(fps_mols) + 1e-6)
        else: novelty = 1.
        
        similarity = 0.
        for i in range(len(fps_mols)):
            sims = DataStructs.BulkTanimotoSimilarity(
                fps_mols[i], fps_mols[:i])
            similarity += sum(sims)
        n = len(fps_mols)
        n_pairs = n * (n - 1) / 2
        diversity = 1 - similarity / (n_pairs + 1e-6)
        
        diversity = min(diversity, 1.)
        novelty = min(novelty, 1.)
        evaluation = {
            'success': success,
            'unique': unique,
            'novelty': novelty,
            'diversity': diversity,
            'prod': success * novelty * diversity
        }

        ### logging and writing tensorboard
        # log.info('Step: {:02d},\tScore: {:.7f}'.format(step, avg_score))
        # self.writer.add_scalar('score_avg', avg_score, step)
        # self.writer.add_scalar('size_avg', avg_size, step)
        # self.writer.add_scalars('success_dict', success_dict, step)
        # self.writer.add_scalars('evaluation', evaluation, step)
        # self.writer.add_histogram('acc_rates', torch.tensor(acc_rates), step)
        # self.writer.add_histogram('scores', torch.tensor(old_scores), step)
        # for k in old_dicts[0].keys():
        #     scores = [score_dict[k] for score_dict in old_dicts]
        #     self.writer.add_histogram(k, torch.tensor(scores), step)
        # print_mols(self.run_dir, step, old_mols, old_scores, old_dicts)
        
        ### early stop
        if evaluation['prod'] > .1 and evaluation['prod'] < self.best_eval_res  + .01 and \
                    avg_score > .1 and          avg_score < self.best_avg_score + .01:
            self.patience -= 1
        else: 
            self.patience = self.PATIENCE
            self.best_eval_res  = max(self.best_eval_res, evaluation['prod'])
            self.best_avg_score = max(self.best_avg_score, avg_score)
        
    def acc_rates(self, new_scores, old_scores, fixings):
        '''
        compute sampling acceptance rates
        @params:
            new_scores : scores of new proposed molecules
            old_scores : scores of old molcules
            fixings    : acceptance rate fixing propotions for each proposal
        '''
        raise NotImplementedError

    def sample(self, mols_init):
        '''
        sample molecules from initial ones
        @params:
            mols_init : initial molecules
        '''
        
        ### sample
        old_mols = [mol for mol in mols_init]
        old_dicts = [{} for i in old_mols]
        old_smiles = [Chem.MolToSmiles(mol) for mol in old_mols]
        for ii,smiles in enumerate(old_smiles): 
            value = self.oracle(smiles)
            old_dicts[ii][smiles] = value 
        old_scores = [self.oracle(smiles) for smiles in old_smiles]
        acc_rates = [0. for _ in old_mols]

        step = 1
        patience = 0

        while True:

            if len(self.oracle) > 100:
                self.oracle.sort_buffer()
                old_scores_forst = [item[1][0] for item in list(self.oracle.mol_buffer.items())[:50]]
            else:
                old_scores_forst = 0

            if self.patience <= 0: break
            self.step = step
            print("Proposing new molecules ......")
            new_mols, fixings = self.proposal.propose(old_mols) 

            new_dicts = [{} for i in new_mols]
            new_smiles = [Chem.MolToSmiles(mol) for mol in new_mols]
            for ii,smiles in enumerate(new_smiles):
                value = self.oracle(smiles)
                new_dicts[ii][smiles] = value 
            new_scores = [self.oracle(smiles) for smiles in new_smiles]

            if self.oracle.finish:
                print('max oracle hit, abort ...... ')
                break 
            
            indices = [i for i in range(len(old_mols)) if new_scores[i] > old_scores[i]]
            
            acc_rates = self.acc_rates(new_scores, old_scores, fixings)
            acc_rates = [min(1., max(0., A)) for A in acc_rates]
            for i in range(self.num_mols):
                A = acc_rates[i] # A = p(x') * g(x|x') / p(x) / g(x'|x)
                if random.random() > A: continue
                old_mols[i] = new_mols[i]
                old_scores[i] = new_scores[i]
                old_dicts[i] = new_dicts[i]

            ### train editor
            if self.train and len(self.oracle) > 500:
                dataset = self.proposal.dataset
                dataset = data.Subset(dataset, indices)
                if self.dataset and len(self.dataset) > 0: 
                    # print(dataset)
                    try:
                        self.dataset.merge_(dataset)
                    except:
                        print(f"Problem happned when merging data, pass this round.")
                else: self.dataset = ImitationDataset.reconstruct(dataset)
                n_sample = len(self.dataset)
                if n_sample > 2 * self.DATASET_MAX_SIZE:
                    indices = [i for i in range(n_sample)]
                    random.shuffle(indices)
                    indices = indices[:self.DATASET_MAX_SIZE]
                    self.dataset = data.Subset(self.dataset, indices)
                    self.dataset = ImitationDataset.reconstruct(self.dataset)
                batch_size = int(self.batch_size * 20 / self.last_avg_size)
                log.info('formed a imitation dataset of size %i' % len(self.dataset))
                loader = data.DataLoader(self.dataset,
                    batch_size=batch_size, shuffle=True,
                    collate_fn=ImitationDataset.collate_fn
                )
                
                print('Training ...')
                train(
                    model=self.proposal.editor, 
                    loaders={'dev': loader}, 
                    optimizer=self.optimizer,
                    n_epoch=1,
                    log_every=25,
                    max_step=25,
                    metrics=[
                        'loss', 
                        'loss_del', 'prob_del',
                        'loss_add', 'prob_add',
                        'loss_arm', 'prob_arm'
                    ]
                )
                
                if not self.proposal.editor.device == \
                    torch.device('cpu'):
                    torch.cuda.empty_cache()

                step += 1

            if len(self.oracle) > 100:
                self.oracle.sort_buffer()
                new_scores_forst = [item[1][0] for item in list(self.oracle.mol_buffer.items())[:50]]
                if new_scores_forst == old_scores_forst:
                    patience += 1
                    if patience >= 5:
                        self.oracle.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0


class Sampler_SA(Sampler):

    def __init__(self, config, proposal, oracle,):
        super().__init__(config, proposal, oracle, )
        self.k = 0
        self.step_cur_T = 0
        self.T = self.T_k(self.k)

    # @staticmethod
    def T_k(self, k):
        T_0 = 1. #.1
        BETA = self.config['beta']
        ALPHA = self.config['alpha']
        # return 1. * T_0 / (math.log(k + 1) + 1e-6)
        # return max(1e-6, T_0 - k * BETA)
        return ALPHA ** k * T_0

    def update_T(self):
        STEP_PER_T = 5
        if self.step_cur_T == STEP_PER_T:
            self.k += 1
            self.step_cur_T = 0
            self.T = self.T_k(self.k)
        else: self.step_cur_T += 1
        self.T = max(self.T, 1e-2)
        return self.T
        
    def acc_rates(self, new_scores, old_scores, fixings):
        acc_rates = []
        T = self.update_T()
        # T = 1. / (4. * math.log(self.step + 8.))
        for i in range(self.num_mols):
            # A = min(1., math.exp(1. * (new_scores[i] - old_scores[i]) / T))
            A = min(1., 1. * new_scores[i] / max(old_scores[i], 1e-6))
            A = min(1., A ** (1. / T))
            acc_rates.append(A)
        return acc_rates


class Sampler_MH(Sampler):
    def __init__(self, config, proposal, oracle, ):
        super().__init__(config, proposal, oracle, )
        self.power = 30.
        
    def acc_rates(self, new_scores, old_scores, fixings):
        acc_rates = []
        for i in range(self.num_mols):
            old_score = max(old_scores[i], 1e-5)
            A = ((new_scores[i] / old_score) ** self.power) * fixings[i]
            acc_rates.append(A)
        return acc_rates
    

class Sampler_Recursive(Sampler):
    def __init__(self, config, proposal, oracle, ):
        super().__init__(config, proposal, oracle,)
        
    def acc_rates(self, new_scores, old_scores, fixings):
        acc_rates = []
        for i in range(self.num_mols):
            A = 1.
            acc_rates.append(A)
        return acc_rates







