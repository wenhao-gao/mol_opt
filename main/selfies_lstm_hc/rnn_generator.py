import logging
import time
from functools import total_ordering
from typing import List, Set
from rdkit import Chem
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm 
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize_list

from rnn_model import SmilesRnn
from rnn_sampler import SmilesRnnSampler
from rnn_trainer import SmilesRnnTrainer
from rnn_utils import get_tensor_dataset, load_selfies_from_list

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from tdc.chem_utils import MolConvert
smiles2selfies = MolConvert(src = 'SMILES', dst = 'SELFIES')
selfies2smiles = MolConvert(src = 'SELFIES', dst = 'SMILES')

@total_ordering
class OptResult:
    def __init__(self, smiles: str, selfies: str, score: float) -> None:
        self.smiles = smiles
        self.selfies = selfies 
        self.score = score

    def __eq__(self, other):
        return (self.score, self.smiles) == (other.score, other.smiles)

    def __lt__(self, other):
        return (self.score, self.smiles) < (other.score, other.smiles)


import os 
path_here = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(path_here,'Voc'), 'r') as fin:
    word_list = fin.readlines() 
vocab_list = [word.strip() for word in word_list]


class SmilesRnnMoleculeGenerator:
    """
    character-based RNN language model optimized by with hill-climbing

    """

    def __init__(self, model: SmilesRnn, max_len: int, device: str) -> None:
        """
        Args:
            model: Pre-trained SmilesRnn
            max_len: maximum SMILES length
            device: 'cpu' | 'cuda'
        """

        self.device = device
        self.model = model

        lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        #### Key I
        self.sampler = SmilesRnnSampler(device=self.device, batch_size=512)  #### would call "self.model" 
        self.max_len = max_len

        #### Key II 
        self.trainer = SmilesRnnTrainer(model=self.model,
                                        criteria=[self.criterion],
                                        optimizer=self.optimizer,
                                        device=self.device)

    def optimise(self, objective, start_population, keep_top, n_epochs, mols_to_sample,
                 optimize_n_epochs, optimize_batch_size, pretrain_n_epochs) -> List[OptResult]:
        """
        Takes an objective and tries to optimise it
        :param objective: MPO
        :param start_population: Initial compounds (list of smiles) or request new (random?) population
        :param kwargs need to contain:
                keep_top: number of molecules to keep at each iterative finetune step
                mols_to_sample: number of molecules to sample at each iterative finetune step
                optimize_n_epochs: number of episodes to finetune
                optimize_batch_size: batch size for fine-tuning
                pretrain_n_epochs: number of epochs to pretrain on start population
        :return: Candidate molecules
        """

        ### start_population is smiles list 
        int_results = self.pretrain_on_initial_population(objective, start_population,
                                                          pretrain_epochs=pretrain_n_epochs)
        print("--- pretrain on initial population finished ----")

        results: List[OptResult] = []
        results_selfies = []
        seen: Set[str] = set()
        seen_selfies = set()

        for k in int_results:
            if k.smiles not in seen:
                results.append(k)
                results_selfies.append(k.selfies)
                seen.add(k.smiles)
                seen_selfies.add(k.selfies)

        for epoch in tqdm(range(1, 1 + n_epochs)):

            t0 = time.time()
            selfies, smiles = self.sampler.sample(self.model, mols_to_sample, max_seq_len=self.max_len)
            print(smiles[:5])
            ### list_of_selfies list_of_smiles
            t1 = time.time()
            selfi2smile = {selfie:smile for selfie, smile in zip(selfies, smiles)}
            smile2selfi = {smile:selfie for selfie, smile in zip(selfies, smiles)}

            canonicalized_samples = set(canonicalize_list(smiles, include_stereocenters=True))
            # payload = list(canonicalized_samples.difference(seen))
            payload = list(canonicalized_samples)
            payload.sort()  # necessary for reproducibility between different runs
            payload_selfies = [smile2selfi[smile] for smile in payload if smile in smile2selfi]

            seen.update(canonicalized_samples)

            # scores = objective.score_list(payload)
            scores = objective(payload)
            scores.sort(reverse=True)
            # print("len(smiles)", len(smiles), "canonicalized_samples", len(canonicalized_samples), \
            #       "len(payload)", len(payload), "payload", len(payload))
            # print("scores", scores[:10])
            if objective.finish: 
                break 


            int_results = [OptResult(smiles=smiles, selfies = selfie, score=score) for smiles, selfie, score in zip(payload, payload_selfies, scores)]

            t2 = time.time()

            results.extend(sorted(int_results, reverse=True)[0:keep_top])
            results.sort(reverse=True)
            subset = [i.selfies for i in results][0:keep_top]  #### selfies 

            np.random.shuffle(subset)
            sub_train = subset[0:int(3 * len(subset) / 4)]
            sub_test = subset[int(3 * len(subset) / 4):]


            #########################
            ####### smiles2selfies
            # sub_train = smiles2selfies(sub_train)
            # sub_test = smiles2selfies(sub_test)
            # for selfies in sub_train:
            #     words = selfies.strip().strip('[]').split('][')
            #     smiles = selfies2smiles(selfies)
            #     assert Chem.MolFromSmiles(smiles) is not None 
            #     words = ['['+word+']' for word in words]
            #     for word in words:
            #         assert word in vocab_list 
            #########################

            train_seqs, _ = load_selfies_from_list(sub_train, max_len=self.max_len)
            valid_seqs, _ = load_selfies_from_list(sub_test, max_len=self.max_len)

            train_set = get_tensor_dataset(train_seqs)
            valid_set = get_tensor_dataset(valid_seqs)

            opt_batch_size = min(len(sub_train), optimize_batch_size)

            print_every = int(len(sub_train) / opt_batch_size)

            if optimize_n_epochs > 0:
                self.trainer.fit(train_set, valid_set,
                                 n_epochs=optimize_n_epochs,
                                 batch_size=opt_batch_size,
                                 print_every=print_every,
                                 valid_every=print_every)

            t3 = time.time()

            logger.info(f'Generation {epoch} --- timings: '
                        f'sample: {(t1 - t0):.3f} s, '
                        f'score: {(t2 - t1):.3f} s, '
                        f'finetune: {(t3 - t2):.3f} s')

            top4 = '\n'.join(f'\t{result.score:.3f}: {result.smiles}' for result in results[:4])
            logger.info(f'Top 4:\n{top4}')

        return sorted(results, reverse=True), objective

    def sample(self, num_mols) -> List[str]:
        """ :return: a list of molecules """
        return self.sampler.sample(self.model,
                                   num_to_sample=num_mols,
                                   max_seq_len=self.max_len)

    # TODO refactor, still has lots of duplication
    def pretrain_on_initial_population(self, scoring_function: ScoringFunction,
                                       start_population, pretrain_epochs) -> List[OptResult]:
        """
        Takes an objective and tries to optimise it
        :param scoring_function: MPO
        :param start_population: Initial compounds (list of smiles) or request new (random?) population
        :param pretrain_epochs: number of epochs to finetune with start_population
        :return: Candidate molecules
        """
        seed: List[OptResult] = []

        start_population_size = len(start_population)
        training = canonicalize_list(start_population, include_stereocenters=True)

        if len(training) != start_population_size:
            logger.warning("Some entries for the start population are invalid or duplicated")
            start_population_size = len(training)

        if start_population_size == 0:
            return seed

        logger.info("finetuning with {} molecules for {} epochs".format(start_population_size, pretrain_epochs))

        # scores = scoring_function.score_list(training)
        scores = scoring_function(training)
        selfies_list = smiles2selfies(training)
        seed.extend(OptResult(smiles=smiles, selfies=selfies, score=score) for smiles, selfies, score in zip(training, selfies_list, scores))


        ##############################
        ###### smiles2selfies 
        training = smiles2selfies(training)
        ##############################


        train_seqs, _ = load_selfies_from_list(training, max_len=self.max_len)
        train_set = get_tensor_dataset(train_seqs)

        batch_size = min(int(len(training)), 32)

        print_every = len(training) / batch_size

        losses = self.trainer.fit(train_set, train_set,
                                  batch_size=batch_size,
                                  n_epochs=pretrain_epochs,
                                  print_every=print_every,
                                  valid_every=print_every)
        logger.info(losses)
        return seed
