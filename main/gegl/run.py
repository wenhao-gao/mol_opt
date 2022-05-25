from __future__ import print_function

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import os
path_here = os.path.dirname(os.path.realpath(__file__))
import torch
from torch.optim import Adam

from runner.gegl_trainer import GeneticExpertGuidedLearningTrainer
from runner.guacamol_generator import GeneticExpertGuidedLearningGenerator
from model.neural_apprentice import SmilesGenerator, SmilesGeneratorHandler
from model.genetic_expert import GeneticOperatorHandler
from util.storage.priority_queue import MaxRewardPriorityQueue
from util.storage.recorder import Recorder
from util.chemistry.benchmarks import (
    similarity_constrained_penalized_logp_atomrings,
    similarity_constrained_penalized_logp_cyclebasis,
    penalized_logp_atomrings,
    penalized_logp_cyclebasis,
    TanimotoScoringFunction,
)
from util.smiles.char_dict import SmilesCharDictionary
from util.smiles.dataset import load_dataset
from random import shuffle 
from main.optimizer import BaseOptimizer


class GEGL_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "gegl"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        device = torch.device(0)

        char_dict = SmilesCharDictionary(dataset=config['dataset'], max_smi_len=config['max_smiles_length'])
        dataset = load_dataset(char_dict=char_dict, smi_path=config['dataset_path'])

        if config['use_atomrings']:
            similarity_constrained_penalized_logp = similarity_constrained_penalized_logp_atomrings
            penalized_logp_score_func = penalized_logp_atomrings().wrapped_objective.score
        else:
            similarity_constrained_penalized_logp = similarity_constrained_penalized_logp_cyclebasis
            penalized_logp_score_func = penalized_logp_cyclebasis().wrapped_objective.score

        for smi_id in range(config['smi_id_min'], config['smi_id_max']):
            print(f"ID: {smi_id}")
            reference_smi = dataset[smi_id]
            benchmark = similarity_constrained_penalized_logp(
                smiles=reference_smi, name=str(smi_id), threshold=config['similarity_threshold']
            )
            scoring_num_list = [1]

            apprentice_storage = MaxRewardPriorityQueue()
            expert_storage = MaxRewardPriorityQueue()

            apprentice = SmilesGenerator.load(load_dir=config['apprentice_load_dir'])
            apprentice = apprentice.to(device)
            apprentice.train()
            apprentice_optimizer = Adam(apprentice.parameters(), lr=config['learning_rate'])
            apprentice_handler = SmilesGeneratorHandler(
                model=apprentice,
                optimizer=apprentice_optimizer,
                char_dict=char_dict,
                max_sampling_batch_size=config['max_sampling_batch_size'],
            )

            expert_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'])

            trainer = GeneticExpertGuidedLearningTrainer(
                apprentice_storage=apprentice_storage,
                expert_storage=expert_storage,
                apprentice_handler=apprentice_handler,
                expert_handler=expert_handler,
                char_dict=char_dict,
                num_keep=config['num_keep'],
                apprentice_sampling_batch_size=config['apprentice_sampling_batch_size'],
                expert_sampling_batch_size=config['expert_sampling_batch_size'],
                apprentice_training_batch_size=config['apprentice_training_batch_size'],
                num_apprentice_training_steps=config['num_apprentice_training_steps'],
                init_smis=[reference_smi],
            )

            recorder = Recorder(scoring_num_list=scoring_num_list, record_filtered=config['record_filtered'])
            # continue 

            exp_generator = GeneticExpertGuidedLearningGenerator(
                trainer=trainer,
                recorder=recorder,
                num_steps=config['num_steps'],
                device=device,
                scoring_num_list=scoring_num_list,
                num_jobs=config['num_jobs'],
            )
            # continue 
            # number_molecules = 500 
            shuffle(self.all_smiles)
            starting_population = self.all_smiles[:500]
            exp_generator.generate_optimized_molecules(self.oracle, 500, starting_population)
            if self.finish:
                break 
            continue 

            result = benchmark.assess_model(exp_generator) #### run and call oracle 
            optimized_smi, score = result.optimized_molecules[0]
            reference_score = penalized_logp_score_func(reference_smi)
            optimized_score = penalized_logp_score_func(optimized_smi)
            similarity = TanimotoScoringFunction(target=reference_smi, fp_type="ECFP4").score(optimized_smi)

