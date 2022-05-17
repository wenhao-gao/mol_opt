import random
import json
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
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

# import neptune

import sys

# sys.stdout = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--smi_id_min", type=int, default=0)
    parser.add_argument("--smi_id_max", type=int, default=800)
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument("--dataset_path", type=str, default="./resource/data/zinc/logp_800.txt")
    parser.add_argument("--max_smiles_length", type=int, default=120)
    parser.add_argument("--similarity_threshold", type=float, default=0.4)
    parser.add_argument("--apprentice_load_dir", type=str, default="./resource/checkpoint/zinc")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--sample_batch_size", type=int, default=512)
    parser.add_argument("--optimize_batch_size", type=int, default=256)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--num_keep", type=int, default=1024)
    parser.add_argument("--max_sampling_batch_size", type=int, default=1024)
    parser.add_argument("--apprentice_sampling_batch_size", type=int, default=1024)
    parser.add_argument("--expert_sampling_batch_size", type=int, default=1024)
    parser.add_argument("--apprentice_training_batch_size", type=int, default=256)
    parser.add_argument("--num_apprentice_training_steps", type=int, default=4)
    parser.add_argument("--num_jobs", type=int, default=8)
    parser.add_argument("--record_filtered", action="store_true")
    parser.add_argument("--use_atomrings", action="store_true")
    args = parser.parse_args()

    args.algorithm = "gegl_constrained"

    random.seed(0)
    device = torch.device(0)

    # neptune.init(project_qualified_name="molopt")
    # experiment = neptune.create_experiment(name=args.algorithm, params=vars(args))
    # neptune.append_tag(
    #     f"{args.smi_id_min:03d}_{args.smi_id_max:03d}_{args.similarity_threshold}".replace(".", "")
    # )

    char_dict = SmilesCharDictionary(dataset=args.dataset, max_smi_len=args.max_smiles_length)
    dataset = load_dataset(char_dict=char_dict, smi_path=args.dataset_path)

    if args.use_atomrings:
        similarity_constrained_penalized_logp = similarity_constrained_penalized_logp_atomrings
        penalized_logp_score_func = penalized_logp_atomrings().wrapped_objective.score
    else:
        similarity_constrained_penalized_logp = similarity_constrained_penalized_logp_cyclebasis
        penalized_logp_score_func = penalized_logp_cyclebasis().wrapped_objective.score

    for smi_id in range(args.smi_id_min, args.smi_id_max):
        print(f"ID: {smi_id}")
        reference_smi = dataset[smi_id]
        benchmark = similarity_constrained_penalized_logp(
            smiles=reference_smi, name=str(smi_id), threshold=args.similarity_threshold
        )
        scoring_num_list = [1]

        apprentice_storage = MaxRewardPriorityQueue()
        expert_storage = MaxRewardPriorityQueue()

        apprentice = SmilesGenerator.load(load_dir=args.apprentice_load_dir)
        apprentice = apprentice.to(device)
        apprentice.train()
        apprentice_optimizer = Adam(apprentice.parameters(), lr=args.learning_rate)
        apprentice_handler = SmilesGeneratorHandler(
            model=apprentice,
            optimizer=apprentice_optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=args.max_sampling_batch_size,
        )

        expert_handler = GeneticOperatorHandler(mutation_rate=args.mutation_rate)

        trainer = GeneticExpertGuidedLearningTrainer(
            apprentice_storage=apprentice_storage,
            expert_storage=expert_storage,
            apprentice_handler=apprentice_handler,
            expert_handler=expert_handler,
            char_dict=char_dict,
            num_keep=args.num_keep,
            apprentice_sampling_batch_size=args.apprentice_sampling_batch_size,
            expert_sampling_batch_size=args.expert_sampling_batch_size,
            apprentice_training_batch_size=args.apprentice_training_batch_size,
            num_apprentice_training_steps=args.num_apprentice_training_steps,
            init_smis=[reference_smi],
        )

        recorder = Recorder(scoring_num_list=scoring_num_list, record_filtered=args.record_filtered)
        # continue 

        exp_generator = GeneticExpertGuidedLearningGenerator(
            trainer=trainer,
            recorder=recorder,
            num_steps=args.num_steps,
            device=device,
            scoring_num_list=scoring_num_list,
            num_jobs=args.num_jobs,
        )
        # continue 

        result = benchmark.assess_model(exp_generator) #### run and call oracle 
        optimized_smi, score = result.optimized_molecules[0]
        reference_score = penalized_logp_score_func(reference_smi)
        optimized_score = penalized_logp_score_func(optimized_smi)
        similarity = TanimotoScoringFunction(target=reference_smi, fp_type="ECFP4").score(optimized_smi)






