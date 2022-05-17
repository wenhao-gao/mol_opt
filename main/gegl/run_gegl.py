import random
import json
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from runner.gegl_trainer import GeneticExpertGuidedLearningTrainer
from runner.guacamol_generator import GeneticExpertGuidedLearningGenerator
from model.neural_apprentice import SmilesGenerator, SmilesGeneratorHandler
from model.genetic_expert import GeneticOperatorHandler
from util.storage.priority_queue import MaxRewardPriorityQueue
from util.storage.recorder import Recorder
from util.chemistry.benchmarks import load_benchmark
from util.smiles.char_dict import SmilesCharDictionary

# import neptune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark_id", type=int, default=12)
    parser.add_argument("--dataset", type=str, default="guacamol")
    parser.add_argument("--max_smiles_length", type=int, default=100)
    parser.add_argument("--apprentice_load_dir", type=str, default="./resource/checkpoint/guacamol")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--sample_batch_size", type=int, default=512)
    parser.add_argument("--optimize_batch_size", type=int, default=256)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--num_keep", type=int, default=1024)
    parser.add_argument("--max_sampling_batch_size", type=int, default=1024)
    parser.add_argument("--apprentice_sampling_batch_size", type=int, default=8192)
    parser.add_argument("--expert_sampling_batch_size", type=int, default=8192)
    parser.add_argument("--apprentice_training_batch_size", type=int, default=256)
    parser.add_argument("--num_apprentice_training_steps", type=int, default=8)
    parser.add_argument("--num_jobs", type=int, default=8)
    parser.add_argument("--record_filtered", action="store_true")
    args = parser.parse_args()

    # Prepare CUDA device
    device = torch.device(0)

    # Initialize neptune
    # neptune.init(project_qualified_name="sungsoo.ahn/deep-molecular-optimization")
    # experiment = neptune.create_experiment(name="gegl", params=vars(args))
    # neptune.append_tag(args.benchmark_id)

    # Load benchmark, i.e., the scoring function and its corresponding protocol
    benchmark, scoring_num_list = load_benchmark(args.benchmark_id)

    # Load character directory used for mapping atoms to integers
    char_dict = SmilesCharDictionary(dataset=args.dataset, max_smi_len=args.max_smiles_length)

    # Prepare max-reward priority queues
    apprentice_storage = MaxRewardPriorityQueue()
    expert_storage = MaxRewardPriorityQueue()

    # Prepare neural apprentice (we use the weights pretrained on existing dataset)
    apprentice = SmilesGenerator.load(load_dir=args.apprentice_load_dir)
    apprentice = apprentice.to(device)
    apprentice_optimizer = Adam(apprentice.parameters(), lr=args.learning_rate)
    apprentice_handler = SmilesGeneratorHandler(
        model=apprentice,
        optimizer=apprentice_optimizer,
        char_dict=char_dict,
        max_sampling_batch_size=args.max_sampling_batch_size,
    )
    apprentice.train()

    # Prepare genetic expert
    expert_handler = GeneticOperatorHandler(mutation_rate=args.mutation_rate)

    # Prepare trainer that collect samples from the models & optimize the neural apprentice
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
        init_smis=[],
    )

    # Prepare recorder that takes care of intermediate logging
    recorder = Recorder(scoring_num_list=scoring_num_list, record_filtered=args.record_filtered)

    # Prepare our version of GoalDirectedGenerator for evaluating our algorithm
    guacamol_generator = GeneticExpertGuidedLearningGenerator(
        trainer=trainer,
        recorder=recorder,
        num_steps=args.num_steps,
        device=device,
        scoring_num_list=scoring_num_list,
        num_jobs=args.num_jobs,
    )

    # Run the experiment
    result = benchmark.assess_model(guacamol_generator)

    # Dump the final result to neptune
    # neptune.set_property("benchmark_score", result.score)
