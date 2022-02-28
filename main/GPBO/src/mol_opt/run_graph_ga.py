import argparse
import json
import logging
import pickle

import pandas as pd

from graph_ga.graph_ga import run_ga_maximization, ga_logger
from mol_opt import get_base_molopt_parser, get_cached_objective_and_dataframe


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max_ga_generations", type=int, default=10000)
    parser.add_argument(
        "--population_size",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--offspring_size",
        type=int,
        default=25,
    )
    parser.add_argument("--mutation_rate", type=float, default=1e-2)
    return parser


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(parents=[get_parser(), get_base_molopt_parser()])
    args = parser.parse_args()

    # Load dataset
    dataset = pd.read_csv(args.dataset, sep="\t", header=0)

    # Get function to be optimized
    opt_func, df_processed = get_cached_objective_and_dataframe(
        objective_name=args.objective,
        dataset=dataset,
        minimize=not args.maximize,
        keep_nan=False,
        max_docking_score=0.0,
        dock_kwargs=dict(
            num_cpus=args.num_cpu,
        ),
    )
    dataset_smiles = set(map(str, df_processed.smiles))

    # Run actual GA
    ga_logger.setLevel(logging.INFO)
    start_population = list(dataset_smiles)
    assert set(start_population) <= set(opt_func.cache.keys())
    all_smiles, value_dict, ga_info = ga_res = run_ga_maximization(
        starting_population_smiles=start_population,
        scoring_function=opt_func,
        max_generations=args.max_ga_generations,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        max_func_calls=args.max_func_calls,
    )

    # Format results by providing new SMILES + scores
    new_smiles = [s for s in all_smiles if s not in dataset_smiles]
    new_smiles_scores = [opt_func(s) for s in new_smiles]
    new_smiles_raw_info = [opt_func.cache[s] for s in new_smiles]
    json_res = dict(
        new_smiles=new_smiles, scores=new_smiles_scores, raw_scores=new_smiles_raw_info
    )

    # Save results
    with open(args.output_path, "w") as f:
        json.dump(json_res, f, indent=2)
    if args.extra_output_path is not None:
        with open(args.extra_output_path, "wb") as f:
            pickle.dump(ga_res, f)
