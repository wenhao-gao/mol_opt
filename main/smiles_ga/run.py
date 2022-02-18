from __future__ import print_function

import argparse
import copy
import json
import os
from collections import namedtuple
from time import time
from typing import List, Optional
from random import shuffle 
import joblib
import nltk
import numpy as np
from joblib import delayed
from rdkit import rdBase

from guacamol.assess_goal_directed_generation import assess_goal_directed_generation 
from guacamol.goal_directed_generator import GoalDirectedGenerator 
from guacamol.scoring_function import ScoringFunction 
# from . import cfg_util, smiles_grammar
import cfg_util, smiles_grammar 

rdBase.DisableLog('rdApp.error')
GCFG = smiles_grammar.GCFG

Molecule = namedtuple('Molecule', ['score', 'smiles', 'genes'])


from tdc import Oracle
from tdc import Evaluator
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
qed = Oracle(name = 'qed')
from sa import sa
def oracle(smiles):
    try:
        scores = [qed(smiles), sa(smiles), jnk(smiles), gsk(smiles)]
    except:
        # return -np.inf
        return -100.0 
    return np.mean(scores)



from rdkit import Chem 
def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.
    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543
    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string
    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None





def cfg_to_gene(prod_rules, max_len=-1):
    gene = []
    for r in prod_rules:
        lhs = GCFG.productions()[r].lhs()
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                          if rule.lhs() == lhs]
        gene.append(possible_rules.index(r))
    if max_len > 0:
        if len(gene) > max_len:
            gene = gene[:max_len]
        else:
            gene = gene + [np.random.randint(0, 256)
                           for _ in range(max_len - len(gene))]
    return gene


def gene_to_cfg(gene):
    prod_rules = []
    stack = [GCFG.productions()[0].lhs()]
    for g in gene:
        try:
            lhs = stack.pop()
        except Exception:
            break
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                          if rule.lhs() == lhs]
        rule = possible_rules[g % len(possible_rules)]
        prod_rules.append(rule)
        rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
                     smiles_grammar.GCFG.productions()[rule].rhs())
        stack.extend(list(rhs)[::-1])
    return prod_rules


def select_parent(population, tournament_size=3):
    idx = np.random.randint(len(population), size=tournament_size)
    best = population[idx[0]]
    for i in idx[1:]:
        if population[i][0] > best[0]:
            best = population[i]
    return best


def mutation(gene):
    idx = np.random.choice(len(gene))
    gene_mutant = copy.deepcopy(gene)
    gene_mutant[idx] = np.random.randint(0, 256)
    return gene_mutant


def deduplicate(population):
    unique_smiles = set()
    unique_population = []
    for item in population:
        score, smiles, gene = item
        if smiles not in unique_smiles:
            unique_population.append(item)
        unique_smiles.add(smiles)
    return unique_population


# My modification: pass in a dictionary of known values to track function calls
def score_mol(smiles, score_fn, known_value_dict):
    # smiles = Chem.MolToSmiles(mol)
    if smiles not in known_value_dict:
        known_value_dict[smiles] = score_fn(smiles)
    return known_value_dict[smiles]

def mutate(p_gene, scoring_function, known_value_dict):
    c_gene = mutation(p_gene)
    c_smiles = canonicalize(cfg_util.decode(gene_to_cfg(c_gene)))
    # c_score = scoring_function.score(c_smiles)
    c_score = score_mol(c_smiles, scoring_function, known_value_dict)
    return Molecule(c_score, c_smiles, c_gene)



# My modification: simplified version of the Goal Directed generator
def generate_optimized_molecules(
    scoring_function,
    start_known_smiles: dict,
    starting_population: List[str],
    n_jobs: int=-1,
    population_size: int = 1000,
    n_mutations: int=200,
    gene_size: int=300,
    generations: int=1000, 
    random_start:bool=True, 
    patience_max:int=5,
    max_total_func_calls: int = 1000,
) -> List[str]:

    pool = joblib.Parallel(n_jobs=n_jobs)

    # Accurately track function evaluations by storing all known scores so far
    f_cache = dict(start_known_smiles)

    # # fetch initial population
    # if starting_population is None:
    #     print('selecting initial population...')
    #     init_size = population_size + n_mutations
    #     # all_smiles = copy.deepcopy(self.all_smiles)
    #     if random_start:
    #         starting_population = np.random.choice(all_smiles, init_size)
    #     else:
    #         starting_population = self.top_k(all_smiles, scoring_function, init_size)

    # select initial population
    # print("Scoring initial population...")
    # population_smiles = list(starting_population)
    # # population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
    # population_scores = [score_mol(m, scoring_function, f_cache) for m in population_smiles]
    # print("Initial population scoring complete!")
    # print(f"Max starting score: {max(population_scores)}")


    # The smiles GA cannot deal with '%' in SMILES strings (used for two-digit ring numbers).

    starting_population = [smiles for smiles in starting_population if '%' not in smiles]
    shuffle(starting_population)
    starting_population = starting_population[:population_size]
    # calculate initial genes
    initial_genes = [cfg_to_gene(cfg_util.encode(s), max_len=gene_size)
                         for s in starting_population]

    # score initial population
    # initial_scores = scoring_function.score_list(starting_population)
    initial_scores = [score_mol(m, scoring_function, f_cache) for m in starting_population]
    population = [Molecule(*m) for m in zip(initial_scores, starting_population, initial_genes)]
    population = sorted(population, key=lambda x: x.score, reverse=True)[:population_size]
    population_scores = [p.score for p in population]

    # evolution: go go go!!
    t0 = time()
    patience = 0
    for generation in range(generations):

        old_scores = population_scores
        # select random genes
        all_genes = [molecule.genes for molecule in population]
        choice_indices = np.random.choice(len(all_genes), n_mutations, replace=True)
        genes_to_mutate = [all_genes[i] for i in choice_indices]

        # evolve genes
        # joblist = (delayed(mutate)(g, scoring_function, f_cache) for g in genes_to_mutate)
        # new_population = pool(joblist)
        new_population = [mutate(g, scoring_function, f_cache) for g in genes_to_mutate]

        # join and dedup
        population += new_population
        population = deduplicate(population)

        # survival of the fittest
        population = sorted(population, key=lambda x: x.score, reverse=True)[:population_size]

        # stats
        gen_time = time() - t0
        mol_sec = (population_size + n_mutations) / gen_time
        t0 = time()

        population_scores = [p.score for p in population]

        # early stopping
        if population_scores == old_scores:
            patience += 1 
            print(f'Failed to progress: {patience}')
            if patience >= patience_max:
                print(f'No more patience, bailing...')
                break
        else:
            patience = 0

        print(f'{generation} | '
                  f'max: {np.max(population_scores):.3f} | '
                  f'avg: {np.mean(population_scores):.3f} | '
                  f'min: {np.min(population_scores):.3f} | '
                  f'std: {np.std(population_scores):.3f} | '
                  f'{gen_time:.2f} sec/gen | '
                  f'{mol_sec:.2f} mol/sec | '
                  f'{len(f_cache):d} oracle calls')

        if len(f_cache) > max_total_func_calls:
            print("Max function calls hit, aborting")
            break

    return f_cache
    # return [molecule.smiles for molecule in population[:number_molecules]]





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_file', default='data/guacamol_v1_all.smiles')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--n_mutations', type=int, default=200)
    parser.add_argument('--gene_size', type=int, default=300)
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_func_calls', type=int, default=200)

    args = parser.parse_args()
    np.random.seed(args.seed)
    # setup_default_logger()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    # # save command line args
    # with open(os.path.join(args.output_dir, 'goal_directed_params.json'), 'w') as jf:
    #     json.dump(vars(args), jf, sort_keys=True, indent=4)

    # optimiser = ChemGEGenerator(smi_file=args.smiles_file,
    #                             population_size=args.population_size,
    #                             n_mutations=args.n_mutations,
    #                             gene_size=args.gene_size,
    #                             generations=args.generations,
    #                             n_jobs=args.n_jobs,
    #                             random_start=args.random_start,
    #                             patience=args.patience)

    # json_file_path = os.path.join(args.output_dir, 'goal_directed_results.json')
    # assess_goal_directed_generation(optimiser, json_output_file=json_file_path, benchmark_version=args.suite)

    smiles_file = args.smiles_file
    with open(smiles_file, 'r') as fin:
        start_smiles = fin.readlines()
        start_smiles = [smiles.strip() for smiles in start_smiles]

    print("begin running smiles-GA")
    all_func_evals = generate_optimized_molecules(
        scoring_function = oracle,
        n_jobs = args.n_jobs, 
        start_known_smiles = dict(),
        starting_population=list(start_smiles),
        population_size=args.population_size,
        n_mutations=args.n_mutations,
        gene_size=args.gene_size,
        generations = args.generations, 
        random_start=args.random_start, 
        patience_max=args.patience,
        max_total_func_calls=args.max_func_calls, 
    )


    # Evaluate 
    new_score_tuples = [(v, k) for k, v in all_func_evals.items() if k not in start_smiles and k is not None and k!='']  # scores of new molecules
    print(new_score_tuples)
    new_score_tuples.sort(reverse=True,key=lambda x:x[0])
    top100_mols = [(k, v) for (v, k) in new_score_tuples[:100]]
    diversity = Evaluator(name = 'Diversity')
    div = diversity([t[0] for t in top100_mols])
    output = dict(
        top_mols=top100_mols,
        AST=np.average([t[1] for t in top100_mols]),
        diversity=div,
        all_func_evals=dict(all_func_evals),
    )
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()



