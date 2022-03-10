from __future__ import print_function

import argparse
import copy
import os
from typing import List, Optional
import nltk
import yaml
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
rdBase.DisableLog('rdApp.error')
from tdc import Oracle

import cfg_util, smiles_grammar 
from main.optimizer import BaseOptimizer

GCFG = smiles_grammar.GCFG


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
        return ""


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
    good_population = []
    for item in population:
        smiles, _ = item
        if len(smiles) > 0:
            if smiles not in unique_smiles:
                good_population.append(item)
            unique_smiles.add(smiles)
    return good_population


def mutate(p_gene):
    c_gene = mutation(p_gene)
    c_smiles = canonicalize(cfg_util.decode(gene_to_cfg(c_gene)))
    return c_smiles, c_gene



class SMILES_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "smiles_ga"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        
        if self.smi_file is not None:
            # Exploitation run
            starting_population = self.all_smiles[:(config["population_size"]+config["n_mutations"])]
        else:
            # Exploration run
            starting_population = np.random.choice(self.all_smiles, (config["population_size"]+config["n_mutations"]))
        starting_population = [smiles for smiles in starting_population if '%' not in smiles]
        population_smi = starting_population[:config["population_size"]]

        # calculate initial genes
        population_genes = [cfg_to_gene(cfg_util.encode(s), max_len=config["gene_size"])
                            for s in starting_population]

        # score initial population
        population_scores = self.oracle(population_smi)
        population = [(population_smi[i], population_genes[i]) for i in range(len(population_smi))]

        patience = 0
        for _ in range(config["max_generations"]):

            old_scores = population_scores
            # select random genes
            all_genes = [molecule[1] for molecule in population]
            choice_indices = np.random.choice(len(all_genes), config["n_mutations"], replace=True)
            genes_to_mutate = [all_genes[i] for i in choice_indices]

            # evolve genes
            joblist = (delayed(mutate)(g) for g in genes_to_mutate)
            new_population = self.pool(joblist)

            # join and dedup
            population += new_population
            population = deduplicate(population)
            population_scores = self.oracle([molecule[0] for molecule in population])

            # survival of the fittest
            population_tuples = list(zip(population_scores, population))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # early stopping
            if population_scores == old_scores:
                patience += 1
                if patience >= config["patience"]:
                    self.log_intermediate(finish=True)
                    break
            else:
                patience = 0
                
            if self.finish:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

    path_here = os.path.dirname(os.path.realpath(__file__))

    if args.output_dir is None:
        args.output_dir = os.path.join(path_here, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    for oracle_name in args.oracles:

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_here, args.config_default)))

        if args.task == "tune":
            try:
                config_tune = yaml.safe_load(open(args.config_tune))
            except:
                config_tune = yaml.safe_load(open(os.path.join(path_here, args.config_tune)))

        oracle = Oracle(name = oracle_name)
        optimizer = SMILES_GA_Optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main()

