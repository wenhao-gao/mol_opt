from rdkit import Chem
from rdkit.Chem import Crippen, QED as qed_module, Descriptors, rdmolops
import networkx as nx

from mol_funcs.SA_Score import sascorer

# Guacamol
guacamol_funcs = dict()
try:
    from guacamol import benchmark_suites as guac_benchmarks

    for benchmark in guac_benchmarks.goal_directed_suite_v2():
        converted_name = benchmark.name.lower().replace(" ", "-")
        guacamol_funcs[converted_name] = benchmark.objective.score
except ImportError:
    pass


def QED(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return qed_module.qed(mol)


def logP(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Crippen.MolLogP(mol)


def molecular_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)


def _cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def penalized_logP(smiles: str) -> float:
    """calculate penalized logP for a given smiles string"""
    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)
    sa = sascorer.calculateScore(mol)

    # Calculate cycle score
    cycle_length = _cycle_score(mol)

    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I think this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    score = (
        (logp - 2.45777691) / 1.43341767
        + (-sa + 3.05352042) / 0.83460587
        + (-cycle_length - -0.04861121) / 0.28746695
    )
    return score
