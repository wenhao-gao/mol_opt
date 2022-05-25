import sys
sys.path.append('.')

import os, sys

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

import networkx as nx

from util.chemistry.standard_benchmarks import (
    similarity,
    isomers_c11h24,
    isomers_c9h10n2o2pf2cl,
    median_camphor_menthol,
    median_tadalafil_sildenafil,
    hard_osimertinib,
    hard_fexofenadine,
    ranolazine_mpo,
    perindopril_rings,
    amlodipine_rings,
    sitagliptin_replacement,
    zaleplon_with_other_formula,
    valsartan_smarts,
    decoration_hop,
    scaffold_hop,
    logP_benchmark,
    tpsa_benchmark,
    cns_mpo,
    qed_benchmark,
    isomers_c7h8n2o2,
    pioglitazone_mpo,
)

from guacamol.common_scoring_functions import (
    TanimotoScoringFunction,
    RdkitScoringFunction,
    CNS_MPO_ScoringFunction,
    IsomerScoringFunction,
    SMARTSScoringFunction,
)
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.scoring_function import (
    ArithmeticMeanScoringFunction,
    GeometricMeanScoringFunction,
    MoleculewiseScoringFunction,
)
from guacamol.utils.descriptors import (
    num_rotatable_bonds,
    num_aromatic_rings,
    logP,
    qed,
    tpsa,
    bertz,
    mol_weight,
    AtomCounter,
    num_rings,
)

import copy


class ThresholdedImprovementScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, objective, constraint, threshold, offset):
        super().__init__()
        self.objective = objective
        self.constraint = constraint
        self.threshold = threshold
        self.offset = offset

    def raw_score(self, smiles):
        score = (
            self.corrupt_score
            if (self.constraint.score(smiles) < self.threshold)
            else (self.objective.score(smiles) + self.offset)
        )

        return score


LOGP_MEAN = 2.4570965532649507
LOGP_STD = 1.4339810636722639
SASCORE_MEAN = 3.0508333383104556
SASCORE_STD = 0.8327034846660627
ATOMRING_CYCLESCORE_MEAN = 0.03805126763956079
ATOMRING_CYCLESCORE_STD = 0.22377819597468795
CYCLEBASIS_CYCLESCORE_MEAN = 0.048152237188108474
CYCLEBASIS_CYCLESCORE_STD = 0.2860582871837183


def _penalized_logp_atomrings(mol: Mol):
    log_p = Descriptors.MolLogP(mol)
    sa_score = sascorer.calculateScore(mol)

    cycle_list = mol.GetRingInfo().AtomRings()
    largest_ring_size = max([len(j) for j in cycle_list]) if cycle_list else 0
    cycle_score = max(largest_ring_size - 6, 0)

    log_p = (log_p - LOGP_MEAN) / LOGP_STD
    sa_score = (sa_score - SASCORE_MEAN) / SASCORE_STD
    cycle_score = (cycle_score - ATOMRING_CYCLESCORE_MEAN) / ATOMRING_CYCLESCORE_STD

    return log_p - sa_score - cycle_score


def _penalized_logp_cyclebasis(mol: Mol):
    log_p = Descriptors.MolLogP(mol)
    sa_score = sascorer.calculateScore(mol)

    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    largest_ring_size = max([len(j) for j in cycle_list]) if cycle_list else 0
    cycle_score = max(largest_ring_size - 6, 0)

    log_p = (log_p - LOGP_MEAN) / LOGP_STD
    sa_score = (sa_score - SASCORE_MEAN) / SASCORE_STD
    cycle_score = (cycle_score - CYCLEBASIS_CYCLESCORE_MEAN) / CYCLEBASIS_CYCLESCORE_STD

    return log_p - sa_score - cycle_score


def penalized_logp_atomrings():
    benchmark_name = "Penalized logP"
    objective = RdkitScoringFunction(descriptor=lambda mol: _penalized_logp_atomrings(mol))
    objective.corrupt_score = -1000.0
    specification = uniform_specification(1)
    return GoalDirectedBenchmark(
        name=benchmark_name, objective=objective, contribution_specification=specification
    )


def penalized_logp_cyclebasis():
    benchmark_name = "Penalized logP CycleBasis"
    objective = RdkitScoringFunction(descriptor=lambda mol: _penalized_logp_cyclebasis(mol))
    objective.corrupt_score = -1000.0
    specification = uniform_specification(1)
    return GoalDirectedBenchmark(
        name=benchmark_name, objective=objective, contribution_specification=specification
    )


def similarity_constrained_penalized_logp_atomrings(smiles, name, threshold, fp_type="ECFP4"):
    benchmark_name = f"{name} {threshold:.1f} Similarity Constrained Penalized logP"

    objective = RdkitScoringFunction(descriptor=lambda mol: _penalized_logp_atomrings(mol))
    offset = -objective.score(smiles)
    constraint = TanimotoScoringFunction(target=smiles, fp_type=fp_type)
    constrained_objective = ThresholdedImprovementScoringFunction(
        objective=objective, constraint=constraint, threshold=threshold, offset=offset
    )
    constrained_objective.corrupt_score = -1000.0

    specification = uniform_specification(1)

    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=constrained_objective,
        contribution_specification=specification,
    )


def similarity_constrained_penalized_logp_cyclebasis(smiles, name, threshold, fp_type="ECFP4"):
    benchmark_name = f"{name} {threshold:.1f} Similarity Constrained Penalized logP"

    objective = RdkitScoringFunction(descriptor=lambda mol: _penalized_logp_cyclebasis(mol))
    offset = -objective.score(smiles)
    constraint = TanimotoScoringFunction(target=smiles, fp_type=fp_type)
    constrained_objective = ThresholdedImprovementScoringFunction(
        objective=objective, constraint=constraint, threshold=threshold, offset=offset
    )
    constrained_objective.corrupt_score = -1000.0

    specification = uniform_specification(1)

    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=constrained_objective,
        contribution_specification=specification,
    )


def load_benchmark(benchmark_id):
    benchmark = {
        0: similarity(
            smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
            name="Celecoxib",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        1: similarity(
            smiles="Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
            name="Troglitazone",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        2: similarity(
            smiles="CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
            name="Thiothixene",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        3: similarity(
            smiles="Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl",
            name="Aripiprazole",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        4: similarity(
            smiles="CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
            name="Albuterol",
            fp_type="FCFP4",
            threshold=0.75,
        ),
        5: similarity(
            smiles="COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
            name="Mestranol",
            fp_type="AP",
            threshold=0.75,
        ),
        6: isomers_c11h24(),
        7: isomers_c9h10n2o2pf2cl(),
        8: median_camphor_menthol(),
        9: median_tadalafil_sildenafil(),
        10: hard_osimertinib(),
        11: hard_fexofenadine(),
        12: ranolazine_mpo(),
        13: perindopril_rings(),
        14: amlodipine_rings(),
        15: sitagliptin_replacement(),
        16: zaleplon_with_other_formula(),
        17: valsartan_smarts(),
        18: decoration_hop(),
        19: scaffold_hop(),
        20: logP_benchmark(target=-1.0),
        21: logP_benchmark(target=8.0),
        22: tpsa_benchmark(target=150.0),
        23: cns_mpo(),
        24: qed_benchmark(),
        25: isomers_c7h8n2o2(),
        26: pioglitazone_mpo(),
        27: penalized_logp_atomrings(),
        28: penalized_logp_cyclebasis(),
    }.get(benchmark_id)

    if benchmark_id in [
        3,
        4,
        5,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        26,
    ]:
        scoring_num_list = [1, 10, 100]
    elif benchmark_id in [6]:
        scoring_num_list = [159]
    elif benchmark_id in [7]:
        scoring_num_list = [250]
    elif benchmark_id in [25]:
        scoring_num_list = [100]
    elif benchmark_id in [0, 1, 2, 27, 28]:
        scoring_num_list = [100]

    return benchmark, scoring_num_list
