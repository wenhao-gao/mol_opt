
import typing
import enum
import collections
import sys
from os import path

import numpy as np
from guacamol import standard_benchmarks
import networkx as nx
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import rdmolops
from rdkit.Chem.Descriptors import MolLogP
from rdkit.Chem import RDConfig
sys.path.append(path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from ..chem_ops import rdkit_general_ops


class PropertyEvaluator:
    """
    Wraps the property calculator so that we can memoize the oracle calls.
    Also can track the order that the molecules are queried in and number of (unique) molecules queried.
    """
    def __init__(self, property_calculator, dim=1):
        self.seen_molecules = collections.OrderedDict()
        self.property_calculator = property_calculator
        self.dim = dim

    @property
    def num_evaluated(self):
        return len(self.seen_molecules)

    @property
    def best_seen(self):
        seen_molecule_vals = list(self.seen_molecules.items())
        return max(seen_molecule_vals, key=lambda x: x[1])

    def evaluate_molecules(self, list_of_smiles: typing.List[str]):
        out = []
        for smi in list_of_smiles:
            canon_smi = rdkit_general_ops.canconicalize(smi)
            if canon_smi not in self.seen_molecules:
                value = self.property_calculator(canon_smi)
                self.seen_molecules[canon_smi] = value

            out.append(self.seen_molecules[canon_smi])
        return np.array(out)



def qed(smi):
    mol = rdkit_general_ops.get_molecule(smi, kekulize=False)
    qed = QED.qed(mol)
    return [qed]


def get_penalized_logp():
    def reward_penalized_log_p_gcpn(smiles):
        """
        Reward that consists of log p penalized by SA and # long cycles,
        as described in (Kusner et al. 2017). Scores are normalized based on the
        statistics of 250k_rndm_zinc_drugs_clean.smi dataset
        :param mol: rdkit mol object
        :return: float
        """
        mol = Chem.MolFromSmiles(smiles)
        # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        SA_mean = -3.0525811293166134
        SA_std = 0.8335207024513095
        cycle_mean = -0.0485696876403053
        cycle_std = 0.2860212110245455

        log_p = MolLogP(mol)
        SA = -sascorer.calculateScore(mol)

        # cycle score
        cycle_list = nx.cycle_basis(nx.Graph(
            Chem.rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_score = -cycle_length

        normalized_log_p = (log_p - logP_mean) / logP_std
        normalized_SA = (SA - SA_mean) / SA_std
        normalized_cycle = (cycle_score - cycle_mean) / cycle_std

        return [normalized_log_p + normalized_SA + normalized_cycle]

    return PropertyEvaluator(reward_penalized_log_p_gcpn)


class GuacTask(enum.Enum):
    """
    We wrap the Guacamole tasks in this class (redefining several of them below) so that we can call the tasks
    individually rather than running a pre-built 'suite'
    """
    ARIPIPRAZOLE = "Aripiprazole_similarity"
    OSIMERTINIB = "Osimertinib_MPO"
    RANOLAZINE = "Ranolazine_MPO"
    ZALEPLON = "Zaleplon_MPO"
    VALSARTAN = "Valsartan_SMARTS"
    DECO = "decoration_hop"
    SCAFFOLD = "scaffold_hop"

    PERINDOPRIL = "Perindopril_MPO"
    AMLODIPINE = "Amlodipine_MPO"
    SITAGLIPTIN = "Sitagliptin_MPO"

    CELECOXIB = "Celecoxib_rediscovery"
    TROGLITAZONE = "Troglitazone_rediscovery"
    THIOTHIXENE = "Thiothixene_rediscovery"
    ALBUTEROL = "Albuterol_similarity"
    MESTRANOL = "Mestranol_similarity"
    FEXOFENADINE = "Fexofenadine_MPO"

    @classmethod
    def get_guac_property_eval(self, task):
        if task is GuacTask.CELECOXIB:
            bench = standard_benchmarks.similarity(smiles='CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F',
                                                   name='Celecoxib', fp_type='ECFP4', threshold=1.0, rediscovery=True)
        elif task is GuacTask.TROGLITAZONE:
            bench = standard_benchmarks.similarity(smiles='Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O',
                                                   name='Troglitazone', fp_type='ECFP4', threshold=1.0, rediscovery=True)
        elif task is GuacTask.THIOTHIXENE:
            bench = standard_benchmarks.similarity(smiles='CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1',
                                                   name='Thiothixene', fp_type='ECFP4', threshold=1.0, rediscovery=True)
        elif task is GuacTask.ARIPIPRAZOLE:
            bench = standard_benchmarks.similarity(smiles='Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl',
                                                   name='Aripiprazole', fp_type='ECFP4', threshold=0.75)
        elif task is GuacTask.ALBUTEROL:
            bench = standard_benchmarks.similarity(smiles='CC(C)(C)NCC(O)c1ccc(O)c(CO)c1', name='Albuterol', fp_type='FCFP4',
                                                   threshold=0.75)
        elif task is GuacTask.MESTRANOL:
            bench = standard_benchmarks.similarity(smiles='COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1',
                                                   name='Mestranol', fp_type='AP', threshold=0.75)
        elif task is GuacTask.OSIMERTINIB:
            bench = standard_benchmarks.hard_osimertinib()
        elif task is GuacTask.RANOLAZINE:
            bench = standard_benchmarks.ranolazine_mpo()
        elif task is GuacTask.ZALEPLON:
            bench = standard_benchmarks.zaleplon_with_other_formula()
        elif task is GuacTask.VALSARTAN:
            bench = standard_benchmarks.valsartan_smarts()
        elif task is GuacTask.DECO:
            bench = standard_benchmarks.decoration_hop()
        elif task is GuacTask.SCAFFOLD:
            bench = standard_benchmarks.scaffold_hop()
        elif task is GuacTask.PERINDOPRIL:
            bench = standard_benchmarks.perindopril_rings()
        elif task is GuacTask.AMLODIPINE:
            bench = standard_benchmarks.amlodipine_rings()
        elif task is GuacTask.SITAGLIPTIN:
            bench = standard_benchmarks.sitagliptin_replacement()
        elif task is GuacTask.FEXOFENADINE:
            bench = standard_benchmarks.hard_fexofenadine()
        else:
            raise NotImplementedError
        smi2score = lambda smi: [bench.objective.score(smi)]
        return PropertyEvaluator(smi2score)

    @classmethod
    def get_name_to_enum(self) -> dict:
        return {k.value: k for k in self}


def get_task(name_of_task: str):
    """
    Given a task name (eg handed in as an argument to a script call) return the relevant PropertyEvaluator.
    See code for definition of class names. NB that Guacamol names are given by 'guac_<name>'
    """
    if name_of_task == 'qed':
        return PropertyEvaluator(qed)
    elif name_of_task == 'sas':
        return PropertyEvaluator(lambda smiles: [sascorer.calculateScore(Chem.MolFromSmiles(smiles))])
    elif name_of_task == 'pen_logp':
        return PropertyEvaluator(lambda smiles: [MolLogP(Chem.MolFromSmiles(smiles))])
    elif name_of_task[:5] == 'guac_':
        task = GuacTask.get_name_to_enum()[name_of_task[5:]]
        return GuacTask.get_guac_property_eval(task)
    else:
        raise NotImplementedError(f"{name_of_task} is not implemented.")

