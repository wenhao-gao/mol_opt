"""
Code define the molecular generation environment in synthesizable chemical space
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import time
import requests
import random
import pickle as pkl

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from six.moves import range
from six.moves import zip
from SA_Score import sascorer

from utils import utils

from guacamol.standard_benchmarks import similarity, logP_benchmark, qed_benchmark, median_camphor_menthol, \
    isomers_c11h24, isomers_c9h10n2o2pf2cl, hard_osimertinib, hard_fexofenadine, perindopril_rings, \
    amlodipine_rings, sitagliptin_replacement, zaleplon_with_other_formula, valsartan_smarts, \
    median_tadalafil_sildenafil, decoration_hop, scaffold_hop, ranolazine_mpo


###################################################################################
#                   General Molecular Generation Environment                      #
###################################################################################


def get_valid_actions(state,
                      buyable,
                      allow_no_modification,
                      HOST):
    """Compute a set of valid action for given state

    Argument
    ------------

        - state. String SMILES.
            The current state.

        - atom_types. Set of string atom types.

        - allow_removal. Boolean
            Whether to allow removal action.

        - allow_no_modification. Boolean.
            Whether to allow no modification.

        - allowed_ring_sizes. Set of integer.

        - allow_bonds_between_rings. Boolean.

    """
    # Check validity

    mol = Chem.MolFromSmiles(state)
    if mol is None:
        raise ValueError('Received invalid state: %s' % state)

    # Get valid actions
    valid_actions = set()

    time_start = time.time()
    valid_actions.update(
        _forward_predict(
            state=state,
            reactants=buyable,
            cutoff=0.5,
            HOST=HOST
        )
    )
    print(f'Forward reaction prediction elapsed: {time.time() - time_start}')

    if allow_no_modification:
        valid_actions.add(Chem.MolToSmiles(mol))

    return list(valid_actions)


def _forward_predict(state, reactants=None, cutoff=0.8, HOST=None):
    """Compute valid forward reaction prediction"""

    assert HOST is not None

    products = set()

    # Predict without reactants
    products |= _get_products(state, HOST, cutoff=cutoff, num_results=5)

    for smi in reactants:
        products |= _get_products(state + '.' + smi, HOST, cutoff=cutoff, num_results=5)

    return products


def _get_products(reac, HOST, cutoff=0.8, num_results=10):

    print(f'Calling the forward synthesis: {reac}')

    products = set()

    params = {
        'reactants': reac,  # required

        # optional with defaults shown
        'reagents': '',
        'solvent': '',

        'num_results': num_results  # default is 100
    }

    resp = requests.get(HOST + '/api/forward/?', params=params, verify=False)

    try:
        for product in resp.json()['outcomes']:
            if product['prob'] > cutoff:
                products.add(product['smiles'])
    except:
        pass

    return products


class Molecule(object):
    """Define a Markov decision process of generating a molecule"""

    def __init__(self,
                 atom_types,
                 discount_factor,
                 init_mol='C',
                 allow_removal=True,
                 allow_no_modification=True,
                 allow_bonds_between_rings=True,
                 allowed_ring_sizes=None,
                 max_steps=10,
                 target_fn=None,
                 record_path=False,
                 args=None):
        """Initialization of Molecule Generation MDP

        Argument
        --------

            - atom_types. The set of elements molecule may contain.

            - init_mol. String SMILES, Chem.Mol or Chem.RWMol
                If None, the process starts from empty.

            - allow_removal. Boolean.
                Whether allow to remove bonds.

            - allow_no_modification. Boolean.
                Whether allow to retain the molecule.

            - allow_bonds_between_rings. Boolean.
                Whether allow to form a bond between rings.

            - allowed_ring_sizes. Set of integers.
                The size of rings allowed to form.

            - max_steps. Integer.
                Maximum steps allowed to take in one episode.

            - target_fn. Function or None.
                The function should take a SMILES as an input and returns a Boolean
                which indicates whether the input satisfy some criterion. If criterion
                is met, the episode will be terminated.

            - record_path. Boolean.
                Whether to record the steps internally.

        """
        self.discount_factor = discount_factor
        self.allow_no_modification = allow_no_modification
        self.allow_bonds_between_rings = allow_bonds_between_rings
        self.allowed_ring_sizes = allowed_ring_sizes
        self.max_steps = max_steps
        self._state = None
        self._valid_actions = []
        self._counter = self.max_steps
        self._target_fn = target_fn
        self.record_path = record_path
        self._path = []
        self._max_bonds = 4
        self.k = 3
        self._max_new_bonds = dict(list(zip(atom_types, utils.atom_valences(atom_types))))
        with open(
                '/Users/gaowh/PycharmProjects/mol_vl/environments/pricer_using_reaxys_v2-chemicals_and_reaxys_v2-buyables.pkl',
                'rb') as f:
            pricer = pkl.load(f)
        self.buyable = list(pricer)[1:]
        self.HOST = 'https://35.202.13.65'

    @property
    def state(self):
        return self._state

    @property
    def num_steps_taken(self):
        return self._counter

    def get_path(self):
        return self._path

    def reset(self):
        """Reset the MDP to its initial state"""
        self._state = random.sample(self.buyable, 1)[0]
        while '*' in self._state:
            self._state = random.sample(self.buyable, 1)[0]
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0
        return self._state, self._counter

    def get_valid_actions(self, state=None, force_rebuild=False):

        if state is None:
            if self._valid_actions and not force_rebuild:
                return copy.deepcopy(self._valid_actions)
            state = self._state

        if isinstance(state, Chem.Mol):
            state = Chem.MolToSmiles(state)

        self._valid_actions = get_valid_actions(
            state=state,
            buyable=random.sample(self.buyable, self.k),
            allow_no_modification=self.allow_no_modification,
            HOST=self.HOST
        )

        return copy.deepcopy(self._valid_actions)

    def _reward(self):
        """Get the reward of the state"""
        return 0.0

    def _goal_reached(self):
        if self._target_fn is None:
            return False
        return self._target_fn(self._state)

    def step(self, action):
        """Take an action

        Argument
        -----------

            - action. Chem.RWMol.
                The molecule after taking this action

        Return

            - result. Result object.
                * state. The molecule reached after taking the action.
                * reward. The reward get after taking the action.
                * terminated. Whether this episode is terminated.

        """
        if self._counter >= self.max_steps or self._goal_reached():
            raise ValueError('This episode is terminated.')
        if action not in self._valid_actions:
            raise ValueError('Invalid action.')

        self._state = action
        if self.record_path:
            self._path.append(self._state)

        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter += 1

        return self._state, self._counter, self._reward(), (self._counter >= self.max_steps or self._goal_reached())

    def visualize_state(self, state=None, **kwargs):
        """Draw the molecule of the state.

        Argument
        ------------

            - state. SMILES, Chem.Mol or Chem.RWMol
                If None, will take current state as input.

            - kwargs. keywords pass to Draw.MolToImage function.

        Return

            A PIL image containing a drawing of the molecule.

        """
        if state is None:
            state = self._state
        if isinstance(state, str):
            state = Chem.MolFromSmiles(state)
        return Draw.MolToImage(state, **kwargs)


###################################################################################
#                 Single-Objective Optimization Environment                       #
###################################################################################


class TargetWeightMolecule(Molecule):
    """Defines the subclass of a molecule MDP with a target molecular weight."""

    def __init__(self, target_weight, **kwargs):
        super(TargetWeightMolecule, self).__init__(**kwargs)
        self.target_weight = target_weight

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return -self.target_weight**2
        lower, upper = self.target_weight - 25, self.target_weight + 25
        mw = Descriptors.MolWt(molecule)
        if lower <= mw <= upper:
            return 1
        return -min(abs(lower - mw), abs(upper - mw))


class OptQEDMolecule(Molecule):
    """The molecule whose reward is the QED."""

    def __init__(self, discount_factor, **kwargs):
        """Initializes the class.
        Args:
            discount_factor: Float. The discount factor. We only
                care about the molecule at the end of modification.
                In order to prevent a myopic decision, we discount
                the reward at each step by a factor of
                discount_factor ** num_steps_left,
                this encourages exploration with emphasis on long term rewards.
            **kwargs: The keyword arguments passed to the base class.
        """
        super(OptQEDMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor

    def _reward(self):
        """Reward of a state.
        Returns:
            Float. QED of the current state.
        """
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        qed = QED.qed(molecule)
        return qed * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptRawLogPMolecule(Molecule):

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return utils.penalized_logp(molecule)


class OptLogPMolecule(Molecule):

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return 1 / (1 + np.exp(- 0.3 * utils.penalized_logp(molecule)))


class TargetSASMolecule(Molecule):
    """Target SAS reward Molecule."""

    def __init__(self, discount_factor, target_sas, loss_type, **kwargs):
        """Initializes the class.
        Args:
            discount_factor: Float. The discount factor. We only care about the
                molecule at the end of modification. In order to prevent a myopic
                decision, we discount the reward at each step by a factor of
                discount_factor ** num_steps_left, this encourages exploration with
                emphasis on long term rewards.
            target_sas: Float. Target synthetic accessibility value.
            loss_type: String. 'l2' for l2 loss, 'l1' for l1 loss.
            **kwargs: The keyword arguments passed to the base class.
        """
        super(TargetSASMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor
        self.target_sas = target_sas
        if loss_type == 'l1':
            self.loss_fn = abs
        elif loss_type == 'l2':
            self.loss_fn = lambda x: x**2
        else:
            raise ValueError('loss_type must by "l1" or "l2"')

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return -self.loss_fn(self.target_sas)
        sas = sascorer.calculateScore(molecule)
        return -self.loss_fn(sas - self.target_sas) * (self.discount_factor**(self.max_steps - self.num_steps_taken))


class TargetSimilarityQED(Molecule):
    """Defines the subclass of generating a molecule with a specific reward.
    The reward is defined as a scalar
    reward = weight * similarity_score + (1 - weight) *  qed_score
    """

    def __init__(self, target_molecule, similarity_weight, discount_factor, **kwargs):
        """Initializes the class.
        Args:
        target_molecule: SMILES string. The target molecule against which we
        calculate the similarity.
        similarity_weight: Float. The weight applied similarity_score.
        discount_factor: Float. The discount factor applied on reward.
        **kwargs: The keyword arguments passed to the parent class.
        """
        super(TargetSimilarityQED, self).__init__(**kwargs)
        target_molecule = Chem.MolFromSmiles(target_molecule)
        self._target_mol_fingerprint = self.get_fingerprint(target_molecule)
        self._sim_weight = similarity_weight
        self._discount_factor = discount_factor

    def get_fingerprint(self, molecule):
        """Gets the morgan fingerprint of the target molecule.
        Args:
        molecule: Chem.Mol. The current molecule.
        Returns:
        rdkit.ExplicitBitVect. The fingerprint of the target.
        """
        return AllChem.GetMorganFingerprint(molecule, radius=2)

    def get_similarity(self, smiles):
        """Gets the similarity between the current molecule and the target molecule.
        Args:
        smiles: String. The SMILES string for the current molecule.
        Returns:
        Float. The Tanimoto similarity.
        """

        structure = Chem.MolFromSmiles(smiles)
        if structure is None:
            return 0.0
        fingerprint_structure = self.get_fingerprint(structure)

        return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint, fingerprint_structure)

    def _reward(self):
        """Calculates the reward of the current state.
        The reward is defined as a tuple of the similarity and QED value.
        Returns:
        A tuple of the similarity and qed value
        """
        # calculate similarity.
        # if the current molecule does not contain the scaffold of the target,
        # similarity is zero.
        if self._state is None:
            return 0.0
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0
        similarity_score = self.get_similarity(self._state)
        # calculate QED
        qed_value = QED.qed(mol)
        reward = (
            similarity_score * self._sim_weight +
            qed_value * (1 - self._sim_weight)
        )
        discount = self._discount_factor**(self.max_steps - self._counter)
        return reward * discount


###################################################################################
#                 Guacamol-Objective Optimization Environment                     #
###################################################################################


class OptGuacamol(Molecule):

    def __init__(self, scoring_function, **kwargs):
        """Initializes the class.
        Args:
            discount_factor: Float. The discount factor. We only
                care about the molecule at the end of modification.
                In order to prevent a myopic decision, we discount
                the reward at each step by a factor of
                discount_factor ** num_steps_left,
                this encourages exploration with emphasis on long term rewards.
            **kwargs: The keyword arguments passed to the base class.
        """
        super(OptGuacamol, self).__init__(**kwargs)
        self.scoring_function = scoring_function

    def _reward(self):
        scorer = self.scoring_function
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptRediscovery_Celecoxib(Molecule):
    def _reward(self):
        scorer = similarity(
            smiles='CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F',
            name='Celecoxib',
            fp_type='ECFP4',
            threshold=1.0,
            rediscovery=True
        )
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptRediscovery_Troglitazone(Molecule):
    def _reward(self):
        scorer = similarity(
            smiles='Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O',
            name='Troglitazone',
            fp_type='ECFP4',
            threshold=1.0,
            rediscovery=True
        )
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptRediscovery_Thiothixene(Molecule):
    def _reward(self):
        scorer = similarity(
            smiles='CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1',
            name='Thiothixene',
            fp_type='ECFP4',
            threshold=1.0,
            rediscovery=True
        )
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptSimilarity_Aripiprazole(Molecule):
    def _reward(self):
        scorer = similarity(
            smiles='Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl',
            name='Aripiprazole',
            fp_type='ECFP4',
            threshold=0.75
        )
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptSimilarity_Albuterol(Molecule):
    def _reward(self):
        scorer = similarity(
            smiles='CC(C)(C)NCC(O)c1ccc(O)c(CO)c1',
            name='Albuterol',
            fp_type='FCFP4',
            threshold=0.75
        )
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptSimilarity_Mestranol(Molecule):
    def _reward(self):
        scorer = similarity(
            smiles='COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1',
            name='Mestranol',
            fp_type='AP',
            threshold=0.75
        )
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptIsomers_c11h24(Molecule):
    def _reward(self):
        scorer = isomers_c11h24()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptIsomers_c9h10n2o2pf2cl(Molecule):
    def _reward(self):
        scorer = isomers_c9h10n2o2pf2cl()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptMedian1(Molecule):
    def _reward(self):
        scorer = median_camphor_menthol()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptMedian2(Molecule):
    def _reward(self):
        scorer = median_tadalafil_sildenafil()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptMPO_Osimertinib(Molecule):
    def _reward(self):
        scorer = hard_osimertinib()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptMPO_Fexofenadine(Molecule):
    def _reward(self):
        scorer = hard_fexofenadine()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptMPO_Ranolazine(Molecule):
    def _reward(self):
        scorer = ranolazine_mpo()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptMPO_Perindopril(Molecule):
    def _reward(self):
        scorer = perindopril_rings()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptMPO_Amlodipine(Molecule):
    def _reward(self):
        scorer = amlodipine_rings()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptMPO_Sitagliptin(Molecule):
    def _reward(self):
        scorer = sitagliptin_replacement()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptMPO_Zaleplon(Molecule):
    def _reward(self):
        scorer = zaleplon_with_other_formula()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptMPO_Valsartan(Molecule):
    def _reward(self):
        scorer = valsartan_smarts()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptHop_Deco(Molecule):
    def _reward(self):
        scorer = decoration_hop()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class OptHop_Scaffold(Molecule):
    def _reward(self):
        scorer = scaffold_hop()
        s_fn = scorer.wrapped_objective
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return s_fn.score(self._state) * self.discount_factor ** (self.max_steps - self.num_steps_taken)


###################################################################################
#                 Multi-Objective Optimization Environment                        #
###################################################################################

class MultiObjectiveRewardMolecule(Molecule):
    """Defines the subclass of generating a molecule with a specific reward.
    The reward is defined as a 1-D vector with 2 entries: similarity and QED
    reward = (similarity_score, qed_score)
    """

    def __init__(self, target_molecule, **kwargs):
        super(MultiObjectiveRewardMolecule, self).__init__(**kwargs)
        target_molecule = Chem.MolFromSmiles(target_molecule)
        self._target_mol_fingerprint = self.get_fingerprint(target_molecule)
        self._target_mol_scaffold = utils.get_scaffold(target_molecule)
        self.reward_dim = 2

    def get_fingerprint(self, molecule):
        return AllChem.GetMorganFingerprint(molecule, radius=2)

    def get_similarity(self, smiles):
        structure = Chem.MolFromSmiles(smiles)
        if structure is None:
            return 0.0
        fingerprint_structure = self.get_fingerprint(structure)

        return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint, fingerprint_structure)

    def _reward(self):
        # calculate similarity.
        # if the current molecule does not contain the scaffold of the target,
        # similarity is zero.
        if self._state is None:
            return 0.0, 0.0

        mol = Chem.MolFromSmiles(self._state)

        if mol is None:
            return 0.0, 0.0

        if utils.contains_scaffold(mol, self._target_mol_scaffold):
            similarity_score = self.get_similarity(self._state)
        else:
            similarity_score = 0.0

        qed_value = QED.qed(mol)
        return similarity_score, qed_value
