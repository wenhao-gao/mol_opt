"""The specific optimization environment for DQN"""

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED

from rl import mol_env
from rl import mol_utils

###################################################################################
#                 Single-Objective Optimization Environment                       #
###################################################################################


class TargetWeightMolecule(mol_env.Molecule):
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


class OptQEDMolecule(mol_env.Molecule):
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


class OptLogPMolecule(mol_env.Molecule):

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return mol_utils.penalized_logp(molecule)


class TargetSASMolecule(mol_env.Molecule):
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


class TargetSimilarityQED(mol_env.Molecule):
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
#                 Multi-Objective Optimization Environment                        #
###################################################################################

class MultiObjectiveRewardMolecule(mol_env.Molecule):
    """Defines the subclass of generating a molecule with a specific reward.
    The reward is defined as a 1-D vector with 2 entries: similarity and QED
    reward = (similarity_score, qed_score)
    """

    def __init__(self, target_molecule, **kwargs):
        super(MultiObjectiveRewardMolecule, self).__init__(**kwargs)
        target_molecule = Chem.MolFromSmiles(target_molecule)
        self._target_mol_fingerprint = self.get_fingerprint(target_molecule)
        self._target_mol_scaffold = mol_utils.get_scaffold(target_molecule)
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

        if mol_utils.contains_scaffold(mol, self._target_mol_scaffold):
            similarity_score = self.get_similarity(self._state)
        else:
            similarity_score = 0.0

        qed_value = QED.qed(mol)
        return similarity_score, qed_value
