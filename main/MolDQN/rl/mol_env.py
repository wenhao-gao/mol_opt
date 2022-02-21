from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import itertools

from rdkit import Chem
from rdkit.Chem import Draw
from six.moves import range
from six.moves import zip

from rl import mol_utils


class Result(collections.namedtuple('Result', ['state', 'reward', 'terminated'])):
    """A named tuple define teh result for a step for the molecular class

    Argument
    ----------

        - state. Chem.RWMol.
            The molecule reached after taking the action

        - reward. Float.
            The reward get after taking the action.

        - terminated. Boolean.
            Whether this episode is terminated.

    """


def get_valid_actions(state,
                      atom_types,
                      allow_removal,
                      allow_no_modification,
                      allowed_ring_sizes,
                      allow_bonds_between_rings):
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
    if not state:
        return copy.deepcopy(atom_types)

    mol = Chem.MolFromSmiles(state)
    if mol is None:
        raise ValueError('Received invalid state: %s' % state)

    # Get additional atoms valences
    atom_valences = {
        atom_type: mol_utils.atom_valences([atom_type])[0]
        for atom_type in atom_types
    }

    # Get available atom location
    atoms_with_free_valence = {}
    for i in range(1, max(atom_valences.values())):
        atoms_with_free_valence[i] = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= i
        ]

    # Get valid actions
    valid_actions = set()

    valid_actions.update(
        _atom_addition(
            mol,
            atom_types=atom_types,
            atom_valences=atom_valences,
            atoms_with_free_valence=atoms_with_free_valence
        )
    )

    valid_actions.update(
        _bond_addition(
            mol,
            atoms_with_free_valence=atoms_with_free_valence,
            allowed_ring_sizes=allowed_ring_sizes,
            allow_bonds_between_rings=allow_bonds_between_rings
        )
    )

    if allow_removal:
        valid_actions.update(_bond_removal(mol))

    if allow_no_modification:
        valid_actions.add(Chem.MolToSmiles(mol))

    return valid_actions


def _atom_addition(state, atom_types, atom_valences, atoms_with_free_valence):
    """Compute valid atom addition operation"""

    bond_orders = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }

    atom_addition = set()

    for i in bond_orders:
        for atom in atoms_with_free_valence[i]:
            for element in atom_types:
                if atom_valences[element] >= i:
                    new_state = Chem.RWMol(state)
                    idx = new_state.AddAtom(Chem.Atom(element))
                    new_state.AddBond(atom, idx, bond_orders[i])
                    sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                    if sanitization_result:
                        continue
                    atom_addition.add(Chem.MolToSmiles(new_state))

    return atom_addition


def _bond_addition(state, atoms_with_free_valence, allowed_ring_sizes, allow_bonds_between_rings):
    """Compute valid bond addition operation"""

    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE
    ]

    bond_addition = set()

    for valence, atoms in atoms_with_free_valence.items():

        for atom1, atom2 in itertools.combinations(atoms, 2):

            bond = Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2)
            new_state = Chem.RWMol(state)
            Chem.Kekulize(new_state, clearAromaticFlags=True)

            if bond is not None:

                if bond.GetBondType() not in bond_orders:
                    continue # skip aromatic bond

                # idx = bond.GetIdx()
                bond_order = bond_orders.index(bond.GetBondType())
                bond_order += valence

                if bond_order < len(bond_orders):
                    idx = bond.GetIdx()
                    bond.SetBondType(bond_orders[bond_order])
                    new_state.ReplaceBond(idx, bond)
                else:
                    continue

            elif (not allow_bonds_between_rings and
                  (state.GetAtomWithIdx(atom1).IsInRing() and state.GetAtomWithIdx(atom2).IsInRing())):
                continue

            elif (allowed_ring_sizes is not None and
                  len(Chem.rdmolops.GetShortestPath(state, atom1, atom2)) not in allowed_ring_sizes):
                continue

            else:
                new_state.AddBond(atom1, atom2, bond_orders[valence])

            sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
            if sanitization_result:
                continue
            bond_addition.add(Chem.MolToSmiles(new_state))

    return bond_addition


def _bond_removal(state):
    """Compute the valid bond removal """

    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE
    ]

    bond_removal = set()

    for valence in [1, 2, 3]:

        for bond in state.GetBonds():

            bond = Chem.Mol(state).GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

            if bond.GetBondType() not in bond_orders:
                continue

            new_state = Chem.RWMol(state)
            Chem.Kekulize(new_state, clearAromaticFlags=True)
            bond_order = bond_orders.index(bond.GetBondType())
            bond_order -= valence

            if bond_order > 0:

                idx = bond.GetIdx()
                bond.SetBondType(bond_orders[bond_order])
                new_state.ReplaceBond(idx, bond)

                sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                if sanitization_result:
                    continue

                bond_removal.add(Chem.MolToSmiles(new_state))

            elif bond_order == 0:
                atom1 = bond.GetBeginAtom().GetIdx()
                atom2 = bond.GetEndAtom().GetIdx()
                new_state.RemoveBond(atom1, atom2)
                sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                if sanitization_result:
                    continue

                smiles = Chem.MolToSmiles(new_state)
                parts = sorted(smiles.split('.'), key=len)

                if len(parts) == 1 or len(parts[0]) == 1:
                    bond_removal.add(parts[-1])

    return bond_removal


class Molecule(object):
    """Define a Markov decision process of generating a molecule"""

    def __init__(self,
                 atom_types,
                 init_mol=None,
                 allow_removal=True,
                 allow_no_modification=True,
                 allow_bonds_between_rings=True,
                 allowed_ring_sizes=None,
                 max_steps=10,
                 target_fn=None,
                 record_path=False):
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

        if isinstance(init_mol, Chem.Mol):
            init_mol = Chem.MolToSmiles(init_mol)

        self.init_mol = init_mol
        self.atom_types = atom_types
        self.allow_removal = allow_removal
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
        atom_types = list(self.atom_types)
        self._max_new_bonds = dict(list(zip(atom_types, mol_utils.atom_valences(atom_types))))

    @property
    def state(self):
        return self._state

    @property
    def num_steps_taken(self):
        return self._counter

    def get_path(self):
        return self._path

    def initialize(self):
        """Reset the MDP to its initial state"""
        self._state = self.init_mol
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0

    def get_valid_actions(self, state=None, force_rebuild=False):

        if state is None:
            if self._valid_actions and not force_rebuild:
                return copy.deepcopy(self._valid_actions)
            state = self._state

        if isinstance(state, Chem.Mol):
            state = Chem.MolToSmiles(state)

        self._valid_actions = get_valid_actions(
            state,
            atom_types=self.atom_types,
            allow_removal=self.allow_removal,
            allow_no_modification=self.allow_no_modification,
            allowed_ring_sizes=self.allowed_ring_sizes,
            allow_bonds_between_rings=self.allow_bonds_between_rings
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

        result = Result(
            state=self._state,
            reward=self._reward(),
            terminated=(self._counter >= self.max_steps or self._goal_reached())
        )

        return result

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
