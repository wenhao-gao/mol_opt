
import typing
import itertools

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

import numpy as np


def get_atom_map_nums(rxn_str) -> typing.Set[int]:
    """
    :return: set of the atom mapping numbers of the atoms in the reaction string
    """
    mol = Chem.MolFromSmiles(rxn_str)
    return set([a.GetPropsAsDict()['molAtomMapNumber'] for a in mol.GetAtoms()])


def get_mol_props(mol: AllChem.Mol):
    """
    Get the properties of a molecule.
    """
    logP = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    alpha = rdMolDescriptors.CalcHallKierAlpha(mol)
    MR = Descriptors.MolMR(mol)
    asa = rdMolDescriptors.CalcLabuteASA(mol)
    return [logP, tpsa, alpha, MR, asa]


def get_molecule(molecule_strs, kekulize=True) -> AllChem.Mol:
    """
    Convert string to molecule
    """
    mol = Chem.MolFromSmiles(molecule_strs)
    if kekulize:
        Chem.Kekulize(mol)
    return mol


def add_atom_mapping(mol) -> typing.Tuple[AllChem.Mol, dict]:
    """
    add atom mappings to molecule
    """
    atom_map_to_index_map = {}
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp('molAtomMapNumber', str(i))  # the atom mapping in the file
        atom_map_to_index_map[i] = i
    return mol, atom_map_to_index_map


def get_atoms_names_charge_and_h_count(mol, atom_mappings: set) -> typing.Mapping[int, tuple]:
    """
    for all atoms with atom mapping numbers present in the `atom_mappings` set put in a dictionary indexed by the atom
    mapping tuples of the atom symbol, formal charge and total number of Hydrogen atoms.
    """
    results = {}
    for atom in mol.GetAtoms():
        props = atom.GetPropsAsDict()
        am = props['molAtomMapNumber']  # the atom mapping in the file
        if am in atom_mappings:
            results[am] = (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs())
            atom_mappings.remove(am)
            if len(atom_mappings) == 0:
                break  # found them all so can quit!
    return results


def create_atom_map_indcs_map(mol) -> typing.Mapping[int, int]:
    """
    return a dictionarry from atom mapping to the rdkit atom index
    """
    return {atom.GetPropsAsDict()['molAtomMapNumber']: idx for idx, atom in
            enumerate(mol.GetAtoms())}


def get_bond_double_between_atom_mapped_atoms(mol, am_start, am_end, am_to_indcs_map=None) -> float:
    """
    Return bond double between the two atoms with atom mappings given by arguments or 0. if no bond exists.
    :param am_to_indcs_map: atom map to atom index map (if not given will be created)
    """
    am_to_indcs_map = am_to_indcs_map or create_atom_map_indcs_map(mol)  # create if necessary
    try:
        new_bond = get_bond_between_indx_atoms(mol, am_to_indcs_map[am_start], am_to_indcs_map[am_end])
    except KeyError:
        new_bond = 0.
    return new_bond


def get_bond_between_indx_atoms(mol, idx_start, idx_end) -> float:
    """
    Return bond double between the two atoms with rdkit indices mappings given by arguments or 0. if no bond exists.
    """
    bnd = mol.GetBondBetweenAtoms(idx_start, idx_end)
    bnd = bnd.GetBondTypeAsDouble() if bnd is not None else 0.
    return bnd


def return_canoncailised_smiles_str(molecule, remove_am=True, allHsExplicit=False, kekuleSmiles=True) -> str:
    """
    Rdkit molecule to smiles str,
    """
    mol_copy = Chem.RWMol(molecule)
    if remove_am:
        for atom in mol_copy.GetAtoms():
            atom.ClearProp('molAtomMapNumber')
    smiles = Chem.MolToSmiles(mol_copy, allHsExplicit=allHsExplicit, kekuleSmiles=kekuleSmiles, canonical=True)
    return smiles


def canconicalize(mol_smi, remove_am=False):
    return return_canoncailised_smiles_str(get_molecule(mol_smi, kekulize=False), kekuleSmiles=False, remove_am=remove_am)


def get_fingerprint_as_array(molecule, radius=4, nbits=2048):
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(
    AllChem.GetMorganFingerprintAsBitVect(molecule, radius=radius, nBits=nbits), arr)
    return arr

