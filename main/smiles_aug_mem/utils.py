import torch
import numpy as np
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import RenumberAtoms
import random

from vocabulary import SMILESTokenizer

_ST = SMILESTokenizer()


def get_randomized_smiles(smiles_list, prior) -> list:
    """takes a list of SMILES and returns a list of randomized SMILES"""
    randomized_smiles_list = []
    for smiles in smiles_list:
        mol = MolFromSmiles(smiles)
        if mol:
            try:
                randomized_smiles = randomize_smiles(mol)
                # there may be tokens in the randomized SMILES that are not in the Vocabulary
                # check if the randomized SMILES can be encoded
                tokens = _ST.tokenize(randomized_smiles)
                encoded = prior.vocabulary.encode(tokens)
                randomized_smiles_list.append(randomized_smiles)
            except KeyError:
                randomized_smiles_list.append(smiles)
        else:
            randomized_smiles_list.append(smiles)

    return randomized_smiles_list


def randomize_smiles(mol) -> str:
    """
    Returns a randomized SMILES given an RDKit Mol object.
    :param mol: An RDKit Mol object
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    from reinvent-chemistry
    """
    new_atom_order = list(range(mol.GetNumHeavyAtoms()))
    # reinvent-chemistry uses random.shuffle
    # use np.random.shuffle for reproducibility since PMO fixes the np seed
    np.random.shuffle(new_atom_order)
    random_mol = RenumberAtoms(mol, newOrder=new_atom_order)
    return MolToSmiles(random_mol, canonical=False, isomericSmiles=False)


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)
