import io
import re
import os
import gzip
import torch
import logging
import random
import numpy as np

from rdkit import Chem
import rdkit.Chem.Draw as Draw

logger = logging.getLogger('utils')


def to_tensor(tensor):
    if isinstance(tensor, list):
        tensor = np.asarray(tensor)
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def set_default_device_cuda(device='gpu'):
    """Sets the default device (cpu or cuda) used for all tensors."""
    if not torch.cuda.is_available() or (device == 'cpu'):
        device = torch.device('cpu')
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        return device
    elif (device in ['gpu', 'cuda']) and torch.cuda.is_available():  # device_name == "cuda":
        device = torch.device('cuda')
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        torch.set_default_tensor_type(tensor)
        return device
    elif torch.cuda.is_available(): # Assume an index
        raise NotImplementedError


def read_smiles(file_path):
    """Read a smiles file separated by \n"""
    if any(['gz' in ext for ext in os.path.basename(file_path).split('.')[1:]]):
        logger.debug('\'gz\' found in file path: using gzip')
        with gzip.open(file_path) as f:
            smiles = f.read().splitlines()
            smiles = [smi.decode('utf-8') for smi in smiles]
    else:
        with open(file_path, 'rt') as f:
            smiles = f.read().splitlines()
    return smiles


def save_smiles(smiles, file_path):
    """Save smiles to a file path seperated by \n"""
    if (not os.path.exists(os.path.dirname(file_path))) and (os.path.dirname(file_path) != ''):
        os.makedirs(os.path.dirname(file_path))
    if any(['gz' in ext for ext in os.path.basename(file_path).split('.')[1:]]):
        logger.debug('\'gz\' found in file path: using gzip')
        with gzip.open(file_path, 'wb') as f:
            _ = [f.write((smi+'\n').encode('utf-8')) for smi in smiles if smi is not None]
    else:
        with open(file_path, 'wt') as f:
            _ = [f.write(smi+'\n') for smi in smiles if smi is not None]
    return


def fraction_valid_smiles(smiles):
    i = 0
    mols = []
    for smile in smiles:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                i += 1
                mols.append(mol)
            else:
                mols.append(None)
        except TypeError:  # None passed as smile
            mols.append(None)
            pass
    fraction = 100 * i / len(smiles)
    return round(fraction, 2), mols


def mol_image(mol, size=(300, 300)):
    """
    Adds a molecule to the images section of Tensorboard.
    """
    image = Draw.MolToImage(mol, size=size)
    return image


def mols_image(mols, mols_per_row=1, legends=None, size_per_mol=(300, 300)):
    """
    Adds molecules in a grid.
    """
    image = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=size_per_mol, legends=legends)
    return image


def get_image(image):
    """
    Adds an image from a PIL image.
    """
    channel = len(image.getbands())
    width, height = image.size

    output = io.BytesIO()
    image.save(output, format='png')
    image_string = output.getvalue()
    output.close()

    return image_string


def randomize_smiles(smi, n_rand=10, random_type="restricted"):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param smi: A SMILES string
    :param n_rand: Number of randomized smiles per molecule
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :param keep_last: Whether to try and enforce the last atom as always the last atom, not always possible so checked at a later point
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    assert random_type in ['restricted', 'unrestricted'], f"Type {random_type} is not valid"
    mol = Chem.MolFromSmiles(smi)

    if not mol:
        return None

    if random_type == "unrestricted":
        rand_smiles = []
        for i in range(n_rand):
            rand_smiles.append(Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False))
        return list(set(rand_smiles))

    if random_type == "restricted":
        rand_smiles = []
        for i in range(n_rand):
            new_atom_order = list(range(mol.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
            rand_smiles.append(Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False))
        return list(set(rand_smiles))
