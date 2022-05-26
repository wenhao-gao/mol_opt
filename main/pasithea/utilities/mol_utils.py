"""
Utilities for handling molecular properties and the conversion between
molecular representations.
"""
import re
import numpy as np
import sys
sys.path.append('datasets')
import selfies as sf
import torch

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from torch import rand
from utilities.utils import make_dir


def edit_hot(hot, upper_bound):
    """Replaces all zeroes with a random float in the range [0,upper_bound]"""
    #t1=time.clock()
    newhot=hot+upper_bound*rand(hot.shape).to(hot.device)
    newhot[newhot>1]=1
    return newhot


def logP_from_molecule(smiles_list):
    """Calculate list of logP from list of SMILES molecules"""
    logP_vals=[]
    for ii in range(len(smiles_list)):
        try:
            res_molecule = MolFromSmiles(smiles_list[ii])
        except Exception:
            res_molecule=None

        if res_molecule==None:
            logP_vals.append(-666)
        else:
            logP_vals.append(Descriptors.MolLogP(res_molecule))
    return logP_vals


def lst_of_logP(hot, encoding_alphabet):
    """Calculate list of logP from list of one-hot encodings"""
    logP_lst = []
    smiles_mol_lst = []
    for mol in hot:
        molec = sf.decoder(indices_to_selfies(mol, encoding_alphabet))
        smiles_mol_lst.append(molec)
        try:
            res_molecule = MolFromSmiles(molec)
        except Exception:
            res_molecule=None

        if res_molecule==None:
            logP_lst.append(-666)
        else:
            logP_lst.append(Descriptors.MolLogP(res_molecule))
    return logP_lst, smiles_mol_lst


def smile_to_hot(smile, largest_smile_len, alphabet):
    """
    Convert a single smile string to a one-hot encoding.
    """
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with ' '
    smile += ' ' * (largest_smile_len - len(smile))

    # integer encode input smile
    integer_encoded = [char_to_int[char] for char in smile]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)


def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """
    Convert a list of smile strings to a one-hot encoding

    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """
    hot_list = []
    for smile in smiles_list:
        _, onehot_encoded = smile_to_hot(smile, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)


def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """
    Convert a single selfies string to a one-hot encoding.
    """
    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    # return integer_encoded, onehot_encoded
    return integer_encoded, np.array(onehot_encoded)


def logp(molecule):
    """
    Calculate the logP of the selfies string
    """
    m = MolFromSmiles(sf.decoder(molecule))
    return rdMolDescriptors.CalcCrippenDescriptors(m)[0]


def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to a one-hot encoding
    """

    hot_list = []
    for s in selfies_list:
        try:
            _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
            hot_list.append(onehot_encoded)
        except:
            pass
    return np.array(hot_list)


def multiple_selfies_to_hot_online(selfies_list, value_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to a one-hot encoding
    """

    hot_list = []
    val_list = []
    for i, s in enumerate(selfies_list):
        try:
            _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
            hot_list.append(onehot_encoded)
            val_list.append(value_list[i])
        except:
            pass
    return np.array(hot_list), val_list


def hot_to_indices(hot):
    """Convert an element of one-hot encoding to indices
    """
    hot = torch.tensor(hot)
    _,max_index=hot.max(1)
    return max_index.data.cpu().numpy().tolist()


def multiple_hot_to_indices(hots):
    """Convert one-hot encoding to list of indices
    """
    gathered_ints=[]
    for hot in hots:
        gathered_ints.append(hot_to_indices(hot))
    return gathered_ints


def indices_to_selfies(hot, alphabet):
    """Convert list of indices to selfies string
    """
    int_to_symbol = dict((i, c) for i, c in enumerate(alphabet))

    selfie = ''
    for num in hot:
        selfie += int_to_symbol[num]
    return selfie


def multiple_indices_to_selfies(hots, alphabet):
    """Convert multiple lists of indices to selfies string
    """
    selfies_list = []
    for hot in hots:
        selfies_list.append(indices_to_selfies(hot, alphabet))
    return np.array(selfies_list)


def draw_mol_to_file(mol_lst, directory):
    """Saves the pictorial representation of each molecule in the list to
    file."""

    directory = 'dream_results/mol_pics'
    make_dir(directory)
    for smiles in mol_lst:
        mol = MolFromSmiles(smiles)
        Draw.MolToFile(mol,directory+'/'+smiles+'.pdf')
