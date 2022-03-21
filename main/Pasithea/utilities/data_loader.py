"""
Methods for reading and preprocessing dataset of SMILES molecular strings.
"""

import selfies as sf
import numpy as np
import pandas as pd
import os
from utilities import utils
from utilities import mol_utils


def get_largest_selfie_len(smiles_list):
    """Returns the length of the largest SELFIES string from a list of SMILES."""

    selfies_list = list(map(sf.encoder, smiles_list))
    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)
    return largest_selfies_len


def get_largest_string_len(smiles_list, filename):
    """Returns the length of the largest SELFIES or SMILES string from a list
    of SMILES. If this dataset has been used already,
    then these values will be accessed from a corresponding file."""

    directory = 'dataset_encoding'
    name = directory + '/encoding_info_'
    i = len(filename) - 1
    dataset = ''
    while i >= 0 and filename[i] != '/':
        dataset = filename[i] + dataset
        i -= 1
    name = name + dataset

    largest_smiles_len = -1
    largest_selfies_len = -1

    if os.path.exists(name):
        f = open(name, "r")
        largest_smiles_len = f.readline()
        largest_smiles_len = int(
            largest_smiles_len[0:len(largest_smiles_len) - 1])

        largest_selfies_len = f.readline()
        largest_selfies_len = int(
            largest_selfies_len[0:len(largest_selfies_len) - 1])
        f.close()
    else:
        utils.make_dir(directory)
        f = open(name, "w+")
        largest_smiles_len = len(max(smiles_list, key=len))
        f.write(str(largest_smiles_len) + '\n')
        largest_selfies_len = get_largest_selfie_len(smiles_list)
        f.write(str(largest_selfies_len) + '\n')
        f.close()

    return (largest_smiles_len, largest_selfies_len)


def get_selfies_alphabet(smiles_list):
    """Returns a sorted list of all SELFIES tokens required to build a
    SELFIES string for each molecule."""

    selfies_list = list(map(sf.encoder, smiles_list))
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')
    selfies_alphabet = list(all_selfies_symbols)
    selfies_alphabet.sort()
    return selfies_alphabet


def get_string_alphabet(smiles_list, filename):
    """Returns a sorted list of all SELFIES tokens and SMILES tokens required
    to build a string representation of each molecule. If this dataset has
    already been used, then these will be accessed from a correspondning file."""

    directory = 'dataset_encoding'
    name1 = directory + '/smiles_alphabet_info_'
    name2 = directory + '/selfies_alphabet_info_'
    i = len(filename) - 1
    dataset = ''
    while i >= 0 and filename[i] != '/':
        dataset = filename[i] + dataset
        i -= 1
    name1 = name1 + dataset
    name2 = name2 + dataset
    selfies_alphabet = []
    smiles_alphabet = []
    # if os.path.exists(name1):
    #     df = pd.read_csv(name1)
    #     smiles_alphabet = np.asanyarray(df.alphabet)
    #     df = pd.read_csv(name2)
    #     selfies_alphabet = np.asanyarray(df.alphabet)
    # else:
    if True: 
        utils.make_dir(directory)
        f = open(name1, "w+")
        f.write('alphabet\n')
        smiles_alphabet = list(set(''.join(smiles_list)))
        smiles_alphabet.append(' ')  # for padding
        smiles_alphabet.sort()
        for s in smiles_alphabet:
            f.write(s + '\n')
        f.close()
        f = open(name2, "w+")
        f.write('alphabet\n')
        selfies_alphabet = get_selfies_alphabet(smiles_list)
        for s in selfies_alphabet:
            f.write(s + '\n')
        f.close()

    return smiles_alphabet, selfies_alphabet


def get_selfie_and_smiles_info(smiles_list, filename):
    """Returns the length of the largest string representation and the list
    of tokens required to build a string representation of each molecule."""

    largest_smiles_len, largest_selfies_len = get_largest_string_len(
        smiles_list,
        filename)
    smiles_alphabet, selfies_alphabet = get_string_alphabet(smiles_list,
                                                            filename)
    return selfies_alphabet, largest_selfies_len, smiles_alphabet, largest_smiles_len


def get_selfie_and_smiles_encodings(smiles_list, nrows=-1):
    """
    Returns encoding of largest molecule in
    SMILES and SELFIES, given a list of SMILES molecules.
    input:
        - list of SMILES
        - number of rows to be read.
    output:
        - selfies encoding
        - smiles encoding
    """

    if nrows > -1:
        smiles_list = np.random.choice(smiles_list, nrows, replace=False)
    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))

    print('Finished translating SMILES to SELFIES.')

    return selfies_list, smiles_list


def read_smiles(filename):
    """Returns the list of SMILES from a csv file of molecules.
    Column's name must be 'smiles'."""

    df = pd.read_csv(filename)
    smiles_list = np.asanyarray(df.smiles)
    return smiles_list


def preprocess(num_mol, file_name):
    """Takes a random subset of num_mol SMILES from a given dataset;
    converts each SMILES to the SELFIES equivalent and one-hot encoding;
    encodes other string information."""

    smiles_list = read_smiles(file_name)
    selfies_alphabet, largest_selfies_len, smiles_alphabet, largest_smiles_len \
        = get_selfie_and_smiles_info(smiles_list, file_name)
    selfies_list, smiles_list = \
        get_selfie_and_smiles_encodings(smiles_list, num_mol)

    print('Finished acquiring data.\n')
    print('Calculating logP of all molecules...')
    prop_vals = mol_utils.logP_from_molecule(smiles_list)

    prop_vals = np.array(prop_vals)
    print('Finished calculating logP of all molecules.\n')

    print('Representation: SELFIES')
    alphabet = selfies_alphabet
    encoding_list = selfies_list
    largest_molecule_len = largest_selfies_len
    print('--> Creating one-hot encoding...')
    data = mol_utils.multiple_selfies_to_hot(encoding_list,
                                             largest_molecule_len,
                                             alphabet)
    print('    Finished creating one-hot encoding.\n')

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_molec1Hot = len_max_molec * len_alphabet
    print(' ')
    print('Alphabet has ', len_alphabet, ' letters, largest molecule is ',
          len_max_molec, ' letters.')

    return data, prop_vals, alphabet, len_max_molec1Hot, largest_molecule_len


def split_train_test(data, prop_vals, num_mol, frac_train):
    """Split data into training and test data. frac_train is the fraction of
    data used for training. 1-frac_train is the fraction used for testing."""

    train_test_size = [frac_train, 1 - frac_train]
    data = data[:num_mol]
    prop_vals = prop_vals[:num_mol]

    idx_traintest = int(len(data) * train_test_size[0])
    idx_trainvalid = idx_traintest + int(len(data) * train_test_size[1])
    data_train = data[0:idx_traintest]
    prop_vals_train = prop_vals[0:idx_traintest]

    data_test = data[idx_traintest:idx_trainvalid]
    prop_vals_test = prop_vals[idx_traintest:idx_trainvalid]

    return data_train, data_test, prop_vals_train, prop_vals_test
