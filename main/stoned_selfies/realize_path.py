"""
Created on Thu Aug 29 11:12:51 2020

@author: akshat
"""

import os
import numpy as np 
import random
from random import randrange
import matplotlib.pyplot as plt
import rdkit
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from selfies import encoder, decoder 
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def get_ECFP4(mol):
    ''' Return rdkit ECFP4 fingerprint object for mol

    Parameters: 
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  

    Returns: 
    rdkit ECFP4 fingerprint object for mol
    '''
    return AllChem.GetMorganFingerprint(mol, 2)

def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    
    Parameters:
    smi (string) : smile string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)

def get_fp_scores(smiles_back, target_smi): 
    '''Calculate the Tanimoto fingerprint (ECFP4 fingerint) similarity between a list 
       of SMILES and a known target structure (target_smi). 
       
    Parameters:
    smiles_back   (list) : A list of valid SMILES strings 
    target_smi (string)  : A valid SMILES string. Each smile in 'smiles_back' will be compared to this stucture
    
    Returns: 
    smiles_back_scores (list of floats) : List of fingerprint similarities
    '''
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    fp_target = get_ECFP4(target)
    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = get_ECFP4(mol)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores


def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie


def randomize_smiles(mol):
    '''Returns a random (dearomatized) SMILES given an rdkit mol object of a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol) :  RdKit mol object (None if invalid smile string smi)
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  (None if invalid smile string smi)
    '''
    if not mol:
        return None

    Chem.Kekulize(mol)
    
    return rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True)


def get_random_smiles(smi, num_random_samples): 
    ''' Obtain 'num_random_samples' non-unique SMILES orderings of smi
    
    Parameters:
    smi (string)            : Input SMILES string (needs to be a valid molecule)
    num_random_samples (int): Number fo unique different SMILES orderings to form 
    
    Returns:
    randomized_smile_orderings (list) : list of SMILES strings
    '''
    mol = Chem.MolFromSmiles(smi)
    if mol == None: 
        raise Exception('Invalid starting structure encountered')
    randomized_smile_orderings  = [randomize_smiles(mol) for _ in range(num_random_samples)]
    randomized_smile_orderings  = list(set(randomized_smile_orderings)) # Only consider unique SMILE strings
    return randomized_smile_orderings


def obtain_path(starting_smile, target_smile, filter_path=False): 
    ''' Obtain a path/chemical path from starting_smile to target_smile
    
    Parameters:
    starting_smile (string) : SMILES string (needs to be a valid molecule)
    target_smile (int)      : SMILES string (needs to be a valid molecule)
    filter_path (bool)      : If True, a chemical path is returned, else only a path
    
    Returns:
    path_smiles (list)                  : A list of smiles in path between starting_smile & target_smile
    path_fp_scores (list of floats)     : Fingerprint similarity to 'target_smile' for each smiles in path_smiles
    smiles_path (list)                  : A list of smiles in CHEMICAL path between starting_smile & target_smile (if filter_path==False, then empty)
    filtered_path_score (list of floats): Fingerprint similarity to 'target_smile' for each smiles in smiles_path (if filter_path==False, then empty)
    '''
    starting_selfie = encoder(starting_smile)
    target_selfie   = encoder(target_smile)
    
    starting_selfie_chars = get_selfie_chars(starting_selfie)
    target_selfie_chars   = get_selfie_chars(target_selfie)
    
    # Pad the smaller string
    if len(starting_selfie_chars) < len(target_selfie_chars): 
        for _ in range(len(target_selfie_chars)-len(starting_selfie_chars)):
            starting_selfie_chars.append(' ')
    else: 
        for _ in range(len(starting_selfie_chars)-len(target_selfie_chars)):
            target_selfie_chars.append(' ')
    
    indices_diff = [i for i in range(len(starting_selfie_chars)) if starting_selfie_chars[i] != target_selfie_chars[i]]
    path         = {}
    path[0]  = starting_selfie_chars
    
    for iter_ in range(len(indices_diff)): 
        idx = np.random.choice(indices_diff, 1)[0] # Index to be operated on
        indices_diff.remove(idx)                   # Remove that index
        
        # Select the last member of path: 
        path_member = path[iter_].copy()
        
        # Mutate that character to the correct value: 
        path_member[idx] = target_selfie_chars[idx]
        path[iter_+1] = path_member.copy()
    
    # Collapse path to make them into SELFIE strings
    paths_selfies = []
    for i in range(len(path)):
        selfie_str = ''.join(x for x in path[i])
        paths_selfies.append(selfie_str.replace(' ', ''))
        
    if paths_selfies[-1] != target_selfie: 
        raise Exception("Unable to discover target structure!")
    
    # Obtain similarity scores, and only choose the increasing members: 
    path_smiles         = [decoder(x) for x in paths_selfies]
    path_fp_scores      = []
    filtered_path_score = []
    smiles_path         = []
    
    if filter_path: 
        path_fp_scores = get_fp_scores(path_smiles, target_smile)
        
        filtered_path_score = []
        smiles_path   = []
        for i in range(1, len(path_fp_scores)-1): 
            if i == 1: 
                filtered_path_score.append(path_fp_scores[1])
                smiles_path.append(path_smiles[i])
                continue
            if filtered_path_score[-1] < path_fp_scores[i]:
                filtered_path_score.append(path_fp_scores[i])
                smiles_path.append(path_smiles[i])

    return path_smiles, path_fp_scores, smiles_path, filtered_path_score


def get_compr_paths(starting_smile, target_smile, num_tries, num_random_samples, collect_bidirectional):
    ''' Obtaining multiple paths/chemical paths from starting_smile to target_smile. 
    
    Parameters:
    starting_smile (string)     : SMILES string (needs to be a valid molecule)
    target_smile (int)          : SMILES string (needs to be a valid molecule)
    num_tries (int)             : Number of path/chemical path attempts between the exact same smiles
    num_random_samples (int)    : Number of different SMILES string orderings to conside for starting_smile & target_smile 
    collect_bidirectional (bool): If true, forms paths from target_smiles-> target_smiles (doubles number of paths)
    
    Returns:
    smiles_paths_dir1 (list): list paths containing smiles in path between starting_smile -> target_smile
    smiles_paths_dir2 (list): list paths containing smiles in path between target_smile -> starting_smile
    '''
    starting_smile_rand_ord = get_random_smiles(starting_smile, num_random_samples=num_random_samples)
    target_smile_rand_ord   = get_random_smiles(target_smile,   num_random_samples=num_random_samples)
    
    smiles_paths_dir1 = [] # All paths from starting_smile -> target_smile
    for smi_start in starting_smile_rand_ord: 
        for smi_target in target_smile_rand_ord: 
            
            if Chem.MolFromSmiles(smi_start) == None or Chem.MolFromSmiles(smi_target) == None: 
                raise Exception('Invalid structures')
                
            for _ in range(num_tries): 
                path, _, _, _ = obtain_path(smi_start, smi_target, filter_path=True)
                smiles_paths_dir1.append(path)
    
    smiles_paths_dir2 = [] # All paths from starting_smile -> target_smile
    if collect_bidirectional == True: 
        starting_smile_rand_ord = get_random_smiles(target_smile, num_random_samples=num_random_samples)
        target_smile_rand_ord   = get_random_smiles(starting_smile,   num_random_samples=num_random_samples)
        
        for smi_start in starting_smile_rand_ord: 
            for smi_target in target_smile_rand_ord: 
                
                if Chem.MolFromSmiles(smi_start) == None or Chem.MolFromSmiles(smi_target) == None: 
                    raise Exception('Invalid structures')
        
                for _ in range(num_tries): 
                    path, _, _, _ = obtain_path(smi_start, smi_target, filter_path=True)
                    smiles_paths_dir2.append(path)
                    
    return smiles_paths_dir1, smiles_paths_dir2




