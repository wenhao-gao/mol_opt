#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:34:33 2020

@author: akshat
"""
import time
import rdkit
import pickle
import itertools
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import rdMolDescriptors
from selfies import decoder 
import numpy as np
from selfies import encoder
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import warnings
warnings.simplefilter('ignore', np.RankWarning)



def get_ECFP4(mol):
    ''' Return rdkit ECFP4 fingerprint object for mol

    Parameters: 
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  

    Returns: 
    rdkit ECFP4 fingerprint object for mol
    '''
    return AllChem.GetMorganFingerprint(mol, 2)

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
        try: 
            mol    = Chem.MolFromSmiles(item)
        except: 
            print('Invalid smile: ', item)
        fp_mol = get_ECFP4(mol)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores


def get_joint_sim(smi_list, triplet): 
    '''Calculate the joint similarity of each SMILES (within smi_list) to a list of three molecules (triplets)
    (Based on Equation 1 of the paper, see Section D :) )
       
    Parameters:
    smi_list (list) : A list of SMILES stings 
    triplet  (list) : List of three SMILES strings
    
    Returns: 
    sim_score (list of floats) : List of joint similarity scores

    '''
    scores_t1 = get_fp_scores(smi_list, triplet[0])   # similarity to triplet 1
    scores_t2 = get_fp_scores(smi_list, triplet[1])   # similarity to triplet 2
    scores_t3 = get_fp_scores(smi_list, triplet[2])   # similarity to triplet 3
    
    z = np.polyfit([-2/3, 0.0, 1.0], [-1.0 , 0.0, 1.0], 3) # Pollynomial coefficients for Equation 1

    data      = np.array([scores_t1, scores_t2, scores_t3])
    sim_score = np.average(data, axis=0) - (np.max(data, axis=0) - np.min(data, axis=0))
    sim_score = (z[0]*(sim_score**3)) + (z[1]*(sim_score**2)) + (z[2]*(sim_score)) 
    
    return sim_score


def form_joint_path(starting_selfie_chars, struct_2_selfie_chars, struct_3_selfie_chars, triplet):  
    ''' Create a generalized chemical path starting from the molecule 'starting_selfie_chars' (provided as a list of chars)
        to 'struct_2_selfie_chars' & 'struct_2_selfie_chars'
        
    Parameters:
    starting_selfie_chars (list)  : A list of characters in a single SELFIES string
    struct_2_selfie_chars  (list) : A list of characters in a single SELFIES string
    struct_3_selfie_chars (list)  : A list of characters in a single SELFIES string
    triplet  (list)               : A list 3 SMILES strings
        
    Returns: 
    path (list of SMILES string)      : List of SMILES strings in a generalized path  
    joint_sim_scores (list of floats) : List of joint similarity score for each SMILES string in path. 
    '''
    best_median = starting_selfie_chars.copy()
    best_score  = -10**6
    
    indices_diff_1 = [i for i in range(len(starting_selfie_chars)) if starting_selfie_chars[i] != struct_2_selfie_chars[i]]
    indices_diff_2 = [i for i in range(len(starting_selfie_chars)) if starting_selfie_chars[i] != struct_3_selfie_chars[i]]
    
    path, joint_sim_scores = [], []
    
    while len(indices_diff_1) > 0 or len(indices_diff_2) > 0:  
                
        # Mutation between 'starting_selfie_chars' & 'struct_2_selfie_chars'
        try: 
            idx_1 = np.random.choice(indices_diff_1, 1)[0] # Index to be operated on
            indices_diff_1.remove(idx_1)                   # Remove that index
            median_1_sf = best_median.copy()
            median_1_sf[idx_1] = struct_2_selfie_chars[idx_1]
            median_1 = decoder(''.join(x for x in median_1_sf).strip())
            median_1_score = get_joint_sim([median_1], triplet)
        except: 
            median_1_score = [-10**7]
        
        # Mutation between 'starting_selfie_chars' & 'struct_3_selfie_chars'
        try: 
            idx_2 = np.random.choice(indices_diff_2, 1)[0] # Index to be operated on
            indices_diff_2.remove(idx_2)                   # Remove that index
            median_2_sf = best_median.copy()
            median_2_sf[idx_2] = struct_3_selfie_chars[idx_2]
            median_2 = decoder(''.join(x for x in median_2_sf).strip())
            median_2_score = get_joint_sim([median_2], triplet)
        except: 
            median_2_score = [-10**7]
        
        if max([median_1_score[0], median_2_score[0]]) > best_score: 
            if median_1_score > median_2_score: 
                best_median = median_1_sf.copy()
                indices_diff_2.append(idx_2)
                path.append(median_1)
                joint_sim_scores.append(median_1_score)
                # print('{} Score: {}'.format(median_1, median_1_score))
            else: 
                best_median = median_2_sf.copy()
                indices_diff_1.append(idx_1)
                path.append(median_2)
                joint_sim_scores.append(median_2_score)
                # print('{} Score: {}'.format(median_2, median_2_score))
            
            best_score = max([median_1_score[0], median_2_score[0]])
    
    return path, joint_sim_scores


# Load in the HCE triplets: 
with open("./data/triplets.pickle", "rb") as fp:   # Unpickling
    triplets_all = pickle.load(fp)

collect_unfilt = {}
collect_filt   = {}
num_paths      = 10000 # Explore 10k paths: 

for trip_id, triplet in enumerate(triplets_all): 
        
    print('On triplet: ', trip_id)
    ALL_PATHS = []
    ALL_SIM   = []
    start_time = time.time()
        
    for iter_ in range(num_paths): 
            
        if iter_ % 10 == 0: 
            print('    Obtaining path {}/{}: '.format(iter_, num_paths))
        # Randomize the ordering of the smiles inside the triplet: 
        triplet = (get_random_smiles(triplet[0], num_random_samples=1)[0], get_random_smiles(triplet[1], num_random_samples=1)[0], get_random_smiles(triplet[2], num_random_samples=1)[0])    
        
        random_choice = np.random.choice([i for i in range(len(triplet))], 3, replace=False)
        
        starting_structure = encoder(triplet[random_choice[0]])
        struct_2 = encoder(triplet[random_choice[1]])  
        struct_3 = encoder(triplet[random_choice[2]]) 
        
        starting_selfie_chars = get_selfie_chars(starting_structure)
        struct_2_selfie_chars = get_selfie_chars(struct_2)
        struct_3_selfie_chars = get_selfie_chars(struct_3)
        
        max_len = max([len(starting_selfie_chars), len(struct_2_selfie_chars), len(struct_3_selfie_chars)])
        
        
        # Make everything the same length: 
        if len(starting_selfie_chars) < max_len: 
            for _ in range(max_len-len(starting_selfie_chars)): starting_selfie_chars.append(' ')
        if len(struct_2_selfie_chars) < max_len: 
            for _ in range(max_len-len(struct_2_selfie_chars)): struct_2_selfie_chars.append(' ')
        if len(struct_3_selfie_chars) < max_len: 
            for _ in range(max_len-len(struct_3_selfie_chars)): struct_3_selfie_chars.append(' ')
                
        
        
        path, joint_sim_scores = form_joint_path(starting_selfie_chars.copy(), struct_2_selfie_chars.copy(), struct_3_selfie_chars.copy(), triplet)
        ALL_PATHS.append(path)
        ALL_SIM.append(joint_sim_scores)

    
    
    ALL_SIM = list(itertools.chain.from_iterable(ALL_SIM))
    ALL_SIM = [x[0] for x in ALL_SIM]
    top_idx = np.argsort(ALL_SIM)[-100: ]
    A = [ALL_SIM[i] for i in top_idx]
    
    # Print statistics for the UNfiltered medians: 
    print('Max: {} Min: {} Mean: {} Std: {}'.format(max(A), min(A), np.mean(A), np.std(A)))
            
    # pick the best filtered
    ALL_PATHS =  list(itertools.chain.from_iterable(ALL_PATHS))
    collect_unfilt[trip_id] = [ALL_PATHS, ALL_SIM]
            
    better_smi = []
    for k,smi in enumerate(ALL_PATHS): 
        mol = Chem.MolFromSmiles(smi)
        if rdMolDescriptors.CalcNumBridgeheadAtoms(mol)==0 and rdMolDescriptors.CalcNumSpiroAtoms(mol)==0:
            # better_smi.append(get_best_taut(mol))
            mol, smi_canon, _ = sanitize_smiles(smi)
            better_smi.append((smi_canon, k))
                           
            
            
    filtered_smiles = [x[0] for x in better_smi]
    filtered_scores = [ALL_SIM[x[1]] for x in better_smi]
    collect_filt[trip_id] = [filtered_smiles, filtered_scores]
            
    top_idx_filt = np.argsort(filtered_scores)[-100: ]
    A = [filtered_scores[i] for i in top_idx_filt]
    top_filt_smi = [filtered_smiles[i] for i in top_idx_filt]
    
    # Print statistics for the filtered medians: 
    print('Time: ', time.time()-start_time)
    print('Max: {} Min: {} Mean: {} Std: {}'.format(max(A), min(A), np.mean(A), np.std(A)))
    

# Saving the results: 
with open("./BEST_MEDN/medn_4.pickle", "wb") as fp:       #Pickling
    pickle.dump(collect_unfilt, fp)
    
with open("./BEST_MEDN/medn_filt_4.pickle", "wb") as fp:   #Pickling
    pickle.dump(collect_filt, fp)