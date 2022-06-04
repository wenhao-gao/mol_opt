#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:03:20 2020

@author: akshat
"""
import seaborn as sns
import selfies
import random
import numpy as np
from selfies import encoder, decoder

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

from realize_path import get_compr_paths

def get_ECFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2)

def get_fp_scores(smiles_back, target_smi): 
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    fp_target = get_ECFP4(target)
    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = get_ECFP4(mol)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores


def get_logP(mol):
    '''Calculate logP of a molecule 
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object, for which logP is to calculates
    
    Returns:
    float : logP of molecule (mol)
    '''
    return Descriptors.MolLogP(mol)

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




starting_smile = 'CN1CC(=O)N2C(C1=O)CC3=C(C2C4=CC5=C(C=C4)OCO5)NC6=CC=CC=C36' # Tadalafil
target_smile   = 'CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C' # Sildenafil 

mol_starting = Chem.MolFromSmiles(starting_smile)
mol_target   = Chem.MolFromSmiles(target_smile)

qed_starting = Chem.QED.qed(mol_starting)
qed_target   = Chem.QED.qed(mol_target)

logP_starting = get_logP(mol_starting)
logP_target   = get_logP(mol_target)


scores_start_1  = get_fp_scores([starting_smile], starting_smile)   # similarity to target
scores_target_1 = get_fp_scores([starting_smile], target_smile)     # similarity to starting structure
data          = np.array([scores_target_1, scores_start_1])
avg_score_1     = np.average(data, axis=0)
better_score_1  = avg_score_1 - (np.abs(data[0] - data[1])) 
better_score_1 = ((1/9) * better_score_1**3) - ((7/9) * better_score_1**2) + ((19/12) * better_score_1)


scores_start_2  = get_fp_scores([target_smile], starting_smile)   # similarity to target
scores_target_2 = get_fp_scores([target_smile], target_smile)     # similarity to starting structure
data          = np.array([scores_target_2, scores_start_2])
avg_score_2     = np.average(data, axis=0)
better_score_2  = avg_score_2 - (np.abs(data[0] - data[1])) 
better_score_2 = ((1/9) * better_score_2**3) - ((7/9) * better_score_2**2) + ((19/12) * better_score_2)


print('Starting logP:{} QED:{}'.format(logP_starting, qed_starting))
print('Target logP:{} QED:{}'.format(logP_target, qed_target))

   
num_tries             = 2 
num_random_samples    = 2 
collect_bidirectional = True # Doubles the number of paths: source->target & target->source

print('Initiating path collection')
smiles_paths_dir1, smiles_paths_dir2 = get_compr_paths(starting_smile, target_smile, num_tries, num_random_samples, collect_bidirectional)
print('Path collection complete')


# Find the median molecule & plot: 
all_smiles_dir_1 = [item for sublist in smiles_paths_dir1 for item in sublist] # all the smile string of dir1
all_smiles_dir_2 = [item for sublist in smiles_paths_dir2 for item in sublist] # all the smile string of dir2

all_smiles = all_smiles_dir_1 + all_smiles_dir_2
logP_path  = [get_logP(Chem.MolFromSmiles(x)) for x in all_smiles]
QED_path   = [Chem.QED.qed(Chem.MolFromSmiles(x)) for x in all_smiles]
    
scores_start  = get_fp_scores(all_smiles, starting_smile)   # similarity to target
scores_target = get_fp_scores(all_smiles, target_smile)     # similarity to starting structure
data          = np.array([scores_target, scores_start])
avg_score     = np.average(data, axis=0)
better_score  = avg_score - (np.abs(data[0] - data[1]))   
better_score = ((1/9) * better_score**3) - ((7/9) * better_score**2) + ((19/12) * better_score)




# Filter based on better score: 
apply_score_threshold = False 
if apply_score_threshold: 
    indices_threshold = []
    for i in range(len(better_score)): 
        if better_score[i] >= -20: # 0.2 = if required, Threshold! 
            indices_threshold.append(i)
            
    all_smiles = [all_smiles[i] for i in indices_threshold]
    logP_path  = [get_logP(Chem.MolFromSmiles(x)) for x in all_smiles]
    QED_path   = [Chem.QED.qed(Chem.MolFromSmiles(x)) for x in all_smiles]

    scores_start  = get_fp_scores(all_smiles, starting_smile)   # similarity to target
    scores_target = get_fp_scores(all_smiles, target_smile)     # similarity to starting structure
    data          = np.array([scores_target, scores_start])
    avg_score     = np.average(data, axis=0)
    better_score  = avg_score - (np.abs(data[0] - data[1]))   
    better_score = ((1/9) * better_score**3) - ((7/9) * better_score**2) + ((19/12) * better_score)



print('Min {} Max {}'.format(min(better_score), max(better_score)))
# raise Exception('get vmax value')

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
cm = plt.cm.get_cmap('viridis')
sc = ax.scatter(logP_path, QED_path, c=better_score.tolist(), cmap=cm, s=13) 
clb = plt.colorbar(sc)

sc = ax.plot([logP_starting, logP_target], [qed_starting, qed_target], 'o', c='black', markersize=7, linewidth=3) # TARGETS 

clb.set_label('Joint Similarity', fontsize=10)
ax.set_xlabel('LogP', fontsize=10)
ax.set_ylabel('QED', fontsize=10)
plt.xlim([-4, 8])

ax.grid(True)
fig.tight_layout()
# plt.savefig('./logP_v_QED_scatter.png', dpi=1000)

plt.show()

alphabet = list(selfies.get_semantic_robust_alphabet()) # 34 SELFIE characters 
max_len_random_struct = max([len(get_selfie_chars(encoder(starting_smile))), len(get_selfie_chars(encoder(target_smile)))])
min_len_random_struct = min([len(get_selfie_chars(encoder(starting_smile))), len(get_selfie_chars(encoder(target_smile)))])
num_samples = len(logP_path)
random_selfies = []

for _ in range(num_samples): 
    selfie = ''

    for i in range(random.randint(min_len_random_struct, max_len_random_struct)): # max_molecules_len = max random selfie string length 
        selfie = selfie + np.random.choice(alphabet, size=1)[0]
    random_selfies.append(selfie)
    
random_smiles = [decoder(x) for x in random_selfies]
scores_start_rnd  = get_fp_scores(random_smiles, starting_smile)   # similarity to target
scores_target_rnd = get_fp_scores(random_smiles, target_smile)     # similarity to starting structure
data_rnd          = np.array([scores_target_rnd, scores_start_rnd])
avg_score_rnd     = np.average(data_rnd, axis=0)
better_score_random  = avg_score_rnd - (np.abs(data_rnd[0] - data_rnd[1]))   

better_score_random = ((1/9) * better_score_random**3) - ((7/9) * better_score_random**2) + ((19/12) * better_score_random)

logP_path_random  = [get_logP(Chem.MolFromSmiles(x)) for x in random_smiles]
QED_path_random   = [Chem.QED.qed(Chem.MolFromSmiles(x)) for x in random_smiles]




# DISTRUUTION PLOTS! 
A = sns.kdeplot(logP_path_random, bw_method=0.2, label="Random SELFIES")
A = sns.kdeplot(logP_path, bw_method=0.2, label="Chemical path", color='yellowgreen')

plt.axvline(logP_starting, 0, 1.0, c='black') # vertical line
plt.axvline(logP_target, 0, 1.0, c='black') # vertical line


A.set_xlabel('LogP', fontsize=10)
A.set_ylabel('Density', fontsize=10)
plt.xlim([-4, 8])
plt.legend()
# plt.savefig('./final_saved/logP_distrb.svg', dpi=500)
plt.show()


B = sns.kdeplot(QED_path_random, bw=.2, label="Random SELFIES")
B = sns.kdeplot(QED_path, bw=.2, label="Chemical path", color='yellowgreen')
plt.axvline(qed_starting, 0, 1.0, c='black') # vertical line
plt.axvline(qed_target, 0, 1.0, c='black') # vertical line
B.set_xlabel('QED', fontsize=10)
B.set_ylabel('Density', fontsize=10)
plt.xlim([0, 1])
plt.legend()
# plt.savefig('./final_saved/QED_distrb.svg', dpi=500)
plt.show()





