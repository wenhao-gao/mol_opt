import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from functools import reduce 
from tqdm import tqdm 
from copy import deepcopy 
import numpy as np 
import torch 
from torch.autograd import Variable
torch.manual_seed(4) 
np.random.seed(1)
import random 
random.seed(1)

'''
	1. vocabulary: find frequent words (atom and ring) 
	2. graph2tree 
    3. generate smiles set 
    4. chemical utility 
        tanimot similarity 
        canonicalize smiles  
        is valid
    5. score modifier  
        logp_modifier [-inf, inf] -> [0,1] 

        qed_logp_jnk_gsk_fusion
            qed, logp, jsn, gsk  -> [0,1]
    
    
'''
def sigmoid(float_x):
    return 1.0 / (1 + np.exp(-float_x))

from scipy.stats import gmean

def logp_modifier(logp_score):
    return max(0.0,min(1.0,1/14*(logp_score+10))) 
'''
[-inf, inf] -> [0,1]
'''

def docking_modifier(docking_score):
    '''
        [-12,-4]  -> [0,1]
        -12  ----->  1
        -4   ----->  0 
    '''
    docking_score = 1/(12-4)*(-docking_score - 4)
    docking_score = max(docking_score, 0.0)
    docking_score = min(docking_score, 1.0) 
    return docking_score 

def qed_logp_fusion(qed_score, logp_score, jnk_score, gsk_score):
    logp_score = logp_modifier(logp_score)
    gmean_score = gmean([qed_score, logp_score])
    modified_score = min(1.0,gmean_score)
    return modified_score

def logp_jnk_gsk_fusion(logp_score, jnk_score, gsk_score):
    logp_score = logp_modifier(logp_score)
    return np.mean([logp_score, jnk_score, gsk_score])


def qed_logp_jnk_gsk_fusion(qed_score, logp_score, jnk_score, gsk_score):
    logp_score = logp_modifier(logp_score)
    gmean_score = gmean([qed_score, logp_score, jnk_score, gsk_score])
    modified_score = min(1.0,gmean_score)
    return modified_score

def qed_logp_jnk_gsk_fusion2(qed_score, logp_score, jnk_score, gsk_score):
    logp_score = logp_modifier(logp_score)
    return  np.mean([qed_score, logp_score, jnk_score, gsk_score])

def qed_logp_fusion(qed_score, logp_score):
    logp_score = logp_modifier(logp_score)
    gmean_score = gmean([qed_score, logp_score])
    modified_score = min(1.0, gmean_score)
    return modified_score 

def jnk_gsk_fusion(jnk_score, gsk_score):
    gmean_score = gmean([jnk_score, gsk_score])
    modified_score = min(1.0,gmean_score)
    return modified_score


def load_vocabulary():
	datafile = "data/vocabulary.txt"
	with open(datafile, 'r') as fin:
		lines = fin.readlines()
	vocabulary = [line.split()[0] for line in lines]
	return vocabulary 

vocabulary = load_vocabulary()
bondtype_list = [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE]


def ith_substructure_is_atom(i):
    substructure = vocabulary[i]
    return True if len(substructure)==1 else False

def word2idx(word):
    return vocabulary.index(word)


# def smiles2fingerprint(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
#     return np.array(fp)
#     ### shape: (2048,)

def smiles2fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=False)
    return np.array(fp)
    ### shape: (1024,)


## similarity of two SMILES 
def similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 


def similarity_matrix(smiles_lst):
    n = len(smiles_lst)
    sim_matrix = np.eye(n)
    mol_lst = [Chem.MolFromSmiles(smiles) for smiles in smiles_lst]
    fingerprint_lst = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False) for mol in mol_lst]
    for i in range(n):
        fp1 = fingerprint_lst[i]
        for j in range(i+1,n):
            fp2 = fingerprint_lst[j]
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            sim_matrix[i,j] = sim_matrix[j,i] = sim
    return sim_matrix 


def canonical(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None 
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True) ### todo double check
    else:
        return None


def smiles2mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None 
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol 

## input: smiles, output: word lst;  
def smiles2word(smiles):
    mol = smiles2mol(smiles)
    if mol is None:
        return None 
    word_lst = []

    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques_smiles = []
    for clique in cliques:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        cliques_smiles.append(clique_smiles)
    atom_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    return cliques_smiles + atom_not_in_rings_list 

## is_valid_smiles 
def is_valid(smiles):
    word_lst = smiles2word(smiles)
    word_set = set(word_lst)
    return word_set.issubset(vocabulary)     


def is_valid_mol(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
    except:
        return False 
    if smiles.strip() == '':
        return False 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return False 
    return True 

def substr_num(smiles):
    mol = smiles2mol(smiles)
    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    return len(clique_lst)


def smiles2substrs(smiles):
    if not is_valid(smiles):
        return None 
    mol = smiles2mol(smiles)
    if mol is None:
        return None
    idx_lst = []

    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    # print(clique_lst)  ## [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15]]
    for clique in clique_lst:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        # print("clique_smiles", clique_smiles)  ## C1=CC=CC=C1, C1=COCC1, C1=CC=CC=C1 
        idx_lst.append(word2idx(clique_smiles))
    atom_symbol_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    atom_idx_not_in_rings_list = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]
    # print(atom_idx_not_in_rings_list)  ## [0, 1, 2, 3, 11, 12, 13, 14, 21]  nonring atom's index in molecule
    for atom in atom_symbol_not_in_rings_list:
        idx_lst.append(word2idx(atom))

    return idx_lst 



def smiles2graph(smiles):
    '''     N is # of substructures in the molecule 

    Output:
        1.
            idx_lst                 [N]      list of substructure's index
            node_mat                [N,d]
        2. 
            substructure_lst 
            atomidx_2substridx     dict 
        3. 
            adjacency_matrix        [N,N]    0/1   np.zeros((4,4))  
        4. 
            leaf_extend_idx_pair    [(x1,y1), (x2,y2), ...]
    '''

    ### 0. smiles -> mol 
    if not is_valid(smiles):
        return None 
    mol = smiles2mol(smiles)
    if mol is None:
        return None

    ### 1. idx_lst & node_mat 
    idx_lst = []
    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    # print(clique_lst)  ## [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15]]
    for clique in clique_lst:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        # print("clique_smiles", clique_smiles)  ## C1=CC=CC=C1, C1=COCC1, C1=CC=CC=C1 
        idx_lst.append(word2idx(clique_smiles))

    atom_symbol_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    atom_idx_not_in_rings_list = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]
    # print(atom_idx_not_in_rings_list)  ## [0, 1, 2, 3, 11, 12, 13, 14, 21]  nonring atom's index in molecule
    for atom in atom_symbol_not_in_rings_list:
        idx_lst.append(word2idx(atom))
    # print(idx_lst) ## [3, 68, 3, 0, 0, 0, 0, 0, 0, 1, 2, 4]  
    d = len(vocabulary)
    N = len(idx_lst)
    node_mat = np.zeros((N, d))
    for i,v in enumerate(idx_lst):
        node_mat[i,v]=1


    ### 2. substructure_lst & atomidx_2substridx     
    ###    map from atom index to substructure index 
    atomidx_2substridx = dict()
    substructure_lst = clique_lst + atom_idx_not_in_rings_list   
    ### [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15], 0, 1, 2, 3, 11, 12, 13, 14, 21] 
    ### 4:0  23:0, 22:0, ...   8:1, 7:1, 22:1, ... 16:2, 17:2, 18:2, ... 0:3, 1:4, 
    for idx, substructure in enumerate(substructure_lst):
    	if type(substructure)==list:
    		for atom in substructure:
    			atomidx_2substridx[atom] = idx 
    	else:
    		atomidx_2substridx[substructure] = idx 


    ### 3. adjacency_matrix 
    adjacency_matrix = np.zeros((N,N),dtype=np.int32)

    ####### 3.1 atom-atom bonds and atom-ring bonds
    for bond in mol.GetBonds():
    	if not bond.IsInRing():
    		a1 = bond.GetBeginAtom().GetIdx()
    		a2 = bond.GetEndAtom().GetIdx()
    		idx1 = atomidx_2substridx[a1] 
    		idx2 = atomidx_2substridx[a2]
    		adjacency_matrix[idx1,idx2] = adjacency_matrix[idx2,idx1] = 1 
    ####### 3.2 ring-ring connection 
    for i1,c1 in enumerate(clique_lst):
    	for i2,c2 in enumerate(clique_lst):
    		if i1>=i2:
    			continue 
    		if len(set(c1).intersection(set(c2))) > 0:
    			adjacency_matrix[i1,i2] = adjacency_matrix[i2,i1] = 1
    assert np.sum(adjacency_matrix)>=2*(N-1)

    leaf_idx_lst = list(np.where(np.sum(adjacency_matrix,1)==1)[0])
    M = len(leaf_idx_lst)
    extend_idx_lst = list(range(N,N+M))
    leaf_extend_idx_pair = list(zip(leaf_idx_lst, extend_idx_lst))
    ####### [(3, 12), (5, 13), (6, 14), (9, 15), (11, 16)]

    return idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair 


def smiles2feature(smiles):
    """
        (1) molecule2tree
        (2) mask leaf node 
    """
    ### 0. smiles -> mol 
    if not is_valid(smiles):
        return None 
    mol = smiles2mol(smiles)
    if mol is None:
        return None

    ### 1. idx_lst  
    idx_lst = []
    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    # print(clique_lst)  ## [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15]]
    for clique in clique_lst:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        # print("clique_smiles", clique_smiles)  ## C1=CC=CC=C1, C1=COCC1, C1=CC=CC=C1 
        idx_lst.append(word2idx(clique_smiles))

    atom_symbol_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    atom_idx_not_in_rings_list = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]
    # print(atom_idx_not_in_rings_list)  ## [0, 1, 2, 3, 11, 12, 13, 14, 21]  nonring atom's index in molecule
    for atom in atom_symbol_not_in_rings_list:
        idx_lst.append(word2idx(atom))
    # print(idx_lst) ## [3, 68, 3, 0, 0, 0, 0, 0, 0, 1, 2, 4]  
    d = len(vocabulary)
    N = len(idx_lst)

    ### 2. substructure_lst & atomidx_2substridx     
    ###    map from atom index to substructure index 
    atomidx_2substridx = dict()
    substructure_lst = clique_lst + atom_idx_not_in_rings_list   
    ### [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15], 0, 1, 2, 3, 11, 12, 13, 14, 21] 
    ### 4:0  23:0, 22:0, ...   8:1, 7:1, 22:1, ... 16:2, 17:2, 18:2, ... 0:3, 1:4, 
    for idx, substructure in enumerate(substructure_lst):
        if type(substructure)==list:
            for atom in substructure:
                atomidx_2substridx[atom] = idx 
        else:
            atomidx_2substridx[substructure] = idx 

    ### 3. adjacency_matrix 
    adjacency_matrix = np.zeros((N,N),dtype=np.int32)
    ####### 3.1 atom-atom bonds and atom-ring bonds
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            idx1 = atomidx_2substridx[a1] 
            idx2 = atomidx_2substridx[a2]
            adjacency_matrix[idx1,idx2] = adjacency_matrix[idx2,idx1] = 1 
    ####### 3.2 ring-ring connection 
    for i1,c1 in enumerate(clique_lst):
        for i2,c2 in enumerate(clique_lst):
            if i1>=i2:
                continue 
            if len(set(c1).intersection(set(c2))) > 0:
                adjacency_matrix[i1,i2] = adjacency_matrix[i2,i1] = 1
    assert np.sum(adjacency_matrix)>=2*(N-1)

    # print(adjacency_matrix, smiles)
    leaf_idx_lst = list(np.where(np.sum(adjacency_matrix,1)==1)[0])
    mask_idx = random.choice(leaf_idx_lst)
    label = idx_lst[mask_idx]

    node_mat = np.zeros((N, d + 1))
    for i,v in enumerate(idx_lst):
        if i==mask_idx:
            node_mat[i,d] = 1 
        else:
            node_mat[i,v] = 1

    return node_mat, adjacency_matrix, mask_idx, label 



def smiles2expandfeature(smiles):
    """
        (1) molecule2tree
        (2) mask leaf node 
    """
    ### 0. smiles -> mol 
    if not is_valid(smiles):
        return None 
    mol = smiles2mol(smiles)
    if mol is None:
        return None

    ### 1. idx_lst
    idx_lst = []
    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    # print(clique_lst)  ## [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15]]
    for clique in clique_lst:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        # print("clique_smiles", clique_smiles)  ## C1=CC=CC=C1, C1=COCC1, C1=CC=CC=C1 
        idx_lst.append(word2idx(clique_smiles))

    atom_symbol_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    atom_idx_not_in_rings_list = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]
    # print(atom_idx_not_in_rings_list)  ## [0, 1, 2, 3, 11, 12, 13, 14, 21]  nonring atom's index in molecule
    for atom in atom_symbol_not_in_rings_list:
        idx_lst.append(word2idx(atom))
    # print(idx_lst) ## [3, 68, 3, 0, 0, 0, 0, 0, 0, 1, 2, 4]  
    d = len(vocabulary)
    N = len(idx_lst)

    ### 2. substructure_lst & atomidx_2substridx     
    ###    map from atom index to substructure index 
    atomidx_2substridx = dict()
    substructure_lst = clique_lst + atom_idx_not_in_rings_list   
    ### [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15], 0, 1, 2, 3, 11, 12, 13, 14, 21] 
    ### 4:0  23:0, 22:0, ...   8:1, 7:1, 22:1, ... 16:2, 17:2, 18:2, ... 0:3, 1:4, 
    for idx, substructure in enumerate(substructure_lst):
        if type(substructure)==list:
            for atom in substructure:
                atomidx_2substridx[atom] = idx 
        else:
            atomidx_2substridx[substructure] = idx 

    ### 3. adjacency_matrix 
    adjacency_matrix = np.zeros((N+1,N+1),dtype=np.int32)
    ####### 3.1 atom-atom bonds and atom-ring bonds
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            idx1 = atomidx_2substridx[a1] 
            idx2 = atomidx_2substridx[a2]
            adjacency_matrix[idx1,idx2] = adjacency_matrix[idx2,idx1] = 1 
    ####### 3.2 ring-ring connection 
    for i1,c1 in enumerate(clique_lst):
        for i2,c2 in enumerate(clique_lst):
            if i1>=i2:
                continue 
            if len(set(c1).intersection(set(c2))) > 0:
                adjacency_matrix[i1,i2] = adjacency_matrix[i2,i1] = 1
    # assert np.sum(adjacency_matrix)>=2*(N-1)

    # print(adjacency_matrix, smiles)
    leaf_idx_lst = list(np.where(np.sum(adjacency_matrix,1)==1)[0])
    mask_idx = random.choice(leaf_idx_lst)
    label = idx_lst[mask_idx]


    node_mat = np.zeros((N + 1, d + 1))
    for i,v in enumerate(idx_lst):
        node_mat[i,v] = 1

    feature_lst = []
    for idx in range(N):
        new_node_mat = deepcopy(node_mat)
        new_adj_mat = deepcopy(adjacency_matrix)
        new_node_mat[-1,d] = 1 
        new_adj_mat[idx,N] = 1 
        new_adj_mat[N,idx] = 1 
        feature_lst.append((new_node_mat, new_adj_mat, N))


    return feature_lst 







def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def add_atom_at_position(editmol, position_idx, new_atom, new_bond):
    '''
        position_idx:   index of edited atom in editmol
        new_atom: 'C', 'N', 'O', ... 
        new_bond: SINGLE, DOUBLE  
    '''
    ######  1 edit mol 
    new_atom = Chem.rdchem.Atom(new_atom)
    rwmol = deepcopy(editmol)
    new_atom_idx = rwmol.AddAtom(new_atom)
    rwmol.AddBond(position_idx, new_atom_idx, order = new_bond)
    ######  2 check valid of new mol 
    if not is_valid_mol(rwmol):
        return None  
    try:
        rwmol.UpdatePropertyCache()
    except:
        return None
    smiles = Chem.MolToSmiles(rwmol)
    assert '.' not in smiles
    return canonical(smiles)


def add_fragment_at_position(editmol, position_idx, fragment, new_bond):
    '''
        position_idx:  index of edited atom in editmol
        fragment: e.g., "C1=CC=CC=C1", "C1=CC=NC=C1", ... 
        new_bond: {SINGLE, DOUBLE}  

        Return:  
            list of SMILES
    '''  
    new_smiles_set = set()
    fragment_mol = Chem.MolFromSmiles(fragment)
    current_atom = editmol.GetAtomWithIdx(position_idx)
    neighbor_atom_set = set()  ## index of neighbor of current atom in new_mol  


    ## (A) add a bond between atom and ring 
    #### 1. initialize empty new_mol
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    #### 2. add editmol into new_mol
    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx()
        new_atom = copy_atom(atom)
        new_idx = new_mol.AddAtom(new_atom)
        old_idx2new_idx[old_idx] = new_idx 
        assert old_idx == new_idx
    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        i1 = a1.GetIdx()
        i2 = a2.GetIdx()
        i1_new = old_idx2new_idx[i1]
        i2_new = old_idx2new_idx[i2]
        bt = bond.GetBondType()
        new_mol.AddBond(i1_new, i2_new, bt)
        ### collect the neighbor atoms of current atom, both are in ring. 
        if (i1==position_idx or i2==position_idx) and (a1.IsInRing() and a2.IsInRing()):
            neighbor_atom_set.add(i1_new)
            neighbor_atom_set.add(i2_new)
    if neighbor_atom_set != set():
        neighbor_atom_set.remove(old_idx2new_idx[position_idx])

    #### 3. combine two components 
    #### 3.1 add fragment into new_mol
    new_atom_idx_lst = []
    old_idx2new_idx2 = dict()  ### fragment idx -> new mol idx 
    for atom in fragment_mol.GetAtoms():
        old_atom_idx = atom.GetIdx()
        new_atom = copy_atom(atom)
        new_atom_idx = new_mol.AddAtom(new_atom)
        new_atom_idx_lst.append(new_atom_idx)
        old_idx2new_idx2[old_atom_idx] = new_atom_idx 
    for bond in fragment_mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        i1 = old_idx2new_idx2[a1]
        i2 = old_idx2new_idx2[a2]
        bt = bond.GetBondType()
        new_mol.AddBond(i1, i2, bt)

    #### 3.2 enumerate possible binding atoms and generate new smiles 
    for i in new_atom_idx_lst:  ### enumeration 
        copy_mol = deepcopy(new_mol)
        copy_mol.AddBond(old_idx2new_idx[position_idx], i, new_bond)
        if is_valid_mol(copy_mol):
            try:
                copy_mol.UpdatePropertyCache()
                new_smiles = Chem.MolToSmiles(copy_mol)
                new_smiles = canonical(new_smiles)
                if new_smiles is not None:
                    assert '.' not in new_smiles
                    new_smiles_set.add(new_smiles) 
            except:
                pass  


    # if not current_atom.IsInRing() or new_bond != rdkit.Chem.rdchem.BondType.SINGLE:
    if not current_atom.IsInRing():
        return new_smiles_set


    # print(new_smiles_set)
    ## (B) share bond between rings 
    #### 1. initialize empty new_mol
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    #### 2. add editmol into new_mol
    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx() 
        new_atom = copy_atom(atom)
        new_idx = new_mol.AddAtom(new_atom)
        old_idx2new_idx[old_idx] = new_idx 
        assert old_idx == new_idx 
    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        i1 = old_idx2new_idx[a1]
        i2 = old_idx2new_idx[a2]
        bt = bond.GetBondType()
        new_mol.AddBond(i1, i2, bt) 

    # print(Chem.MolToSmiles(new_mol))
    #### 3. fragment mol  
    ####### 3.1 find 2 common atoms and 1 bond  
    current_atom = editmol.GetAtomWithIdx(old_idx2new_idx[position_idx])
    current_atom_symbol = current_atom.GetSymbol()

    atom_lst = list(fragment_mol.GetAtoms())
    for neighbor_atom in neighbor_atom_set:
        neighbor_atom_symbol = editmol.GetAtomWithIdx(neighbor_atom).GetSymbol()
        bondtype_edit = new_mol.GetBondBetweenAtoms(neighbor_atom, old_idx2new_idx[position_idx]).GetBondType()
        for i,v in enumerate(atom_lst):
            v_idx = v.GetIdx()
            ### v1 is neighbor of v 
            for v1 in [atom_lst[i-1], atom_lst[i+1-len(atom_lst)]]: 
                v1_idx = v1.GetIdx()
                bondtype_frag = fragment_mol.GetBondBetweenAtoms(v_idx, v1_idx).GetBondType()
                # print("current:", current_atom_symbol, "neighbor:", neighbor_atom_symbol, bondtype_edit)
                # print(v.GetSymbol(), v1.GetSymbol(), bondtype_frag)
                if v.GetSymbol()==current_atom_symbol and v1.GetSymbol()==neighbor_atom_symbol and bondtype_edit==bondtype_frag: 
                    ####### 3.1 find 2 common atoms and 1 bond  
                    # print("2 common atoms and 1 bond ")
                    ############################################
                    ####### 3.2 add other atoms and bonds 
                    new_mol2 = deepcopy(new_mol)
                    old_idx2new_idx2 = dict()
                    old_idx2new_idx2[v_idx] = current_atom.GetIdx()
                    old_idx2new_idx2[v1_idx] = neighbor_atom
                    for atom in fragment_mol.GetAtoms():
                        old_idx = atom.GetIdx()
                        if not (old_idx==v_idx or old_idx==v1_idx):
                            new_atom = copy_atom(atom)
                            new_idx = new_mol2.AddAtom(new_atom)
                            old_idx2new_idx2[old_idx] = new_idx 
                    for bond in fragment_mol.GetBonds():
                        a1 = bond.GetBeginAtom()
                        a2 = bond.GetEndAtom()
                        i1 = a1.GetIdx()
                        i2 = a2.GetIdx()
                        i1_new = old_idx2new_idx2[i1]
                        i2_new = old_idx2new_idx2[i2]
                        bt = bond.GetBondType()
                        if not (set([i1,i2]) == set([v1.GetIdx(), v.GetIdx()])):
                            new_mol2.AddBond(i1_new, i2_new, bt)
                    ####### 3.2 add other atoms and bonds 
                    ####### 3.3 check validity and canonicalize
                    if not is_valid_mol(new_mol2):
                        continue 
                    try:
                        new_mol2.UpdatePropertyCache()
                        # print("success")
                    except:
                        continue 
                    new_smiles = Chem.MolToSmiles(new_mol2)
                    new_smiles = canonical(new_smiles)
                    if new_smiles is not None:
                        assert '.' not in new_smiles
                        new_smiles_set.add(new_smiles)
                    # print(new_smiles)
    # print(new_smiles_set)
    return new_smiles_set



def delete_substructure_at_idx(editmol, atom_idx_lst):
    edit_smiles = Chem.MolToSmiles(editmol)
    #### 1. initialize with empty mol
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    #### 2. add editmol into new_mol
    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx()
        if old_idx in atom_idx_lst: 
            continue 
        new_atom = copy_atom(atom)
        new_idx = new_mol.AddAtom(new_atom)
        old_idx2new_idx[old_idx] = new_idx 
    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if a1 in atom_idx_lst or a2 in atom_idx_lst:
            continue 
        a1_new = old_idx2new_idx[a1]
        a2_new = old_idx2new_idx[a2]
        bt = bond.GetBondType()
        new_mol.AddBond(a1_new, a2_new, bt) 

    if not is_valid_mol(new_mol):
        return None
    try:
        new_mol.UpdatePropertyCache()
    except:
        return None 
    return new_mol, old_idx2new_idx 






def differentiable_graph2smiles_lgp(origin_smiles, differentiable_graph, 
                                leaf_extend_idx_pair, leaf_nonleaf_lst, 
                                max_num_offspring = 100, topk = 3):
    '''
        origin_smiles:
            origin_idx_lst              [N]      0,1,...,d-1 
            origin_node_mat             [N,d]
            origin_substructure_lst     
            origin_atomidx_2substridx   
            origin_adjacency_matrix     [N,N]    0/1
        differentiable_graph:   returned results 
            node_indicator              [N+M,d]
            adjacency_weight            [N+M,N+M]
        N is # of substructures in the molecule
        M is # of leaf node, also number of extended node. 
    main utility
        add_atom_at_position 
        add_fragment_at_position 
        delete_substructure_at_idx 
        REPLACE = delete + add 
    Output:
        new_smiles_set
    '''
    new_smiles_set = set()
    #### 1. data preparation 
    origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(origin_smiles))
    origin_idx_lst, origin_node_mat, origin_substructure_lst, \
    origin_atomidx_2substridx, origin_adjacency_matrix, leaf_extend_idx_pair = smiles2graph(origin_smiles)
    node_indicator, adjacency_weight = differentiable_graph 
    N = len(origin_idx_lst)
    M = len(leaf_extend_idx_pair) 
    d = len(vocabulary)

    ####### 2.3 add   todo: use adjacency_weight to further narrow scope
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### int: single atom;   else: list of integer
            leaf_atom_idx_lst = [leaf_atom_idx_lst]
        for leaf_atom_idx in leaf_atom_idx_lst:
            added_substructure_lst = list(np.argsort(-node_indicator[extend_idx]))[:topk]
            for substructure_idx in added_substructure_lst:
                new_substructure = vocabulary[substructure_idx]
                for new_bond in bondtype_list:
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                                    fragment = new_substructure , new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)

    return new_smiles_set.difference(set([None]))  





def differentiable_graph2smiles_v0(origin_smiles, differentiable_graph, 
                                leaf_extend_idx_pair, leaf_nonleaf_lst, 
                                max_num_offspring = 100, topk = 3):
    '''
        origin_smiles:
            origin_idx_lst              [N]      0,1,...,d-1 
            origin_node_mat             [N,d]
            origin_substructure_lst     
            origin_atomidx_2substridx   
            origin_adjacency_matrix     [N,N]    0/1
        differentiable_graph:   returned results 
            node_indicator              [N+M,d]
            adjacency_weight            [N+M,N+M]
        N is # of substructures in the molecule
        M is # of leaf node, also number of extended node. 
    main utility
        add_atom_at_position 
        add_fragment_at_position 
        delete_substructure_at_idx 
        REPLACE = delete + add 
    Output:
        new_smiles_set
    '''
    new_smiles_set = set()
    #### 1. data preparation 
    origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(origin_smiles))
    origin_idx_lst, origin_node_mat, origin_substructure_lst, \
    origin_atomidx_2substridx, origin_adjacency_matrix, leaf_extend_idx_pair = smiles2graph(origin_smiles)
    node_indicator, adjacency_weight = differentiable_graph 
    N = len(origin_idx_lst)
    M = len(leaf_extend_idx_pair) 
    d = len(vocabulary)

    #### 2. edit the original molecule  
    ####### 2.1 delete & 2.2 replace 
    for leaf_idx, _ in leaf_extend_idx_pair:
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### single atom
            new_leaf_atom_idx_lst = [leaf_atom_idx_lst]
        else:  #### ring     
            ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
            new_leaf_atom_idx_lst = []
            remaining_atoms_idx_lst = []
            for i,v in enumerate(origin_substructure_lst):
                if i==leaf_idx:
                    continue 
                if type(v)==int:
                    remaining_atoms_idx_lst.append(v)
                else: #### list 
                    remaining_atoms_idx_lst.extend(v)
            new_leaf_atom_idx_lst = [leaf_atom_idx for leaf_atom_idx in leaf_atom_idx_lst if leaf_atom_idx not in remaining_atoms_idx_lst]
        ### leaf_atom_idx_lst v.s. new_leaf_atom_idx_lst 
        ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
        result = delete_substructure_at_idx(editmol = origin_mol, atom_idx_lst = new_leaf_atom_idx_lst) 
        if result is None: 
            continue
        delete_mol, old_idx2new_idx = result
        delete_smiles = Chem.MolToSmiles(delete_mol)
        if delete_smiles is None or '.' in delete_smiles:
            continue
        delete_smiles = canonical(delete_smiles)
        new_smiles_set.add(delete_smiles)  #### 2.1 delete done
        ####  2.2 replace  a & b 
        ######### (a) get neighbor substr
        neighbor_substructures_idx = [idx for idx,value in enumerate(origin_adjacency_matrix[leaf_idx]) if value==1]
        assert len(neighbor_substructures_idx)==1 
        neighbor_substructures_idx = neighbor_substructures_idx[0]
        neighbor_atom_idx_lst = origin_substructure_lst[neighbor_substructures_idx]
        if type(neighbor_atom_idx_lst)==int:
            neighbor_atom_idx_lst = [neighbor_atom_idx_lst] 
        ######### (b) add new substructure  todo, enumerate several possibility 
        added_substructure_lst = list(np.argsort(-node_indicator[leaf_idx]))[:topk]  ### topk 
        for substructure_idx in added_substructure_lst: 
            new_substructure = vocabulary[substructure_idx]
            for new_bond in bondtype_list:
                for leaf_atom_idx in neighbor_atom_idx_lst:
                    new_leaf_atom_idx = old_idx2new_idx[leaf_atom_idx] 
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                                    fragment = new_substructure, new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)



    ####### 2.3 add   todo: use adjacency_weight to further narrow scope
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        expand_prob = (adjacency_weight[leaf_idx][extend_idx] + adjacency_weight[extend_idx][leaf_idx])/2  ### [-inf, inf]
        # print("expand prob", expand_prob)
        if expand_prob < -3:
            continue 
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### int: single atom;   else: list of integer
            leaf_atom_idx_lst = [leaf_atom_idx_lst]
        for leaf_atom_idx in leaf_atom_idx_lst:
            added_substructure_lst = list(np.argsort(-node_indicator[extend_idx]))[:topk]
            for substructure_idx in added_substructure_lst:
                new_substructure = vocabulary[substructure_idx]
                for new_bond in bondtype_list:
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                                    fragment = new_substructure , new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)



    return new_smiles_set.difference(set([None]))  




def differentiable_graph2smiles(origin_smiles, differentiable_graph, 
                                leaf_extend_idx_pair, leaf_nonleaf_lst, 
                                max_num_offspring = 100, topk = 3):
    '''
        origin_smiles:
            origin_idx_lst              [N]      0,1,...,d-1 
            origin_node_mat             [N,d]
            origin_substructure_lst     
            origin_atomidx_2substridx   
            origin_adjacency_matrix     [N,N]    0/1

        differentiable_graph:   returned results 
            node_indicator              [N+M,d]
            adjacency_weight            [N+M,N+M]

        N is # of substructures in the molecule
        M is # of leaf node, also number of extended node. 


    main utility
        add_atom_at_position 
        add_fragment_at_position 
        delete_substructure_at_idx 
        REPLACE = delete + add 

    Output:
        new_smiles_set
    '''
    leaf2nonleaf = {leaf:nonleaf for leaf,nonleaf in leaf_nonleaf_lst}
    leaf2extend = {leaf:extend for leaf,extend in leaf_extend_idx_pair}
    new_smiles_set = set()
    #### 1. data preparation 
    origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(origin_smiles))
    origin_idx_lst, origin_node_mat, origin_substructure_lst, \
    origin_atomidx_2substridx, origin_adjacency_matrix, leaf_extend_idx_pair = smiles2graph(origin_smiles)
    node_indicator, adjacency_weight = differentiable_graph 
    N = len(origin_idx_lst)
    M = len(leaf_extend_idx_pair) 
    d = len(vocabulary)


    #### 2. edit the original molecule  
    ####### 2.1 delete & 2.2 replace 
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### single atom
            new_leaf_atom_idx_lst = [leaf_atom_idx_lst]
        else:  #### ring     
            ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
            new_leaf_atom_idx_lst = []
            remaining_atoms_idx_lst = []
            for i,v in enumerate(origin_substructure_lst):
                if i==leaf_idx:
                    continue 
                if type(v)==int:
                    remaining_atoms_idx_lst.append(v)
                else: #### list 
                    remaining_atoms_idx_lst.extend(v)
            new_leaf_atom_idx_lst = [leaf_atom_idx for leaf_atom_idx in leaf_atom_idx_lst if leaf_atom_idx not in remaining_atoms_idx_lst]
        ### leaf_atom_idx_lst v.s. new_leaf_atom_idx_lst 
        ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
        result = delete_substructure_at_idx(editmol = origin_mol, atom_idx_lst = new_leaf_atom_idx_lst) 
        if result is None: 
            continue
        delete_mol, old_idx2new_idx = result
        delete_smiles = Chem.MolToSmiles(delete_mol)
        if delete_smiles is None or '.' in delete_smiles:
            continue
        delete_smiles = canonical(delete_smiles)
        nonleaf_idx = leaf2nonleaf[leaf_idx]
        shrink_prob = (adjacency_weight[leaf_idx,nonleaf_idx] + adjacency_weight[nonleaf_idx,leaf_idx])/2
        if shrink_prob > -3: ### sigmoid(-3)=0.1
            new_smiles_set.add(delete_smiles)
        #### 2.1 delete done
        ####  2.2 replace  a & b 
        ######### (a) get neighbor substr
        neighbor_substructures_idx = [idx for idx,value in enumerate(origin_adjacency_matrix[leaf_idx]) if value==1]
        assert len(neighbor_substructures_idx)==1 
        neighbor_substructures_idx = neighbor_substructures_idx[0]
        neighbor_atom_idx_lst = origin_substructure_lst[neighbor_substructures_idx]
        if type(neighbor_atom_idx_lst)==int:
            neighbor_atom_idx_lst = [neighbor_atom_idx_lst] 
        ######### (b) add new substructure  todo, enumerate several possibility 
        added_substructure_lst = list(np.argsort(-node_indicator[leaf_idx]))[:topk]
        for substructure_idx in added_substructure_lst: 
            new_substructure = vocabulary[substructure_idx]
            for new_bond in bondtype_list:
                for leaf_atom_idx in neighbor_atom_idx_lst:
                    new_leaf_atom_idx = old_idx2new_idx[leaf_atom_idx] 
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                                    fragment = new_substructure, new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)
    expand_prob = (adjacency_weight[leaf_idx,extend_idx] + adjacency_weight[extend_idx,leaf_idx])/2
    if expand_prob < -3:
        return new_smiles_set.difference(set([None]))


    ####### 2.3 add   todo: use adjacency_weight to further narrow scope
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        expand_prob = (adjacency_weight[leaf_idx][extend_idx] + adjacency_weight[extend_idx][leaf_idx])/2  ### [-inf, inf]
        # print("expand prob", expand_prob)
        if expand_prob < -3:
            continue 
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### int: single atom;   else: list of integer
            leaf_atom_idx_lst = [leaf_atom_idx_lst]
        for leaf_atom_idx in leaf_atom_idx_lst:
            added_substructure_lst = list(np.argsort(-node_indicator[extend_idx]))[:topk]
            for substructure_idx in added_substructure_lst:
                new_substructure = vocabulary[substructure_idx]
                for new_bond in bondtype_list:
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                                    fragment = new_substructure , new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)

    return new_smiles_set.difference(set([None]))  




def differentiable_graph2smiles_sample(origin_smiles, differentiable_graph, 
                                leaf_extend_idx_pair, leaf_nonleaf_lst, 
                                topk, epsilon):
    '''
        origin_smiles:
            origin_idx_lst              [N]      0,1,...,d-1 
            origin_node_mat             [N,d]
            origin_substructure_lst     
            origin_atomidx_2substridx   
            origin_adjacency_matrix     [N,N]    0/1

        differentiable_graph:   returned results 
            node_indicator              [N+M,d]
            adjacency_weight            [N+M,N+M]

        N is # of substructures in the molecule
        M is # of leaf node, also number of extended node. 


    main utility
        add_atom_at_position 
        add_fragment_at_position 
        delete_substructure_at_idx 
        REPLACE = delete + add 

    Output:
        new_smiles_set
    '''
    leaf2nonleaf = {leaf:nonleaf for leaf,nonleaf in leaf_nonleaf_lst}
    leaf2extend = {leaf:extend for leaf,extend in leaf_extend_idx_pair}
    new_smiles_set = set()
    #### 1. data preparation 
    origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(origin_smiles))
    origin_idx_lst, origin_node_mat, origin_substructure_lst, \
    origin_atomidx_2substridx, origin_adjacency_matrix, leaf_extend_idx_pair = smiles2graph(origin_smiles)
    node_indicator, adjacency_weight = differentiable_graph 
    N = len(origin_idx_lst)
    M = len(leaf_extend_idx_pair) 
    d = len(vocabulary)


    #### 2. edit the original molecule  
    ####### 2.1 delete & 2.2 replace 
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### single atom
            new_leaf_atom_idx_lst = [leaf_atom_idx_lst]
        else:  #### ring     
            ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
            new_leaf_atom_idx_lst = []
            remaining_atoms_idx_lst = []
            for i,v in enumerate(origin_substructure_lst):
                if i==leaf_idx:
                    continue 
                if type(v)==int:
                    remaining_atoms_idx_lst.append(v)
                else: #### list 
                    remaining_atoms_idx_lst.extend(v)
            new_leaf_atom_idx_lst = [leaf_atom_idx for leaf_atom_idx in leaf_atom_idx_lst if leaf_atom_idx not in remaining_atoms_idx_lst]
        ### leaf_atom_idx_lst v.s. new_leaf_atom_idx_lst 
        ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
        result = delete_substructure_at_idx(editmol = origin_mol, atom_idx_lst = new_leaf_atom_idx_lst) 
        if result is None: 
            continue
        delete_mol, old_idx2new_idx = result
        delete_smiles = Chem.MolToSmiles(delete_mol)
        if delete_smiles is None or '.' in delete_smiles:
            continue
        delete_smiles = canonical(delete_smiles)
        nonleaf_idx = leaf2nonleaf[leaf_idx]
        shrink_prob = (adjacency_weight[leaf_idx,nonleaf_idx] + adjacency_weight[nonleaf_idx,leaf_idx])/2
        if shrink_prob > -3: ### sigmoid(-3)=0.1
            new_smiles_set.add(delete_smiles)
        #### 2.1 delete done
        ####  2.2 replace  a & b 
        ######### (a) get neighbor substr
        neighbor_substructures_idx = [idx for idx,value in enumerate(origin_adjacency_matrix[leaf_idx]) if value==1]
        assert len(neighbor_substructures_idx)==1 
        neighbor_substructures_idx = neighbor_substructures_idx[0]
        neighbor_atom_idx_lst = origin_substructure_lst[neighbor_substructures_idx]
        if type(neighbor_atom_idx_lst)==int:
            neighbor_atom_idx_lst = [neighbor_atom_idx_lst] 
        ######### (b) add new substructure  todo, enumerate several possibility 
        u = random.random()
        if u < epsilon:
            added_substructure_lst = list(np.argsort(-node_indicator[leaf_idx]))[:topk]  ### topk (greedy)
        else:
            added_substructure_lst = random.choices(population=list(range(len(vocabulary))), weights = node_indicator[leaf_idx], k=topk + 3)
            added_substructure_lst = list(set(added_substructure_lst))[:topk]  ### avoid repetition
        for substructure_idx in added_substructure_lst: 
            new_substructure = vocabulary[substructure_idx]
            for new_bond in bondtype_list:
                for leaf_atom_idx in neighbor_atom_idx_lst:
                    new_leaf_atom_idx = old_idx2new_idx[leaf_atom_idx] 
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                                    fragment = new_substructure, new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)
    expand_prob = (adjacency_weight[leaf_idx,extend_idx] + adjacency_weight[extend_idx,leaf_idx])/2
    if expand_prob < -3:
        return new_smiles_set.difference(set([None]))


    ####### 2.3 add   todo: use adjacency_weight to further narrow scope
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        expand_prob = (adjacency_weight[leaf_idx][extend_idx] + adjacency_weight[extend_idx][leaf_idx])/2  ### [-inf, inf]
        # print("expand prob", expand_prob)
        if expand_prob < -3:
            continue 
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### int: single atom;   else: list of integer
            leaf_atom_idx_lst = [leaf_atom_idx_lst]
        for leaf_atom_idx in leaf_atom_idx_lst:
            u = random.random() 
            if u < epsilon:
                added_substructure_lst = list(np.argsort(-node_indicator[extend_idx]))[:topk] 
            else:
                added_substructure_lst = random.choices(population=list(range(len(vocabulary))), weights = node_indicator[extend_idx], k=topk + 3)
                added_substructure_lst = list(set(added_substructure_lst))[:topk]  ### avoid repetition
            for substructure_idx in added_substructure_lst:
                new_substructure = vocabulary[substructure_idx]
                for new_bond in bondtype_list:
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                                    fragment = new_substructure , new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)

    return new_smiles_set.difference(set([None]))  



def differentiable_graph2smiles_sample_v2(origin_smiles, differentiable_graph, 
                                leaf_extend_idx_pair, leaf_nonleaf_lst, 
                                topk, epsilon):
    '''
        origin_smiles:
            origin_idx_lst              [N]      0,1,...,d-1 
            origin_node_mat             [N,d]
            origin_substructure_lst     
            origin_atomidx_2substridx   
            origin_adjacency_matrix     [N,N]    0/1

        differentiable_graph:   returned results 
            node_indicator              [N+M,d]
            adjacency_weight            [N+M,N+M]

        N is # of substructures in the molecule
        M is # of leaf node, also number of extended node. 

    main utility
        add_atom_at_position 
        add_fragment_at_position 
        delete_substructure_at_idx 
        REPLACE = delete + add 

    Output:
        new_smiles_set
    '''
    leaf2nonleaf = {leaf:nonleaf for leaf,nonleaf in leaf_nonleaf_lst}
    leaf2extend = {leaf:extend for leaf,extend in leaf_extend_idx_pair}
    new_smiles_set = set()
    #### 1. data preparation 
    origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(origin_smiles))
    origin_idx_lst, origin_node_mat, origin_substructure_lst, \
    origin_atomidx_2substridx, origin_adjacency_matrix, leaf_extend_idx_pair = smiles2graph(origin_smiles)
    node_indicator, adjacency_weight = differentiable_graph  #### both are np.array 
    N = len(origin_idx_lst)
    M = len(leaf_extend_idx_pair) 
    d = len(vocabulary)


    #### 2. edit the original molecule  
    ####### 2.1 delete & 2.2 replace 
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### single atom
            new_leaf_atom_idx_lst = [leaf_atom_idx_lst]
        else:  #### ring     
            ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
            new_leaf_atom_idx_lst = []
            remaining_atoms_idx_lst = []
            for i,v in enumerate(origin_substructure_lst):
                if i==leaf_idx:
                    continue 
                if type(v)==int:
                    remaining_atoms_idx_lst.append(v)
                else: #### list 
                    remaining_atoms_idx_lst.extend(v)
            new_leaf_atom_idx_lst = [leaf_atom_idx for leaf_atom_idx in leaf_atom_idx_lst if leaf_atom_idx not in remaining_atoms_idx_lst]
        ### leaf_atom_idx_lst v.s. new_leaf_atom_idx_lst 
        ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
        result = delete_substructure_at_idx(editmol = origin_mol, atom_idx_lst = new_leaf_atom_idx_lst) 
        if result is None: 
            continue
        delete_mol, old_idx2new_idx = result
        delete_smiles = Chem.MolToSmiles(delete_mol)
        if delete_smiles is None or '.' in delete_smiles:
            continue
        delete_smiles = canonical(delete_smiles)
        nonleaf_idx = leaf2nonleaf[leaf_idx]
        u = random.random() 
        shrink_prob = sigmoid(adjacency_weight[leaf_idx,nonleaf_idx]) + sigmoid(adjacency_weight[nonleaf_idx,leaf_idx])
        if u < shrink_prob:
            new_smiles_set.add(delete_smiles) 
        # if shrink_prob < 0: ### sigmoid(-3)=0.1
        #     new_smiles_set.add(delete_smiles)
        #### 2.1 delete done
        ####  2.2 replace  a & b 
        ######### (a) get neighbor substr
        neighbor_substructures_idx = [idx for idx,value in enumerate(origin_adjacency_matrix[leaf_idx]) if value==1]
        assert len(neighbor_substructures_idx)==1 
        neighbor_substructures_idx = neighbor_substructures_idx[0]
        neighbor_atom_idx_lst = origin_substructure_lst[neighbor_substructures_idx]
        if type(neighbor_atom_idx_lst)==int:
            neighbor_atom_idx_lst = [neighbor_atom_idx_lst] 
        ######### (b) add new substructure  todo, enumerate several possibility 
        u = random.random()

        node_indicator_leaf = node_indicator[leaf_idx]  ### before softmax
        node_indicator_leaf[12:] -= 5
        node_indicator_leaf = np.exp(node_indicator_leaf)
        node_indicator_leaf = node_indicator_leaf / np.sum(node_indicator_leaf)
        if u < epsilon:
            added_substructure_lst = list(np.argsort(-node_indicator_leaf))[:topk]  ### topk (greedy)
        else:
            added_substructure_lst = random.choices(population=list(range(len(vocabulary))), weights = node_indicator_leaf, k=topk + 3)
            added_substructure_lst = list(set(added_substructure_lst))[:topk]  ### avoid repetition
        for substructure_idx in added_substructure_lst: 
            new_substructure = vocabulary[substructure_idx]
            for new_bond in bondtype_list:
                for leaf_atom_idx in neighbor_atom_idx_lst:
                    new_leaf_atom_idx = old_idx2new_idx[leaf_atom_idx] 
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                                    fragment = new_substructure, new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)

    expand_prob = sigmoid(adjacency_weight[leaf_idx,extend_idx]) + sigmoid(adjacency_weight[extend_idx,leaf_idx])/2
    u = random.random() 
    if u > expand_prob:
        return new_smiles_set.difference(set([None]))


    ####### 2.3 add   todo: use adjacency_weight to further narrow scope
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        expand_prob = (adjacency_weight[leaf_idx][extend_idx] + adjacency_weight[extend_idx][leaf_idx])/2  ### [-inf, inf]
        # print("expand prob", expand_prob)
        if expand_prob < -3:
            continue 
        leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
        if type(leaf_atom_idx_lst)==int:  ### int: single atom;   else: list of integer
            leaf_atom_idx_lst = [leaf_atom_idx_lst]
        for leaf_atom_idx in leaf_atom_idx_lst:
            u = random.random() 
            node_indicator_leaf = node_indicator[extend_idx]
            node_indicator_leaf[12:]-=5
            node_indicator_leaf = np.exp(node_indicator_leaf)
            node_indicator_leaf = node_indicator_leaf / np.sum(node_indicator_leaf)
            if u < epsilon:
                added_substructure_lst = list(np.argsort(-node_indicator_leaf))[:topk] 
            else:
                added_substructure_lst = random.choices(population=list(range(len(vocabulary))), weights = node_indicator_leaf, k=topk + 3)
                added_substructure_lst = list(set(added_substructure_lst))[:topk]  ### avoid repetition
            for substructure_idx in added_substructure_lst:
                new_substructure = vocabulary[substructure_idx]
                for new_bond in bondtype_list:
                    if ith_substructure_is_atom(substructure_idx):
                        new_smiles = add_atom_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                        new_smiles_set.add(new_smiles)
                    else:
                        new_smiles_batch = add_fragment_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                                    fragment = new_substructure , new_bond = new_bond)
                        new_smiles_set = new_smiles_set.union(new_smiles_batch)

    return new_smiles_set.difference(set([None])) 


def differentiable_graph_to_smiles_purely_randomwalk(origin_smiles, differentiable_graph, 
                                             leaf_extend_idx_pair, leaf_nonleaf_lst, 
                                             topk = 3, epsilon = 0.7,):
    # print(origin_smiles)
    leaf2nonleaf = {leaf:nonleaf for leaf,nonleaf in leaf_nonleaf_lst}
    leaf2extend = {leaf:extend for leaf,extend in leaf_extend_idx_pair}
    new_smiles_set = set()
    #### 1. data preparation 
    origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(origin_smiles))
    origin_idx_lst, origin_node_mat, origin_substructure_lst, \
    origin_atomidx_2substridx, origin_adjacency_matrix, leaf_extend_idx_pair = smiles2graph(origin_smiles)
    node_indicator, adjacency_weight = differentiable_graph 
    N = len(origin_idx_lst)
    M = len(leaf_extend_idx_pair) 
    d = len(vocabulary)



    #### 2. edit the original molecule  
    ####### 2.1 delete & 2.2 replace 
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        u_shrink = random.random() 
        shrink, unchange, expand = False, False, False 
        if u_shrink < 0.7 and substr_num(origin_smiles) > 1:
            shrink = True 
        else:
            u_expand = random.random()
            if u_expand < 0.3:
                expand = True 
            else:
                unchange = True  

        if shrink or unchange:
            leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
            if type(leaf_atom_idx_lst)==int:  ### single atom
                new_leaf_atom_idx_lst = [leaf_atom_idx_lst]
            else:  #### ring     
                ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
                new_leaf_atom_idx_lst = []
                remaining_atoms_idx_lst = []
                for i,v in enumerate(origin_substructure_lst):
                    if i==leaf_idx:
                        continue 
                    if type(v)==int:
                        remaining_atoms_idx_lst.append(v)
                    else: #### list 
                        remaining_atoms_idx_lst.extend(v)
                new_leaf_atom_idx_lst = [leaf_atom_idx for leaf_atom_idx in leaf_atom_idx_lst if leaf_atom_idx not in remaining_atoms_idx_lst]
            ### leaf_atom_idx_lst v.s. new_leaf_atom_idx_lst 
            ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
            result = delete_substructure_at_idx(editmol = origin_mol, atom_idx_lst = new_leaf_atom_idx_lst) 
            if result is None: 
                continue
            delete_mol, old_idx2new_idx = result
            delete_smiles = Chem.MolToSmiles(delete_mol)
            if delete_smiles is None or '.' in delete_smiles:
                continue
            delete_smiles = canonical(delete_smiles)
            nonleaf_idx = leaf2nonleaf[leaf_idx]

            if shrink: 
                new_smiles_set.add(delete_smiles)
                continue 
            #### 2.1 delete done
            ####  2.2 replace  a & b 
            ######### (a) get neighbor substr
            neighbor_substructures_idx = [idx for idx,value in enumerate(origin_adjacency_matrix[leaf_idx]) if value==1]
            assert len(neighbor_substructures_idx)==1 
            neighbor_substructures_idx = neighbor_substructures_idx[0]
            neighbor_atom_idx_lst = origin_substructure_lst[neighbor_substructures_idx]
            if type(neighbor_atom_idx_lst)==int:
                neighbor_atom_idx_lst = [neighbor_atom_idx_lst] 
            ######### (b) add new substructure  todo, enumerate several possibility 
            # added_substructure_lst = list(np.argsort(-node_indicator[leaf_idx]))[:topk]
            added_substructure_lst = [random.choice(list(range(len(vocabulary)))) for i in range(topk)]
            for substructure_idx in added_substructure_lst: 
                new_substructure = vocabulary[substructure_idx]
                for new_bond in bondtype_list:
                    for leaf_atom_idx in neighbor_atom_idx_lst:
                        new_leaf_atom_idx = old_idx2new_idx[leaf_atom_idx] 
                        if ith_substructure_is_atom(substructure_idx):
                            new_smiles = add_atom_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                            new_smiles_set.add(new_smiles)
                        else:
                            new_smiles_batch = add_fragment_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                                    fragment = new_substructure, new_bond = new_bond)
                            new_smiles_set = new_smiles_set.union(new_smiles_batch)
            continue ### end of shrink or unchange 

        ####### 2.3 add   todo: use adjacency_weight to further narrow scope
        for leaf_idx, extend_idx in leaf_extend_idx_pair:
            leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
            if type(leaf_atom_idx_lst)==int:  ### int: single atom;   else: list of integer
                leaf_atom_idx_lst = [leaf_atom_idx_lst]
            for leaf_atom_idx in leaf_atom_idx_lst:
                added_substructure_lst = [random.choice(list(range(len(vocabulary)))) for i in range(topk)]
                for substructure_idx in added_substructure_lst:
                    new_substructure = vocabulary[substructure_idx]
                    for new_bond in bondtype_list:
                        if ith_substructure_is_atom(substructure_idx):
                            new_smiles = add_atom_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                          new_atom = new_substructure, new_bond = new_bond)
                            new_smiles_set.add(new_smiles)
                        else:
                            new_smiles_batch = add_fragment_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                                    fragment = new_substructure , new_bond = new_bond)
                            new_smiles_set = new_smiles_set.union(new_smiles_batch)

    return new_smiles_set.difference(set([None]))  




def differentiable_graph2smiles_plus_random(origin_smiles, differentiable_graph, 
                                             leaf_extend_idx_pair, leaf_nonleaf_lst, 
                                             max_num_offspring = 100, topk = 3, epsilon = 0.7,
                                             random_topology = False, random_substr = False):
    '''
        origin_smiles:
            origin_idx_lst              [N]      0,1,...,d-1 
            origin_node_mat             [N,d]
            origin_substructure_lst     
            origin_atomidx_2substridx   
            origin_adjacency_matrix     [N,N]    0/1

        differentiable_graph:   returned results 
            node_indicator              [N+M,d]
            adjacency_weight            [N+M,N+M]

        N is # of substructures in the molecule
        M is # of leaf node, also number of extended node. 


    main utility
        add_atom_at_position 
        add_fragment_at_position 
        delete_substructure_at_idx 
        REPLACE = delete + add 

    Output:
        new_smiles_set
    '''
    leaf2nonleaf = {leaf:nonleaf for leaf,nonleaf in leaf_nonleaf_lst}
    leaf2extend = {leaf:extend for leaf,extend in leaf_extend_idx_pair}
    new_smiles_set = set()
    #### 1. data preparation 
    origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(origin_smiles))
    origin_idx_lst, origin_node_mat, origin_substructure_lst, \
    origin_atomidx_2substridx, origin_adjacency_matrix, leaf_extend_idx_pair = smiles2graph(origin_smiles)
    node_indicator, adjacency_weight = differentiable_graph 
    N = len(origin_idx_lst)
    M = len(leaf_extend_idx_pair) 
    d = len(vocabulary)

    u_topology = random.random() 
    ### shrink, unchange, expand prob = 0.2, 0.3, 0.5 
    shrink, unchange, expand = False, False, False 
    for leaf_idx, extend_idx in leaf_extend_idx_pair:
        u_topology = random.random()
        #### 1. topology 
        if random_topology:
            # if u_topology < 0.1:
            #     shrink = True
            # elif 0.4 > u_topology >= 0.2:
            #     unchange = True
            if u_topology < 0.2:
                unchange = True 
            else:
                expand = True 
        else: ## dmg topology 
            nonleaf_idx = leaf2nonleaf[leaf_idx]
            shrink_prob = sigmoid((adjacency_weight[leaf_idx,nonleaf_idx] + adjacency_weight[nonleaf_idx,leaf_idx])/2)
            # if u_topology < shrink_prob:
            if False:
                shrink = True 
            else:
                u_topology2 = random.random() 
                expand_prob = (adjacency_weight[leaf_idx,extend_idx] + adjacency_weight[extend_idx,leaf_idx])/2
                if u_topology2 < expand_prob:
                    expand_prob = True
                else:
                    unchange = True 

        if shrink or unchange: 

            leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
            if type(leaf_atom_idx_lst)==int:  ### single atom
                new_leaf_atom_idx_lst = [leaf_atom_idx_lst]
            else:  #### ring     
                ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
                new_leaf_atom_idx_lst = []
                remaining_atoms_idx_lst = []
                for i,v in enumerate(origin_substructure_lst):
                    if i==leaf_idx:
                        continue 
                    if type(v)==int:
                        remaining_atoms_idx_lst.append(v)
                    else: #### list 
                        remaining_atoms_idx_lst.extend(v)
                new_leaf_atom_idx_lst = [leaf_atom_idx for leaf_atom_idx in leaf_atom_idx_lst if leaf_atom_idx not in remaining_atoms_idx_lst]
            ### leaf_atom_idx_lst v.s. new_leaf_atom_idx_lst 
            ### consider the case that ring1 and ring2 share 2 atoms and 1 bond. 
            result = delete_substructure_at_idx(editmol = origin_mol, atom_idx_lst = new_leaf_atom_idx_lst) 
            if result is None: 
                continue
            delete_mol, old_idx2new_idx = result
            delete_smiles = Chem.MolToSmiles(delete_mol)
            if delete_smiles is None or '.' in delete_smiles:
                continue
            delete_smiles = canonical(delete_smiles)
            if shrink:
                new_smiles_set.add(delete_smiles)
            if unchange:
                ######### (a) get neighbor substr
                neighbor_substructures_idx = [idx for idx,value in enumerate(origin_adjacency_matrix[leaf_idx]) if value==1]
                assert len(neighbor_substructures_idx)==1 
                neighbor_substructures_idx = neighbor_substructures_idx[0]
                neighbor_atom_idx_lst = origin_substructure_lst[neighbor_substructures_idx]
                if type(neighbor_atom_idx_lst)==int:
                    neighbor_atom_idx_lst = [neighbor_atom_idx_lst] 
                ######### (b) add new substructure  todo, enumerate several possibility 
                if random_substr: ## random sample 
                    added_substructure_lst = random.choices(list(range(len(vocabulary))), k=topk)
                else: ## dmg sampling
                    u = random.random()
                    if u < epsilon:
                        added_substructure_lst = list(np.argsort(-node_indicator[leaf_idx]))[:topk]  ### topk (greedy)
                    else:
                        added_substructure_lst = random.choices(population=list(range(len(vocabulary))), weights = node_indicator[leaf_idx], k=topk + 3)
                        added_substructure_lst = list(set(added_substructure_lst))[:topk]  ### avoid repetition
                for substructure_idx in added_substructure_lst: 
                    new_substructure = vocabulary[substructure_idx]
                    for new_bond in bondtype_list:
                        for leaf_atom_idx in neighbor_atom_idx_lst:
                            new_leaf_atom_idx = old_idx2new_idx[leaf_atom_idx] 
                            if ith_substructure_is_atom(substructure_idx):
                                new_smiles = add_atom_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                                  new_atom = new_substructure, new_bond = new_bond)
                                new_smiles_set.add(new_smiles)
                            else:
                                new_smiles_batch = add_fragment_at_position(editmol = delete_mol, position_idx = new_leaf_atom_idx, 
                                                                            fragment = new_substructure, new_bond = new_bond)
                                new_smiles_set = new_smiles_set.union(new_smiles_batch)
        else:  ## expand 

            leaf_atom_idx_lst = origin_substructure_lst[leaf_idx]
            if type(leaf_atom_idx_lst)==int:  ### int: single atom;   else: list of integer
                leaf_atom_idx_lst = [leaf_atom_idx_lst]
            if random_substr:
                added_substructure_lst = random.choices(list(range(len(vocabulary))), k=topk)
            else:    
                for leaf_atom_idx in leaf_atom_idx_lst:
                    u = random.random() 
                    if u < epsilon:
                        added_substructure_lst = list(np.argsort(-node_indicator[extend_idx]))[:topk] 
                    else:
                        added_substructure_lst = random.choices(population=list(range(len(vocabulary))), weights = node_indicator[extend_idx], k=topk + 3)
                        added_substructure_lst = list(set(added_substructure_lst))[:topk]  ### avoid repetition
                    for substructure_idx in added_substructure_lst:
                        new_substructure = vocabulary[substructure_idx]
                        for new_bond in bondtype_list:
                            if ith_substructure_is_atom(substructure_idx):
                                new_smiles = add_atom_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                                  new_atom = new_substructure, new_bond = new_bond)
                                new_smiles_set.add(new_smiles)
                            else:
                                new_smiles_batch = add_fragment_at_position(editmol = origin_mol, position_idx = leaf_atom_idx, 
                                                                            fragment = new_substructure , new_bond = new_bond)
                                new_smiles_set = new_smiles_set.union(new_smiles_batch)



    return new_smiles_set.difference(set([None]))  


def draw_smiles(smiles, figfile_name):
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToImageFile(mol, figfile_name, size = (300,180))
    return 





if __name__ == "__main__":

    # s = 'FC1NCC(-C1=CC=CC(Br)=C1)C1'
    s = 'C1=CC=CC=C1NC2=NC=CC(F)=N2'
    draw_smiles(s, "figure/tmp.png")
    # rawdata_file = "raw_data/zinc.tab"
    # with open(rawdata_file) as fin:
    #     lines = fin.readlines()[1:]
    #     smiles_lst = [line.strip().strip('"') for line in lines]

    # from random import shuffle 
    # # shuffle(smiles_lst)
    # fragment_lst = ['C1NCC1', 'C1CNCCN1', 'C1=CC=CC=C1', 'C1CNNC1']


    # smiles = smiles_lst[0]
    # differentiable_graph = smiles2differentiable_graph(smiles)  
    # ### optimize differentiable_graph using GNN   
    # smiles_set = differentiable_graph2smiles(origin_smiles = smiles, differentiable_graph = differentiable_graph, max_num_offspring = 100)
    # print(len(smiles_set))

    # s = "CCc1ccc(Nc2nc(-c3ccccc3)cs2)cc1"
    # s = 'Oc1ccc(Nc2nc(-c3ccc(Cl)cc3)cs2)cc1'
    # draw_smiles(s, "figure/tmp.png")
    # from tdc import Oracle 
    # qed = Oracle('qed')
    # logp = Oracle('logp')
    # jnk = Oracle('jnk3')
    # gsk = Oracle('gsk3b')
    # print(qed(s), logp(s), jnk(s), gsk(s))


    # smiles_lst = ['NO', 'ONO', 'CNO', 'CS']
    # print(similarity_matrix(smiles_lst))



    ##### test over zinc 
    # for smiles in tqdm(smiles_lst):
    #     mol = Chem.MolFromSmiles(smiles)
    #     print(smiles)
    #     new_smiles_lst = []
    #     for idx in range(mol.GetNumAtoms()):
    #         for fragment in fragment_lst:
    #             smiles_set = add_fragment_at_position(editmol = mol, position_idx = idx, fragment = fragment, new_bond = bondtype_list[0])
    #             new_smiles_lst.extend(list(smiles_set))
    #         new_smiles_lst = list(set(new_smiles_lst))
    #     print("length of smiles set is", len(new_smiles_lst))



    ### single test
    # smiles = 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'
    # draw_smiles(smiles, "figure/origin.png")
    # fragment = 'C1CCNCN1'
    # mol = Chem.MolFromSmiles(smiles)
    # for idx in range(mol.GetNumAtoms()):
    #     smiles_set = add_fragment_at_position(editmol = mol, position_idx = idx, fragment = fragment, new_bond = bondtype_list[0])
    #     print("length of smiles set is", len(smiles_set), smiles_set)
    #     for i,smiles in enumerate(smiles_set):
    #         name = "figure/" + str(idx) + '_' + str(i) + '.png'
    #         draw_smiles(smiles, name)




'''

"CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
"C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1",
"N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1",
"CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1",
"N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])[C@H](C#N)C12CCCCC2",
"CC[NH+](CC)[C@](C)(CC)[C@H](O)c1cscc1Br"

CCc1ccc(Nc2nc(-c3ccccc3)cs2)cc1



rawdata_file = "raw_data/zinc.tab"
with open(rawdata_file) as fin:
	lines = fin.readlines()[1:]
	smiles_lst = [line.strip().strip('"') for line in lines]



test case:
    
    smiles         fragment 
    C1CCCC1         C1NCC1
    C1=CC=CC=C1    C1CNCCN1  
    C1=CC=CC=C1    C1CCNCN1
    CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1   
'''


