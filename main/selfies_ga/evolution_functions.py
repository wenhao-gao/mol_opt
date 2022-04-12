from __future__ import print_function
import os
import rdkit
import shutil
import multiprocessing
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import Descriptors
from selfies import decoder 
import numpy as np
import inspect
from collections import OrderedDict
manager = multiprocessing.Manager()
lock = multiprocessing.Lock()


def get_logP(mol):
    '''Calculate logP of a molecule 
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object, for which logP is to calculates
    
    Returns:
    float : logP of molecule (mol)
    '''
    return Descriptors.MolLogP(mol)
    

def make_clean_results_dir():
    # Create the results folder 
    root_folder = './results'
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    else:
        shutil.rmtree(root_folder)
        os.makedirs(root_folder)
    return root_folder


def make_clean_directories(beta, root_folder, iteration):
    '''Create or clean directories: 'images' & 'saved_models'
    
    Create directories from scratch, if they do not exist
    Clean (remove all content) if directories already exist
    
    Parameters:
    None
    
    Returns:
    None    : Folders in current directory modified
    '''
    image_dir= root_folder + '/images_generation_' + str(beta) + '_' + str(iteration)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    else:
        if len(os.listdir(image_dir)) > 0:
            os.system("rm -r %s/*"%(image_dir))

    models_dir = root_folder + '/saved_models_' + str(beta) + '_' + str(iteration)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    else:
        if len(os.listdir(models_dir)) > 0:
            os.system("rm -r %s/*"%(models_dir))            
            
    data_dir = root_folder + '/results_' + str(beta) + '_' + str(iteration)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
        if len(os.listdir(data_dir)) > 0:
            os.system("rm -r %s/*"%(data_dir))            


    return (image_dir, models_dir, data_dir)



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
    
    
def sanitize_multiple_smiles(smi_ls):
    '''Calls function sanitize_smiles for each item in list smi_ls
    '''
    sanitized_smiles = []
    for smi in smi_ls:
        smi_converted = sanitize_smiles(smi)
        sanitized_smiles.append(smi_converted[1])
        if smi_converted[2] == False or smi_converted[1] == '':
            raise Exception("Invalid SMILE ecncountered. Value =", smi)
    return sanitized_smiles
    

def read_dataset(filename):
    '''Return a list of smiles contained in file filename
    
    Parameters:
    filename (string) : Name of file containg smiles seperated by '\n'
    
    Returns
    content  (list)   : list of smile string in file filename
    '''
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 

    return content


def read_dataset_encoding(disc_enc_type):
    '''Return zinc-data set based on disc_enc_type choice of 'smiles' or 'selfies'
    
    Parameters:
    disc_enc_type (string): 'smiles' or 'selfies'
    '''
    if disc_enc_type == 'smiles' or disc_enc_type == 'properties_rdkit':
        smiles_reference = read_dataset(filename='./main/selfies_ga/datasets/zinc_dearom.txt')
        return smiles_reference
    elif disc_enc_type == 'selfies':
        selfies_reference = read_dataset(filename='./main/selfies_ga/datasets/SELFIES_zinc.txt')
        return selfies_reference
    
    
def create_100_mol_image(mol_list, file_name, fitness, logP, SAS, RingCount, discr_scores):
    '''Create a single picture of multiple molecules in a single Grid. Property information is
       added below each molecule
    '''
    assert len(mol_list) == 100
    if logP == None and SAS == None and RingCount == None and discr_scores == None:
        Draw.MolsToGridImage(mol_list, molsPerRow=10, subImgSize=(200,200)).save(file_name)
        return

    for i,m in enumerate(mol_list):
        m.SetProp('_Name','%s %s %s %s %s' % (round(fitness[i], 3), round(logP[i], 3), round(SAS[i], 3), round(RingCount[i], 3), round(discr_scores[i][0], 3))) 
    try:
        Draw.MolsToGridImage(mol_list, molsPerRow=10, subImgSize=(200,200), legends=[x.GetProp("_Name") for x in mol_list]).save(file_name)
    except:
        print('Failed to produce image!')
    return 
    



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


def smiles_alphabet(disc_enc_type):
    '''Return a list of characters present in the zinc dataset
    
    Parameters:
    disc_enc_type (string): Indicates whether to return SMILES/SELFiES characters 
    
    Returns:
    alphabet: list of SELFIE/SMILE alphabets in Zinc
    '''
    if disc_enc_type == 'smiles':   
        alphabet = ['C', 'c', 'H','O','o', 'N','n', 'S','s', 'F', 'P', 'I',
                    'Cl','Br', '=','#','(',')','[',']','1','2','3','4','5',
                    '6','7','8','9','+','-','X'] # SMILES Alphabets in zinc
    
    elif disc_enc_type == 'selfies':
        alphabet = ['[Ring1]',   '[Branch1_1]', '[Branch1_2]','[Branch1_3]', '[Cl]', 
                    '[Ring2]',   '[Branch2_1]', '[Branch2_2]','[Branch2_3]', '[NH3+]',
                    '[N]',       '[=N]',        '[#N]',       '[C]',         '[=C]', 
                    '[#C]',      '[S]',         '[=S]',       '[=O]',        '[Br]',
                    '[epsilon]', '[N+]',        '[NH+]',      '[NH2+]',      '[=NH+]',
                    '[=NH2+]',   '[I]',         '[O-]',       '[P]',         '[=P]', 
                    '[S-]',      '[=N-]',       '[NH-]',      '[=O+]',       '[CH-]', 
                    '[PH+]',     '[=S+]',       '[S+]',       '[CH2-]',      '[P+]',
                    '[O+]',      '[=N+]',       '[N-]' ,       '[=SH+]',     '[=OH+]',
                    '[#N+]',     '[=PH2]',      'X',           '[F]',        '[O]',
                   ] # SELFIES Alphabets in zinc     
    else:
        exit('Invalid choice. Only possible choices are: smiles/selfies.')
        
    return alphabet


def _to_onehot(molecule_str, disc_enc_type, max_molecules_len):
    '''Convert given molecule string into a one-hot encoding, with characters 
       obtained from function 'smiles_alphabet'.
    
    One-hot encoding of arbitrary molecules is converted to len 
    'max_molecules_len' by padding with character 'X'
       
    Parameters:
    molecule_str      (string): SMILE/SELFIE string of molecule 
    disc_enc_type     (string): Indicating weather molecule string is either
                                SMILE or SELFIE 
    max_molecules_len (string): Length of the one-hot encoding 

    
    Returns:
    one_hots   (list of lists): One-Hot encoding of molecule string, padding 
                                till length max_molecules_len (dim: len(alphabet) * max_molecules_len)
    '''
    one_hots=[]
    alphabet = smiles_alphabet(disc_enc_type)
    alphabet_length = len(alphabet)

    if disc_enc_type == 'smiles': 
        alphabet.remove('Cl')      # Replace 'Cl' & 'Br' with 'Y' & 'Z' for convenience
        alphabet.remove('Br')      # (Searching for single characters is easier)
        alphabet.append('Y')
        alphabet.append('Z')
    
    for smi in molecule_str:
        # Relace 'Cl' and 'Br' with 'Y', 'Z' from smi (for conveninece)
        if disc_enc_type == 'smiles':
            smi = smi.replace('Cl', 'Y')
            smi = smi.replace('Br', 'Z')
            
        one_hot=[]
        
        if disc_enc_type == 'selfies':
            smi = get_selfie_chars(smi)
        if len(smi) > max_molecules_len:
            exit("Molecule is too large!")
        for char in smi:
            if char not in alphabet:
                print("smiles character %s not in alphabet MOLECULE: %s"%(char, smi))
            zeros = np.zeros((alphabet_length)).astype(np.int32).tolist()
            zeros[alphabet.index(char)] = 1
            one_hot+=zeros
            
        # Padding with 'X's
        for char in range(max_molecules_len-len(smi)):
            zeros = np.zeros((alphabet_length)).astype(np.int32).tolist()
            zeros[alphabet.index("X")] = 1
            one_hot += zeros
        one_hots.append(one_hot)
    one_hots = np.array(one_hots)
    return (one_hots)


def mutations_random_grin(selfie, max_molecules_len, write_fail_cases=False):
    '''Return a mutated selfie string
    
    Mutations are done until a valid molecule is obtained 
    Rules of mutation: With a 50% propbabily, either: 
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
    
    Parameters:
    selfie            (string)  : SELFIE string to be mutated 
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"
    
    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    '''
    valid=False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)
    
    while not valid:
        fail_counter += 1
                
        alphabet = ['[Branch1_1]', '[Branch1_2]','[Branch1_3]', '[epsilon]', '[Ring1]', '[Ring2]', '[Branch2_1]', '[Branch2_2]', '[Branch2_3]', '[F]', '[O]', '[=O]', '[N]', '[=N]', '[#N]', '[C]', '[=C]', '[#C]', '[S]', '[=S]', '[C][=C][C][=C][C][=C][Ring1][Branch1_1]']

        # Insert a character in a Random Location
        if np.random.random() < 0.5: 
            random_index = np.random.randint(len(chars_selfie)+1)
            random_character = np.random.choice(alphabet, size=1)[0]

            selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]

        # Replace a random character 
        else:                         
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]

                
        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)
        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(smiles_canon) > max_molecules_len or smiles_canon=="":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid=False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write('Tried to mutate SELFIE: '+str(sf)+' To Obtain: '+str(selfie_mutated) + '\n')
                f.close()
    return (selfie_mutated, smiles_canon)


def count_atoms(mol, atomic_num):
    '''Count the number of atoms in mol with atomic number atomic_num
    
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : Molecule in which search is conducted
    atomic_num            (int) : Counting is done in mol for atoms with this atomic number

    Returns:
    (int) :  final count of atom
    '''
    pat = Chem.MolFromSmarts("[#{}]".format(atomic_num))
    return len(mol.GetSubstructMatches(pat))


def get_num_bond_types(mol):
    '''Calculate the ratio of total number of  (single, double, triple, aromatic) bonds to the 
       total number of bonds. 
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : Molecule for which ratios arre retuned 
    
    Returns:
    (list):  [num_single/num_bonds, num_double/num_bonds, num_triple/num_bonds, num_aromatic/num_bonds]
    '''
    bonds = mol.GetBonds()    
    
    num_bonds    = 0
    num_double   = 0
    num_triple   = 0
    num_single   = 0
    num_aromatic = 0
    
    for b in bonds:
        num_bonds += 1
        if b.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            num_single += 1
        if b.GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE:
            num_double += 1
        if b.GetBondType() == rdkit.Chem.rdchem.BondType.TRIPLE:
            num_triple += 1
        if b.GetBondType() == rdkit.Chem.rdchem.BondType.AROMATIC:
            num_aromatic += 1
    if num_bonds == 0:
        return [0, 0, 0, 0]
    else:
        return [num_single/num_bonds, num_double/num_bonds, num_triple/num_bonds, num_aromatic/num_bonds]


def count_conseq_double(mol):
    '''Return the number of consequtive double bonds in an entire molecule
       including rings 

    Examples 
    >>> count_conseq_double(Chem.MolFromSmiles('C1=CC=C=C=C1'))
    2
    >>> count_conseq_double(Chem.MolFromSmiles('C1=CC=CC=C1'))
    0
    >>> count_conseq_double(Chem.MolFromSmiles('C1=CC2=C(C=C1)C=C=C=C2'))
    2
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : Molecule for conseq. double bonds are to be counted 
    
    Returns:
    (int):  The integer number of coseq. double bonds 
    '''
    bonds = mol.GetBonds()    
    
    previous_BType    = None
    count_conseq_doub = 0
    for b in bonds:
        curr_BType = b.GetBondType()
        if previous_BType == curr_BType and curr_BType == rdkit.Chem.rdchem.BondType.DOUBLE:
            count_conseq_doub += 1
        previous_BType = curr_BType
    
    return count_conseq_doub


def get_rot_bonds_posn(mol):
    '''Return atom indices with Rotatable bonds 
    
    Examples:
    >>> get_rot_bonds_posn('CC1=CC=CC=C1')  # Toluene  (Rotatable Bonds At: CH3 & Benzene)
    ((0, 1),)
    >>> get_rot_bonds_posn('CCC1=CC=CC=C1') # (Rotatable Bonds At: CH3, CH3 & Benzene)
    ((0, 1), (1, 2))
    '''
    RotatableBond = Chem.MolFromSmarts('*-&!@*')
    rot = mol.GetSubstructMatches(RotatableBond)
    return rot


def get_bond_indeces(mol, rot):
    '''Get all the bond indices with Rotatable bonds atoms (generated from 'get_rot_bonds_posn')
    '''
    bonds_idx = []
    for i in range(len(rot)):
        bond = mol.GetBondBetweenAtoms(rot[i][0],rot[i][1])
        bonds_idx.append(bond.GetIdx())
    return bonds_idx


def obtain_rings(smi):
    '''Obtain a list of all rings present in SMILE string smi
    
    Examples:
    >>> obtain_rings('CCC1=CC=CC=C1')
    ['c1ccccc1']
    >>> obtain_rings('C1=CC=C(C=C1)C1=CC=CC=C1')
    ['c1ccccc1', 'c1ccccc1']
    >>> obtain_rings('C1=CC2=C(C=C1)C=CC=C2')
    (None, None)
    
    Parameters:
    smi (string) : SMILE string of a molecule 
    
    Returns
    (list)       : List if all rings in a SMILE string 
    '''
    mol = Chem.MolFromSmiles(smi)
    rot = get_rot_bonds_posn(mol) # Get rotatble bond positions
    
    if len(rot) == 0:
        return None, None
    
    bond_idx = get_bond_indeces(mol, rot)
    new_mol = Chem.FragmentOnBonds(mol, bond_idx, addDummies=False) 
    new_smile = Chem.MolToSmiles(new_mol)
    
    smile_split_list = new_smile.split(".") 
    rings = []
    for item in smile_split_list:
        if '1' in item:
            rings.append(item)
    return rings 


def size_ring_counter(ring_ls):
    '''Get the number of rings of sizes 3 to 20 and the number of consequtive double bonds in a ring
    
    Parameters:
    ring_ls (list)  : list of rings of a molecule 
    
    Returns
    (list)          : Of size 19 (1 for number of conseq. double bonds)
                                 (18 for number of rings between size 3 to 20)
    '''
    ring_counter = []
    
    if ring_ls == (None, None): # Presence of no rings, return 0s for the 19 feature
        return [0 for i in range(19)] 
    
    mol_ring_ls  = [Chem.MolFromSmiles(smi) for smi in ring_ls]
    
    # Cont number consequtive double bonds in ring 
    conseq_dbl_bnd_in_ring = 0
    for item in mol_ring_ls:
        conseq_dbl_bnd_in_ring += count_conseq_double(item)
    ring_counter.append(conseq_dbl_bnd_in_ring) # concatenate onto list ring_counter
    
    # Count the number of consequtive double bonds in rings 
    for i in range(3, 21):
        count = 0
        for mol_ring in mol_ring_ls:
            if mol_ring.GetNumAtoms() == i:
                count += 1
        ring_counter.append(count)
    return ring_counter
            


def get_mol_info(smi):                 
    ''' Calculate a set of 51 RdKit properties, collected from above helper functions. 
    
    Parameters:
    smi (string) : SMILE string of molecule 
    
    Returns:
    (list of float) : list of 51 calculated properties  
    '''
    mol = Chem.MolFromSmiles(smi)
        
    num_atoms   = mol.GetNumAtoms()       
    num_hydro   = Chem.AddHs(mol).GetNumAtoms() - num_atoms 
    num_carbon  = count_atoms(mol, 6)
    num_nitro   = count_atoms(mol, 7)
    num_sulphur = count_atoms(mol, 16)
    num_oxy     = count_atoms(mol, 8)
    num_clorine = count_atoms(mol, 17)
    num_bromine = count_atoms(mol, 35)
    num_florine = count_atoms(mol, 9)

    
    if num_carbon == 0: # Avoid division by zero error, set num_carbon to a very small value 
        num_carbon = 0.0001
    
    basic_props = [num_atoms/num_carbon, num_hydro/num_carbon, num_nitro/num_carbon, 
                     num_sulphur/num_carbon, num_oxy/num_carbon, num_clorine/num_carbon,
                     num_bromine/num_carbon, num_florine/num_carbon]
    
    to_caculate = ["RingCount", "HallKierAlpha", "BalabanJ", "NumAliphaticCarbocycles","NumAliphaticHeterocycles",
                   "NumAliphaticRings","NumAromaticCarbocycles","NumAromaticHeterocycles",
                   "NumAromaticRings","NumHAcceptors","NumHDonors","NumHeteroatoms",
                   "NumRadicalElectrons","NumSaturatedCarbocycles","NumSaturatedHeterocycles",
                   "NumSaturatedRings","NumValenceElectrons"]    

    # Calculate all propoerties listed in 'to_calculate'
    calc_props = OrderedDict(inspect.getmembers(Descriptors, inspect.isfunction))
    for key in list(calc_props.keys()):
        if key.startswith('_'):
            del calc_props[key]
            continue
        if len(to_caculate)!=0 and key not in to_caculate:
            del calc_props[key]
    features = [val(mol) for key,val in calc_props.items()] # List of properties 
    
    
    # Ratio of total number of  (single, double, triple, aromatic) bonds to the total number of bonds. 
    simple_bond_info = get_num_bond_types(mol) 
    
    # Obtain all rings in a molecule and calc. #of triple bonds in rings & #of rings in molecule 
    ring_ls = obtain_rings(smi)
    num_triple = 0      # num triple bonds in ring

    
    if len(ring_ls) > 0 and ring_ls != (None, None):
        for item in ring_ls:
            num_triple += item.count('#')
        simple_bond_info.append(len(ring_ls))     # append number of Rings in molecule 
    else:   simple_bond_info.append(0)            # no rotatable bonds

        
    simple_bond_info.append(num_triple)          # number of triple bonds in rings
                                                 # appended onto 'simple_bond_info'
                                              
                                    
    # Calculate the number of rings of size 3 to 20 & number of conseq. double bonds in rings 
    simple_bond_info = simple_bond_info + size_ring_counter(ring_ls)
    
    # Calculate the number of consequitve double bonds in entire molecule
    simple_bond_info.append(count_conseq_double(mol)) 
    
    return np.array(features + basic_props + simple_bond_info)
    

def get_chunks(arr, num_processors, ratio):
    """
    Get chunks based on a list 
    """
    chunks = []  # Collect arrays that will be sent to different processorr 
    counter = int(ratio)
    for i in range(num_processors):
        if i == 0:
            chunks.append(arr[0:counter])
        if i != 0 and i<num_processors-1:
            chunks.append(arr[counter-int(ratio): counter])
        if i == num_processors-1:
            chunks.append(arr[counter-int(ratio): ])
        counter += int(ratio)
    return chunks 


def get_mult_mol_info(smiles_list):
    ''' Collect results of 'get_mol_info' for multiple smiles (smiles_list)
    
    Parameters:
    smiles_list (list) : List of SMILE strings
    
    Returns:
    np.array : Concatenated array of results with shape (len(smiles_list), 51)
               51 is the number of RdKit properties calculated in  'get_mol_info'.
    '''
    concat_arr = []
    for smi in smiles_list:
        concat_arr.append(get_mol_info(smi))
    return np.array(concat_arr)
    

def get_mult_mol_info_parr(smiles_list, dataset_x):
    ''' Record calculated rdkit property results for each smile in smiles_list,
    and add record result in dictionary dataset_x.
    '''
    for smi in smiles_list:
        dataset_x['properties_rdkit'][smi] = get_mol_info(smi)
        
    
def create_parr_process(chunks):
    '''This function initiates parallel execution (based on the number of cpu cores)
    to calculate all the properties mentioned in 'get_mol_info()'
    
    Parameters:
    chunks (list)   : List of lists, contining smile strings. Each sub list is 
                      sent to a different process
    dataset_x (dict): Locked dictionary for recording results from different processes. 
                      Locking allows communication between different processes. 
                      
    Returns:
    None : All results are recorde in dictionary 'dataset_x'
    '''
    # Assign data to each process 
    process_collector = []
    collect_dictionaries = []
    
    for chunk in chunks:                # process initialization 
        dataset_x         = manager.dict(lock=True)
        smiles_map_props  = manager.dict(lock=True)

        dataset_x['properties_rdkit'] = smiles_map_props
        collect_dictionaries.append(dataset_x)
        
        process_collector.append(multiprocessing.Process(target=get_mult_mol_info_parr, args=(chunk, dataset_x,  )))

    for item in process_collector:      # initite all process 
        item.start()
    
    for item in process_collector:      # wait for all processes to finish
        item.join()   
    
    combined_dict = {}
    for i,item in enumerate(collect_dictionaries):
        combined_dict.update(item['properties_rdkit'])

    return combined_dict


def obtain_discr_encoding(molecules_here, disc_enc_type, max_molecules_len, num_processors, generation_index):
    '''Obtain features for showing to the discrimantor
    
    if disc_enc_type is 'smiles' or 'selfies', obtain a one-hto encoding 
    if disc_enc_type is 'properties_rdkit' obtain calculated rdkit properties 
    
    Parameters:
    molecules_here    (list)  : List of a string of molecules (as either SMILE or SELFIE)
    disc_enc_type     (string): 'selfie' or 'smile', indicating what alphabets to use
                                for obtaining one-hot encoding
    max_molecules_len (int)   : Length of the largest molecule
    
    Returns
    (np.array) : Concatenated list of properties or one-hot encodings
    '''
    if disc_enc_type == 'smiles':
        dataset_x = _to_onehot(molecules_here, disc_enc_type, max_molecules_len)
        
    elif disc_enc_type == 'selfies':
        dataset_x = _to_onehot(molecules_here, disc_enc_type, max_molecules_len)
        
    elif disc_enc_type == 'properties_rdkit':
        # Parallel generation method         
        molecules_here_unique = list(set(molecules_here))
        ratio            = len(molecules_here_unique) / num_processors              # number of smiles each process shall handle
        chunks           = get_chunks(molecules_here_unique, num_processors, ratio) 
        chunks           = [item for item in chunks if len(item) >= 1]

        results_dict = create_parr_process(chunks) 

        collect_data_x = [results_dict[smi] for smi in molecules_here]
        dataset_x = np.array(collect_data_x)                                        # Collect results from all processes 

    return dataset_x





