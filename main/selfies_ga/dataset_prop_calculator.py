'''
Calculate logP, SAS-score & ring penalty of a given data set 


Pretty much dummy code for going throught the zinc data set

Standardized Values:
    x' = (x - x_mean) / (x_std)

@author: akshat
'''
import evolution_functions as evo
from rdkit import Chem
from SAS_calculator.sascorer import calculateScore


def calc_prop_RingP(unseen_smile_ls):
    '''Calculate Ring penalty for each molecule in unseen_smile_ls,
       results are recorded in locked dictionary props_collect 
    '''
    RingP_collect = []
    for i,smi in enumerate(unseen_smile_ls):
        mol, smi_canon, did_convert = evo.sanitize_smiles(smi)
        if did_convert:    
            cycle_list = mol.GetRingInfo().AtomRings() 
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([ len(j) for j in cycle_list ])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            RingP_collect.append(cycle_length)
            if i % 1000 == 0:
                print(i)
        else:
            raise Exception('Invalid smile encountered while atempting to calculate Ring penalty')
    return RingP_collect
    
    

dataset_smiles = evo.read_dataset_encoding('smiles')
#A = calc_prop_RingP(dataset_smiles)
#
#f = open('./zinc_RingP.txt', 'a+')
#f.writelines(["%s\n" % item  for item in A])
#f.close()



logP_collect = []
for i , smi in enumerate(dataset_smiles):
    
    
    mol = Chem.MolFromSmiles(smi)
    
    # Calculate logP values 
    logP_collect.append(evo.get_logP(mol))
    
    # RESULTS (logP):
    # max : 8.252100000000006
    # min : -6.0328
    # mean: 2.4729421499641497
    # std : 1.4157879815362406

    if i % 1000 == 0:
        print(i)

    # Calculate the SAS-score
#    SAS_collect.append(calculateScore(mol))
    
    
    # RESULTS (SAS):
    # max : 7.289282840617412
    # min : 1.1327382804544328
    # mean: 3.0470797085649894
    # std : 0.830643172314514
f = open('./zinc_logP.txt', 'a+')
f.writelines(["%s\n" % item  for item in logP_collect])
f.close()


    
    

    
    
    
    
    
    
    