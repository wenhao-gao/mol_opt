"""
This file contains a function for search available building blocks 
for a matching reaction template. Prepared for multiprocessing.
"""
import pandas as pd

path_to_building_blocks = '/home/whgao/scGen/synth_net/data/enamine_us.csv.gz'
building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
print('Finish reading the building blocks list!')

def func(rxn_):
    rxn_.set_available_reactants(building_blocks)
    return rxn_
