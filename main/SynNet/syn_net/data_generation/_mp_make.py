"""
This file contains a function to generate a single synthetic tree, prepared for
multiprocessing.
"""
import pandas as pd
import numpy as np
# import dill as pickle
# import gzip

from syn_net.data_generation.make_dataset import synthetic_tree_generator
from syn_net.utils.data_utils import ReactionSet


path_reaction_file = '/pool001/whgao/data/synth_net/st_pis/reactions_pis.json.gz'
path_to_building_blocks = '/pool001/whgao/data/synth_net/st_pis/enamine_us_matched.csv.gz'

building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
r_set = ReactionSet()
r_set.load(path_reaction_file)
rxns = r_set.rxns
# with gzip.open(path_reaction_file, 'rb') as f:
#     rxns = pickle.load(f)

print('Finish reading the templates and building blocks list!')

def func(_):
    np.random.seed(_)
    tree, action = synthetic_tree_generator(building_blocks, rxns, max_step=15)
    return tree, action
