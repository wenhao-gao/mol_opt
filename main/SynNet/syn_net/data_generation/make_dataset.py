"""
This file generates synthetic tree data in a sequential fashion.
"""
import dill as pickle
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from syn_net.utils.data_utils import SyntheticTreeSet
from syn_net.utils.prep_utils import synthetic_tree_generator



if __name__ == '__main__':
    path_reaction_file = '/home/whgao/shared/Data/scGen/reactions_pis.pickle.gz'
    path_to_building_blocks = '/home/whgao/shared/Data/scGen/enamine_building_blocks_nochiral_matched.csv.gz'

    np.random.seed(6)

    building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
    with gzip.open(path_reaction_file, 'rb') as f:
        rxns = pickle.load(f)

    Trial = 5
    num_finish = 0
    num_error = 0
    num_unfinish = 0

    trees = []
    for _ in tqdm(range(Trial)):
        tree, action = synthetic_tree_generator(building_blocks, rxns, max_step=15)
        if action == 3:
            trees.append(tree)
            num_finish += 1
        elif action == -1:
            num_error += 1
        else:
            num_unfinish += 1

    print('Total trial: ', Trial)
    print('num of finished trees: ', num_finish)
    print('num of unfinished tree: ', num_unfinish)
    print('num of error processes: ', num_error)

    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save('st_data.json.gz')

    # data_file = gzip.open('st_data.pickle.gz', 'wb')
    # pickle.dump(trees, data_file)
    # data_file.close()
