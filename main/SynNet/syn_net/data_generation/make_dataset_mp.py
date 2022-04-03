"""
This file generates synthetic tree data in a multi-thread fashion.

Usage:
    python make_dataset_mp.py
"""
import numpy as np
import multiprocessing as mp
from time import time

from syn_net.utils.data_utils import SyntheticTreeSet
import syn_net.data_generation._mp_make as make


if __name__ == '__main__':

    pool = mp.Pool(processes=100)

    NUM_TREES = 600000

    t = time()
    results = pool.map(make.func, np.arange(NUM_TREES).tolist())
    print('Time: ', time() - t, 's')

    trees = [r[0] for r in results if r[1] == 3]
    actions = [r[1] for r in results]

    num_finish = actions.count(3)
    num_error = actions.count(-1)
    num_unfinish = NUM_TREES - num_finish - num_error

    print('Total trial: ', NUM_TREES)
    print('num of finished trees: ', num_finish)
    print('num of unfinished tree: ', num_unfinish)
    print('num of error processes: ', num_error)

    tree_set = SyntheticTreeSet(trees)
    tree_set.save('/pool001/whgao/data/synth_net/st_pis/st_data.json.gz')
