
import typing
import pickle
import json
import random

import tqdm
import numpy as np
import uuid

from ..data import synthesis_trees

EXP_UUID = None


def load_tuple_trees(path_to_trees: str,
                     rng: np.random.RandomState):
    """
    :param path_to_trees: This will point to a pickle file containing a list of tuples. First element of tuple is depth,
    second is the tuple tree.
    """
    with open(path_to_trees, 'rb') as fo:
        data = pickle.load(fo)

    def create_tuple_trees(elem):
        _, tuple_tree = elem
        return tuple_tree
    syn_trees = [create_tuple_trees(el) for el in tqdm.tqdm(data, desc="Loading in tuple trees")]
    return syn_trees


def load_reactant_vocab(path_to_json: str) -> typing.List[str]:
    with open(path_to_json, 'r') as fo:
        d = json.load(fo)
    return sorted(list(d.keys()), key=lambda x: d[x])
