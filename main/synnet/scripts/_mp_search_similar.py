"""
This function is used to identify the most similar molecule in the training set
to a given molecular fingerprint.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
from syn_net.utils.data_utils import *


data_path = '/pool001/whgao/data/synth_net/st_hb/st_train.json.gz'
st_set = SyntheticTreeSet()
st_set.load(data_path)
data = st_set.sts
data_train = [t.root.smiles for t in data]
fps_train = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=1024) for smi in data_train]


def func(fp):
    """
    Finds the most similar molecule in the training set to the input molecule
    using the Tanimoto similarity.

    Args:
        fp (np.ndarray): Morgan fingerprint to find similars to in the training set.

    Returns:
        np.float: The maximum similarity found to the training set fingerprints.
        np.ndarray: Fingerprint of the most similar training set molecule.
    """
    dists = np.array([DataStructs.FingerprintSimilarity(fp, fp_, metric=DataStructs.TanimotoSimilarity) for fp_ in fps_train])
    return dists.max(), dists.argmax()
