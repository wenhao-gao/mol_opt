""" Code for molecular fingerprints """

import functools

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

standard_fingerprint = functools.partial(
    rdMolDescriptors.GetMorganFingerprintAsBitVect, radius=2, nBits=1024
)


def _fp_to_array(fp):
    fp_arr = np.zeros((1,), dtype=np.int8)
    ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def smiles_to_fp_array(smiles: str, fingerprint_func: callable = None) -> np.array:
    """Convert individual SMILES into a 1D fingerprint array"""
    if fingerprint_func is None:
        fingerprint_func = standard_fingerprint
    mol = Chem.MolFromSmiles(smiles)
    fp = fingerprint_func(mol)
    return _fp_to_array(fp).flatten()
