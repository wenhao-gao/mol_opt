import gzip
import numpy as np
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors, DataStructs


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit Mol object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    elif isinstance(smiles_or_mol, Chem.rdchem.Mol):
        return smiles_or_mol
    else:
        return None


class SillyWalks:
    """Adapted from https://github.com/PatWalters/silly_walks"""
    def __init__(self, ref_mols, n_jobs=1):
        self.count_dict = defaultdict(int)
        self._n_jobs = n_jobs
        with Pool(n_jobs) as pool:
            bit_counts = [bc for bc in pool.imap(self.count_bits, ref_mols) if bc is not None]
        for count_dict in bit_counts:
            for k, v in count_dict.items():
                self.count_dict[k] += v
            
    @staticmethod
    def count_bits(mol):
        count_dict = {}
        mol = get_mol(mol)
        if mol is not None:
            fp = Chem.GetMorganFingerprint(mol, 2)
            for k, v in fp.GetNonzeroElements().items():
                count_dict[k] = v
        return count_dict
        
    def score(self, mol):
        mol = get_mol(mol)
        if mol is not None:
            bi = {}
            fp = Chem.GetMorganFingerprint(mol, 2, bitInfo=bi)
            on_bits = fp.GetNonzeroElements().keys()
            silly_bits = [bit for bit in on_bits if self.count_dict[bit] == 0]
            score = len(silly_bits) / len(on_bits)
            return score, silly_bits, bi


class ChemistryFilters():
    def __init__(self, ref_dataset, std_thresh=4, denovobit_thresh=0.1, diverse_thresh=0.35, n_jobs=1):
        self.n_jobs = n_jobs
        self.std_thresh = std_thresh
        self.denovobit_thresh = denovobit_thresh
        self.diverse_thresh = diverse_thresh
        # Load dataset
        if ref_dataset.endswith('.gz'):
            with gzip.open(ref_dataset, 'rb') as f:
                ref_smiles = f.read().splitlines()
        else:
            with open(ref_dataset, 'r') as f:
                ref_smiles = f.read().splitlines()
        # Compute pre-statistics
        self.ref_SW = SillyWalks(ref_mols=ref_smiles, n_jobs=self.n_jobs)
        self.ref_MW = self._compute_mean_std(self._mols2prop(ref_smiles, 'MolWt'))
        self.ref_LogP = self._compute_mean_std(self._mols2prop(ref_smiles, 'MolLogP'))
        self._property_filters = [
            lambda x: (self.ref_MW[0]-self.std_thresh*self.ref_MW[1]) <= self.MolWt(x) <= (self.ref_MW[0]+self.std_thresh*self.ref_MW[1]),
            lambda x: (self.ref_LogP[0]-self.std_thresh*self.ref_LogP[1]) <= self.MolLogP(x) <= (self.ref_LogP[0]+self.std_thresh*self.ref_LogP[1]),
            lambda x: self.ref_SW.score(x)[0] <= self.denovobit_thresh
        ]

    @staticmethod
    def MolWt(x):
        mol = get_mol(x)
        if mol: 
            return Descriptors.MolWt(mol)

    @staticmethod
    def MolLogP(x):
        mol = get_mol(x)
        if mol:
            return Descriptors.MolLogP(mol)

    def _mols2prop(self, mols, prop):
        tfunc = getattr(self, prop)
        with Pool(self.n_jobs) as pool:
            r = [x for x in pool.imap(tfunc, mols) if x is not None]
        return r

    @staticmethod
    def _compute_mean_std(x):
        return np.mean(x), np.std(x)

    def passes_property_filters(self, mol):
        mol = get_mol(mol)
        if (mol is not None) and all([f(mol) for f in self._property_filters]):
            return True
        else:
            return False

    def top_n(self, mols, n=10, property_filters=False, diverse_filters=False, v=False):
        """
        Extract the top-n molecules according to input order and filters
        :return: Index of Top-n
        """
        failed_record = {'Property': 0, 'Diversity': 0, 'Total': 0}
        top_n_fps = []
        top_n = []
        i = 0
        while (len(top_n) < n) and (i < len(mols)):
            failed = False
            mol = get_mol(mols[i])
            if not mol:
                failed_record['Total'] += 1  
                i += 1
                continue

            if property_filters:
                if not self.passes_property_filters(mol):
                    failed_record['Property'] += 1
                    if not failed:
                        failed_record['Total'] += 1
                        failed = True
                
            if diverse_filters:
                fp = Chem.GetMorganFingerprint(mol, 2)
                if len(top_n_fps) > 0:
                    if not all([DataStructs.TanimotoSimilarity(fp, div_fp) <= self.diverse_thresh for div_fp in top_n_fps]):
                        failed_record['Diverse'] += 1
                        if not failed:
                            failed_record['Total'] += 1
                            failed = True
                        
            if not failed:
                top_n.append(i)
                if diverse_filters: top_n_fps.append(fp)

            i += 1
            
        if v:
            return top_n, failed_record
        else:    
            return top_n 