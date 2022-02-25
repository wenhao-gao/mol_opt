"""
Unit tests for explorer.py

TODO:
* update imports from rdkit_contrib to import from mols
"""

import numpy as np
from time import time
from myrdkit import Chem

# Local imports
from explorer.mol_explorer import RandomExplorer
from datasets.loaders import get_chembl_prop
from mols.molecule import Molecule
from rdkit_contrib.sascorer import calculateScore as calculateSAScore
from dragonfly.utils.base_test_class import BaseTestClass, execute_tests


class ExplorerTestCase(BaseTestClass):
    def __init__(self, *args, **kwargs):
        super(ExplorerTestCase, self).__init__(*args, **kwargs)

    def _test_len(self):
        dummy_func = lambda mol: len(mol.smiles)
        test_pool = ["CC", "O=C=O", "C#N", "CCN(CC)CC", "CC(=O)O", "C1CCCCC1", "c1ccccc1"]
        test_pool = [Molecule(smiles) for smiles in test_pool]
        exp = RandomExplorer(dummy_func, initial_pool=test_pool)
        print("Starting len of SMILES optimization")
        exp.run(2)

        #check
        print(exp.pool)

    def _test_sas(self):
        sas_func = lambda mol: calculateSAScore(Chem.MolFromSmiles(mol.smiles))
        print(sas_func(Molecule("CC")))
        test_pool = ["CC", "O=C=O", "C#N", "CCN(CC)CC", "CC(=O)O", "C1CCCCC1", "c1ccccc1"]
        test_pool = [Molecule(smiles) for smiles in test_pool]
        exp = RandomExplorer(sas_func, initial_pool=test_pool)
        print("Starting SA score optimization")
        t0 = time()
        exp.run(10)

        #check
        print("Completed SA score optimization, time elapsed: %.3fs" % (time()-t0))
        print(exp.pool)
        top = exp.get_best(1)[0]
        print(top.get_synthesis_path())

    def test_chembl(self):
        """
        Problem with fixed-prop testing:
        Almost all of the results (<10% for init_pool of 50) seem to be outside of the database,
        and even less for smaller pool. Hence cannot get its score for testing;
        setting them to zero leads to slow exploration.
        """
        pool_all, dd = get_chembl_prop()

        # loading with mol conversions takes 8 minutes
        # pool_all = [Molecule(smiles, conv_enabled=True) for smiles in tqdm(pool_all[:10000])]
        pool_all = [Molecule(smiles, conv_enabled=False) for smiles in pool_all]
        start_pool = list(np.random.choice(pool_all, size=100, replace=False))

        def print_props(pool):
            props = [dd[mol.smiles] for mol in pool]
            print("Props of pool", len(pool), np.min(props), np.mean(props), np.max(props))
        print_props(pool_all)
        print_props(start_pool)

        func = lambda mol: dd[mol.smiles]
        exp = RandomExplorer(lambda mol_list: func(mol_list[0]), initial_pool=start_pool)

        print("Starting ChEMBL score 1 optimization")
        t0 = time()
        exp.run(30)
        print("Completed ChEMBL score 1 optimization, time elapsed: %.3fs" % (time()-t0))

        # print(exp.pool)
        top = exp.get_best(1)[0]
        print(top.get_synthesis_path())

        print("Best achieved score: %.3f" % func(top))
        props = [dd[mol.smiles] for mol in pool_all]
        print("Best possible score: %.3f" % np.max(props))


if __name__ == '__main__':
    execute_tests()
