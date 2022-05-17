

import pytest

from rdkit import Chem
import numpy as np

from syn_dags.data import smiles_to_feats


TEST_MOL_STR = '[SiH2:1]1[cH:2][cH:3][cH:4][c:5]1-[c:6]1[cH:7][cH:8][c:9]([s:10]1)-[c:11]1[cH:12][cH:13][c:14]([cH:15][n:16]1)-[c:17]1[n:18][cH:19][n:20][cH:21][n:22]1'




@pytest.fixture()
def smiles_featurizer():
    atom_featurizer = smiles_to_feats.AtmFeaturizer(['C', 'N', 'O', 'S', 'Se', 'Si'])
    smiles_featurizer = smiles_to_feats.SmilesFeaturizer(atom_featurizer)
    return smiles_featurizer


def test_atom_featurizer():
    atom_featurizer = smiles_to_feats.AtmFeaturizer(['C', 'N', 'O', 'S', 'Se', 'Si'])
    test_mol = Chem.MolFromSmiles(TEST_MOL_STR)

    # not really part of this test but makes sure we are reading consistently
    atm_at_idx_9 = test_mol.GetAtomWithIdx(9)
    assert atm_at_idx_9.GetAtomMapNum() == 10, "atom indexing changing!"

    feats = atom_featurizer.atom_to_feat(atm_at_idx_9, test_mol, 9)

    np.testing.assert_array_almost_equal(feats.numpy(), np.array([ 0.,  0.,  0.,  1.,  0.,  0., 16.,  0.,  0.,  0.,  1.,  0.,  1.,  0.]))
    # S, 16 atomic number, acceptor/donor using rdkit def, hybridisation SP2, aromatic, no Hydrogens,


def test_accpetors():
    # got acceptors from Marvin
    acceptors = {15, 17, 19, 21}

    atom_featurizer = smiles_to_feats.AtmFeaturizer(['C', 'N', 'O', 'S', 'Se', 'Si'])
    test_mol = Chem.MolFromSmiles(TEST_MOL_STR)

    found_acceptors_rdkit, _ = atom_featurizer.get_acceptor_and_donor_ids(test_mol)
    assert found_acceptors_rdkit == acceptors


def test_smiles_featurizer__bond_features(smiles_featurizer):
    _, _, bond_features = smiles_featurizer.smi_to_feats(TEST_MOL_STR)

    # manually check first bond:
    feat = [1,0,0,0]
    np.testing.assert_array_almost_equal(bond_features[0,:].numpy(), np.array(feat))



def test_smiles_featurizer__bond_idx(smiles_featurizer):
    bonds = [(1,2), (1,5), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (6,10),
                 (9,11), (11,12), (12,13), (13,14), (14,15), (15,16), (11,16), (14,17),
                 (18,17), (17,22), (22,21), (21,20),(19,20), (18,19)]  # wrt atom mapping
    bonds = set([frozenset((x_0-1, x_1-1)) for x_0, x_1 in bonds])    # wrt indices

    _, bnds_calc, _ = smiles_featurizer.smi_to_feats(TEST_MOL_STR)

    for start, end in bnds_calc.numpy().T:
        bonds.remove(frozenset((start, end)))

    assert len(bonds) == 0, "more bonds in atom than expected"


def test_smiles_featurizer__node_feat_loc(smiles_featurizer):
    atm_feat, _, _ = smiles_featurizer.smi_to_feats(TEST_MOL_STR)
    np.testing.assert_array_almost_equal(atm_feat[9,:].numpy(), np.array([ 0.,  0.,  0.,  1.,  0.,  0., 16.,  0.,  0.,  0.,  1.,  0.,  1.,  0.]))



