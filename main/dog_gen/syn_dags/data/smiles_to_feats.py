
import typing
import functools
import itertools
import warnings
import collections
import os


from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.RDConfig import RDDataDir

import torch

from graph_neural_networks.sparse_pattern import graph_as_adj_list as grphs

from ..utils import settings

class AtmFeaturizer:
    """
    See Table 1 of Gilmer et al, https://arxiv.org/pdf/1704.01212.pdf
    """

    def __init__(self, atms: typing.List[str]):
        self.atms_to_idx = dict(zip(atms, range(len(atms))))
        self.number_atom_options = len(self.atms_to_idx)

        self.hyb_mapping = {Chem.rdchem.HybridizationType.SP:0 ,
                            Chem.rdchem.HybridizationType.SP2: 1,
                            Chem.rdchem.HybridizationType.SP3: 2}
        self.number_hyb_options = len(self.hyb_mapping)

        self.fdef_name = os.path.join(RDDataDir, 'BaseFeatures.fdef')
        self.feats_factory = ChemicalFeatures.BuildFeatureFactory(self.fdef_name)

    def atom_to_feat(self, atm: Chem.Atom, owning_mol: Chem.Mol, idx):
        # nb the func GetOwningMol could not be used for mol as would return different object each time
        this_atms_idx = atm.GetIdx()
        assert idx == this_atms_idx

        feat = torch.zeros(len(self), dtype=torch.float32)

        # One hot encoding of element
        try:
            feat[self.atms_to_idx[atm.GetSymbol()]] = 1.
        except KeyError as ex:
            warnings.warn(f"Ignoring the symbol {atm.GetSymbol()}")
        idx_up_to = self.number_atom_options

        # Atomic Number
        feat[idx_up_to] = float(atm.GetAtomicNum())
        idx_up_to += 1

        # Acceptor/donor
        acceptor_ids, donor_ids = self.get_acceptor_and_donor_ids(owning_mol)

        # Acceptor
        feat[idx_up_to] = float(this_atms_idx in acceptor_ids)
        idx_up_to += 1

        # Donor
        feat[idx_up_to] = float(this_atms_idx in donor_ids)
        idx_up_to += 1


        # Hybridization
        hyb_idx = self.hybridization(atm.GetHybridization())
        if hyb_idx is not None:
            feat[idx_up_to + hyb_idx] = 1.
        idx_up_to += self.number_hyb_options

        # Aromatic
        feat[idx_up_to] = float(atm.GetIsAromatic())
        idx_up_to += 1

        # Number of Hydrogens
        feat[idx_up_to] = float(atm.GetNumImplicitHs())
        idx_up_to += 1

        return feat

    @functools.lru_cache(maxsize=10)
    def get_acceptor_and_donor_ids(self, molecule: Chem.Mol):
        # This approach for getting the acceptors/donors has been taken from:
        # https://github.com/priba/nmp_qc/blob/master/GraphReader/graph_reader.py
        feats = self.feats_factory.GetFeaturesForMol(molecule)

        acceptor_ids = set(itertools.chain(*[x.GetAtomIds() for x in feats if x.GetFamily() == 'Acceptor']))
        donor_ids = set(itertools.chain(*[x.GetAtomIds() for x in feats if x.GetFamily() == 'Donor']))
        return acceptor_ids, donor_ids


    def hybridization(self, hybridization_type):
        return self.hyb_mapping.get(hybridization_type, None)

    def __len__(self):
        return self.number_atom_options + 5 + self.number_hyb_options


class BondFeaturizer:
    """
    One hot 
    """
    def __init__(self):
        self.bond_type_to_oh_loc = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }

    def bond_to_feat(self, bnd: Chem.Bond):
        bond_indices = torch.tensor([bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()])
        feat = torch.zeros(len(self.bond_type_to_oh_loc), dtype=torch.float32)
        feat[self.bond_type_to_oh_loc[bnd.GetBondType()]] = 1.
        return bond_indices, feat


class SmilesFeaturizer:
    def __init__(self, atm_featurizer: AtmFeaturizer):
        self.atm_featurizer = atm_featurizer
        self.bond_featurizer = BondFeaturizer()

    def smi_to_feats(self, smi: str):
        mol = Chem.MolFromSmiles(smi)

        # Get atoms features
        atm_feats = torch.stack([self.atm_featurizer.atom_to_feat(atm, mol, i) for i, atm in enumerate(mol.GetAtoms())])

        # Get edges and their features
        bonds, bond_features = zip(*[self.bond_featurizer.bond_to_feat(bnd) for bnd in mol.GetBonds()])
        bonds = torch.stack(bonds, dim=1)
        bond_features = torch.stack(bond_features)

        return atm_feats, bonds, bond_features




class SmilesToGraphAsAdjListFeaturizer:
    def __init__(self, atm_featurizer: AtmFeaturizer):
        self.atm_featurizer = atm_featurizer
        self.bond_featurizer = BondFeaturizer()

    def smi_to_feats(self, smi: str):
        bond_names = ["single", "double", "triple", "aromatic"]

        mol = Chem.MolFromSmiles(smi)

        # Get atoms features
        atm_feats = torch.stack([self.atm_featurizer.atom_to_feat(atm, mol, i) for i, atm in enumerate(mol.GetAtoms())])

        # Get edges and their features
        edge_to_bond_type = {k: [] for k in bond_names}
        for bonds, bond_features in [self.bond_featurizer.bond_to_feat(bnd) for bnd in mol.GetBonds()]:
            bond_type = bond_names[torch.argmax(bond_features)]
            edge_to_bond_type[bond_type].extend([bonds, bonds[[1,0]]])
            # ^ nb also add reverse to signify that it is an undirected edge.
        edge_to_bond_type = {k:
                                 (torch.stack(v, dim=1) if len(v) else
                                  torch.tensor([[],[]], device=str(atm_feats.device), dtype=settings.TORCH_INT))
                             for k,v in edge_to_bond_type.items()}
        node_to_graph_id = torch.zeros(atm_feats.shape[0], device=str(atm_feats.device), dtype=settings.TORCH_INT)
        return grphs.DirectedGraphAsAdjList(atm_feats, edge_to_bond_type, node_to_graph_id)


DEFAULT_SMILES_FEATURIZER = SmilesToGraphAsAdjListFeaturizer(AtmFeaturizer(
                    ['Ag', 'Al', 'Ar', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Br', 'C',
                    'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Eu', 'F',
                    'Fe', 'Ga', 'Ge', 'H', 'He', 'Hf', 'Hg', 'I', 'In', 'Ir', 'K', 'La',
                    'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nd', 'Ni', 'O', 'Os', 'P', 'Pb',
                    'Pd', 'Pr', 'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se',
                    'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Te', 'Ti', 'Tl', 'V', 'W', 'Xe', 'Y',
                    'Yb', 'Zn', 'Zr']
))

