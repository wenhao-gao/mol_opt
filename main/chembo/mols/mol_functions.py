"""
Functions defined on Molecules.

A list of examples of how add objective functions.
(Some are from rdkit, but new ones should be user-definable.)

NOTES:
* List of RDkit mol descriptors:
  https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors

"""

import networkx as nx
from myrdkit import Chem
from rdkit_contrib.sascorer import calculateScore as calculateSAScore
from myrdkit import qed
from myrdkit import Descriptors

def get_objective_by_name(name):
    """Get a function computing molecular property.
    See individual functions below for descriptions.
    
    Arguments:
        name {str} -- one of "sascore", "logp", "qed"

    Returns:
        function: mol->float

    Raises:
        NotImplementedError -- function not implemented
    """
    if name == "sascore":
        return SAScore
    elif name == "logp":
        return LogP
    elif name == "qed":
        return QED
    elif name == "plogp":
        return PenalizedLogP
    else:
        raise NotImplementedError

def to_rdkit(mol):
    if isinstance(mol, list):
        rdkit_mol = mol[0].to_rdkit()
    else:
        rdkit_mol = mol.to_rdkit()
    return rdkit_mol

def to_graph(mol):
    if isinstance(mol, list):
        graph_mol = mol[0].to_graph('networkx')
    else:
        graph_mol = mol.to_graph('networkx')
    return graph_mol

def SAScore(mol):
    """ Synthetic accessibility score.
    Larger value means harder to synthesize.
    """
    rdkit_mol = to_rdkit(mol)
    return calculateSAScore(rdkit_mol)

def LogP(mol):
    """ Range of LogP between [0, 5] corresponds to drug-like mols """
    rdkit_mol = to_rdkit(mol)
    return Descriptors.MolLogP(rdkit_mol)

def QED(mol):
    """ Quantative estimation of drug-likeliness.
    `High` ranges - [0.9, 1.]

    """
    rdkit_mol = to_rdkit(mol)
    return qed(rdkit_mol)

def PenalizedLogP(mol):
    """ Penalized LogP score
    Implementation follows the official JT-VAE implementation:
    https://github.com/wengong-jin/icml18-jtnn/blob/5777b0599aa826ecda1b119e2f878518d4ad9b3f/bo/gen_latent.py
    """
    rdkit_mol = to_rdkit(mol)
    molgraph = to_graph(mol)
    logp = Descriptors.MolLogP(rdkit_mol)
    sa = calculateSAScore(rdkit_mol)
    cycle_list = nx.cycle_basis(molgraph)
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return logp - sa - cycle_length

def SMILES_len(mol):
    return len(mol.smiles)


