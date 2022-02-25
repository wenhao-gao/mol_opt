"""
Synthesis validators.
To be used internally with Explorers,
or a separate sanity check.

TODO:
* implement backward predictors
  or find any likelihood evaliation tool
  (e.g. rdkit may have something helpful).
"""

from mols.molecule import Molecule
from mols.mol_functions import get_objective_by_name


def compute_min_sa_score(mol):
    """ Compute sas scores along the synthesis path of molecule. """
    sa_score = get_objective_by_name("sascore")
    def get_min_score(syn):
        res = float('inf')
        for mol, syn_graph in syn.items():
            # if mol.begin_flag:
            if isinstance(syn_graph, str):
                return sa_score(Molecule(mol))
            res = min(res, get_min_score(syn_graph))
        return res
    synthesis_path = mol.get_synthesis_path()
    if isinstance(synthesis_path, dict):
        min_sa_score = get_min_score(synthesis_path)
    else:
        min_sa_score = sa_score(Molecule(synthesis_path))
    return min_sa_score

def check_validity(mol):
    """ 
    If convertation to rdkit.Mol fails,
    the molecule is not valid.
    """
    try:
        mol.to_rdkit()
        return True
    except:
        return False
