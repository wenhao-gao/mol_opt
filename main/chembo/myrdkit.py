"""
A fix for rdkit broken imports:
Import this script in any script to run.
"""

import rdkit
from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem import rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdMolDescriptors import CalcMolFormula, CalcExactMolWt
from rdkit.Chem import Draw
from rdkit.Chem import MolSurf, Crippen
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import rdMolDescriptors
try:
    from rdkit.six.moves import cPickle
except:
    from six.moves import cPickle
from rdkit.six import iteritems
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import rdkit.Chem.rdChemReactions as rdRxns
from rdkit import RDLogger
