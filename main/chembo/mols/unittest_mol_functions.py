"""
Unit tests for molecular propertiy computation
"""

from dragonfly.utils.base_test_class import BaseTestClass, execute_tests
from mols.mol_functions import get_objective_by_name
from mols.molecule import Molecule

class ExplorerTestCase(BaseTestClass):
    def setUp(self):
        self.mol = Molecule("CC")

    def test_sas(self):
        sas = get_objective_by_name("sascore")
        sas(self.mol)

    def test_qed(self):
        qed = get_objective_by_name("qed")
        qed(self.mol)

    def test_plogp(self):
        plogp = get_objective_by_name("plogp")
        print(plogp(self.mol))

if __name__ == "__main__":
    execute_tests()