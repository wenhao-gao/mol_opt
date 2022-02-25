"""
Molecular function callers.

A harness for calling functions defined over Molecules.
Makes use of the mols/mol_functions.py
"""

from argparse import Namespace
from copy import deepcopy
import numpy as np
from time import sleep

from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.exd.exd_core import EVAL_ERROR_CODE
from dragonfly.utils.reporters import get_reporter
from dragonfly.exd.domains import CartesianProductDomain

# Local imports
from mols.mol_domains import MolDomain


def get_cp_func_caller_args(domain_config):
    index_ordering = [0]
    kernel_ordering = [""]  # "" or None for not having a kernel associated with func caller, will use `chemist_args`
    name_ordering = ["molecule"]
    dim_ordering = [1]
    raw_name_ordering = ["molecule"]

    orderings = Namespace(index_ordering=index_ordering,
                          kernel_ordering=kernel_ordering,
                          dim_ordering=dim_ordering,
                          name_ordering=name_ordering,
                          raw_name_ordering=raw_name_ordering)
    list_of_domains = [MolDomain(**domain_config)]

    # Create a namespace with additional information
    domain_info = Namespace()
    domain_info.config_orderings = orderings
    # domain_info.constraints = None

    # Create a cartesian product domain
    cp_domain = CartesianProductDomain(list_of_domains, domain_info)

    fidel_space, fidel_to_opt, fidel_space_orderings = None, None, None

    ret = {
        'domain': cp_domain,
        'domain_orderings': orderings,
        'fidel_space': fidel_space,
        'fidel_to_opt': fidel_to_opt,
        'fidel_space_orderings': fidel_space_orderings
    }
    return ret


class MolFunctionCaller(CPFunctionCaller):
    """ Function Caller for Mol evaluations. """
    def __init__(self, objective, domain_config, descr='', reporter='silent'):
        constructor_args = get_cp_func_caller_args(domain_config)
        super(MolFunctionCaller, self).__init__(objective, descr=descr, **constructor_args)
        self.reporter = get_reporter(reporter)

    @classmethod
    def is_mf(cls):
        """ Returns True if Multi-fidelity. """
        return False


if __name__ == "__main__":
    from mols.molecule import Molecule
    mol = Molecule("C=C1NC(N(C)C)=NC12CCN(CC(C)c1ccccc1)CC2")
    print("---")
    domain_config = {'data_source': 'chembl',
                     'constraint_checker': 'organic',
                     'sampling_seed': 42}
    cp_domain = get_cp_func_caller_args(domain_config)['domain']
    print(cp_domain.is_a_member(mol))
    print(cp_domain.is_a_member([mol]))
    print(cp_domain.is_a_member([None]))


