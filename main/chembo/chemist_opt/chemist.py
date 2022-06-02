"""
Module for BO with graph kernel and synthesizeable exploration.

Available Explorers: 'rand_explorer', ...

NOTE:
* Currently unneded options are filtered from options list;
  however, they may be needed when more complicated CP domains appear,
  so better filtering regulation might be needed later
  (see reset_default_options method in Chemist)
"""



import logging
from argparse import Namespace

from dragonfly.opt.gp_bandit import get_all_cp_gp_bandit_args
from dragonfly.utils.reporters import get_reporter
from dragonfly.utils.option_handler import load_options
from dragonfly.exd.worker_manager import RealWorkerManager, SyntheticWorkerManager
from dragonfly.exd.exd_utils import get_cp_domain_initial_qinfos

from chemist_opt.gp_bandit import CPGPBandit, get_cp_domain_initial_qinfos
from chemist_opt.mol_function_caller import MolFunctionCaller
from dist.ot_dist_computer import OTChemDistanceComputer
from mols.mol_kernels import MOL_DISTANCE_KERNEL_TYPES, MOL_SUM_KERNEL_TYPES
from mols.mol_gp import get_default_kernel_type


class Chemist:
    def __init__(self, objective_func,  
                 domain_config, chemist_args=None, 
                 worker_manager='default', reporter='default', 
                 is_mf=False, mf_strategy=None):

        self.reporter = get_reporter(reporter)
        self.worker_manager = get_worker_manager(worker_manager)
        if domain_config is None:
            domain_config = {}
        self.func_caller = MolFunctionCaller(objective_func,
                                             domain_config=domain_config,
                                             reporter=self.reporter)
        self.is_mf = is_mf
        self.mf_strategy = mf_strategy
        # kernel and explorer-related settings:
        chemist_args = self.fill_with_default_chemist_args(chemist_args)
        self.domain_dist_computers = self.get_dist_computers(chemist_args)
        self.options = self.prepare_chemist_options(chemist_args, domain_config)

    def fill_with_default_chemist_args(self, chemist_args):
        """ Updates options that may not be set by user """
        if chemist_args is None:
            chemist_args = {}
        default_args = {'acq_opt_method': 'rand_explorer',
                        'init_capital': 'default',
                        'dom_mol_kernel_type': 'default'}
        default_args.update(chemist_args)
        return default_args

    def get_dist_computers(self, chemist_args):
        domain_dist_computers = []
        for domain, kernel_type in zip(self.func_caller.domain.list_of_domains,
                                       self.func_caller.domain_orderings.kernel_ordering):
            domain_type = domain.get_type()
            if domain_type == "molecule":
                # first check function caller kernel, then `chemist_args`, and finally default kernel
                if kernel_type is None or kernel_type == '':
                    kernel_type = chemist_args["dom_mol_kernel_type"]
                if kernel_type == "default":
                    kernel_type = get_default_kernel_type(domain_type)
                if kernel_type in MOL_DISTANCE_KERNEL_TYPES or kernel_type in MOL_SUM_KERNEL_TYPES:
                    computer = OTChemDistanceComputer()
                    domain_dist_computers.append(computer)
                else:
                    domain_dist_computers.append(None)
            else:
                raise NotImplementedError("Distance computers not implemented for other domains.")
        logging.info(f"domain_dist_computers: {domain_dist_computers}")
        return domain_dist_computers

    def reset_default_options(self, list_of_options, chemist_args):
        """ Reset entries in list with entries in kwargs
            if name matches. Hence non-matching entries will be ignored.
        """
        new_list_of_options = []
        added_names = []
        for d in list_of_options:
            if d['name'] in chemist_args:
                d['default'] = chemist_args[d['name']]
                new_list_of_options.append(d)
                added_names.append(d['name'])
            # TODO: regexp for filtering unneeded options
            elif 'dom_int' in d['name']:
                continue
            elif 'dom_disc' in d['name']:
                continue
            elif 'dom_int' in d['name']:
                continue
            elif 'dom_euc' in d['name']:
                continue
            elif 'dom_nn' in d['name']:
                continue
            elif 'otmann' in d['name']:
                continue
            elif 'nn_report' in d['name']:
                continue
            else:
                # add other names that are in list_of_options
                # but not in chemist_args
                new_list_of_options.append(d)
                added_names.append(d['name'])
        # add some options that are in chemist_args
        # but not in list_of_options
        for name, value in chemist_args.items():
            if name not in added_names:
                new_list_of_options += [{'name': name, 'default': value}]
        return new_list_of_options

    def prepare_chemist_options(self, chemist_args, domain_config):
        """ Resets default gp_bandit options with chemist arguments """
        dflt_list_of_options = get_all_cp_gp_bandit_args()
        list_of_options = self.reset_default_options(dflt_list_of_options, chemist_args)

        for name, value in domain_config.items():
            list_of_options += [{'name': name, 'default': value}]
        options = load_options(list_of_options, reporter=self.reporter)
 
        if self.mf_strategy is not None:
            options.mf_strategy = self.mf_strategy
        if isinstance(self.worker_manager, RealWorkerManager):
            options.capital_type = 'realtime'
        elif isinstance(self.worker_manager, SyntheticWorkerManager):
            options.capital_type = 'return_value'
        options.get_initial_qinfos = \
            lambda num: get_cp_domain_initial_qinfos(self.func_caller.domain, num)
        return options

    def run(self, max_capital):
        """ Main Chemist method

        Returns:
            opt_val, opt_point, history
        """

        # create optimiser and return
        optimiser = CPGPBandit(
            self.func_caller,
            self.worker_manager,
            is_mf=self.is_mf,
            options=self.options,
            reporter=self.reporter,
            domain_dist_computers=self.domain_dist_computers, 
        ) ### is sub-class of **BlackboxOptimiser**, instead of multi-objective optimizer 
        print('create optimizer, before optimization')
        # exit()  ### Q: initial pool is called before or after exit?  after. 
        return optimiser.optimise(max_capital) ### use blackbox instead of multiobjective


    # def get_raw_domain_point_from_processed(self, opt_point):
    #     return self.func_caller.get_raw_domain_point_from_processed(opt_point)


def get_worker_manager(worker_manager):
    """TODO
    Arguments:
        worker_manager {WorkerManager or str} -- [description]
    """
    return worker_manager

