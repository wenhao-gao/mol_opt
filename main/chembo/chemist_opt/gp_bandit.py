"""
A mirror of gp_bandit.py of Dragonfly:
imports class, resets a method and inherits further

This is an example of how to make this work:
>> sneaky_func = lambda: print("Hi!")
>> gpb_acquisitions.asy_ei.__code__ = sneaky_func.__code__
>> gpb_acquisitions.asy.ei()

TODO:
* anc_data has domain: use it for looping while the result is not in domain

NOTE:
* adding a new explorer, add it to GPBandit._opt_method_optimise_initalise
"""

from argparse import Namespace
import numpy as np
import logging

from dragonfly.opt.gp_bandit import GPBandit as GPBandit_
from dragonfly.opt.gp_bandit import CPGPBandit
import dragonfly.opt.gpb_acquisitions as gpb_acquisitions

# for sampling initial points
from dragonfly.utils.general_utils import block_augment_array
from dragonfly.utils.general_utils import transpose_list_of_lists

from mols.mol_gp import cartesian_product_gp_args, MolCPGPFitter
from explorer.mol_explorer import RandomExplorer
from mols.mol_domains import sample_mols_from_cartesian_domain


def mol_maximise_acquisition(acq_fn, anc_data, *args, **kwargs):
    """ returns optimal point """
    import logging
    acq_opt_method = anc_data.acq_opt_method

    if anc_data.domain.get_type() == 'euclidean':
        if acq_opt_method in ['rand']:
            acquisition = acq_fn
        else:
            # these methods cannot handle vectorised functions.
            acquisition = lambda x: acq_fn(x.reshape((1, -1)))
    elif anc_data.domain.get_type() == 'cartesian_product':
        # these methods cannot handle vectorised functions.
        acquisition = lambda x: acq_fn([x])
    else:
        raise NotImplementedError("Choose vectorization option for acquisition.")

    if acq_opt_method == "rand_explorer":
        explorer = anc_data.acq_optimizer
        explorer.reset_params(acquisition, anc_data.capital_type)
        received_valid_point = False
        while not received_valid_point:
            logging.info(f'Running explorer for {anc_data.max_evals} steps')
            top_value, top_point, history = explorer.run(anc_data.max_evals)
            received_valid_point = anc_data.domain.is_a_member([top_point])  # domain is CP domain
        logging.info(f"Returning explorer's result: {top_point}")
        return [top_point]
    else:
        raise NotImplementedError("Acq opt method {} not implemented.".format(acq_opt_method))


gpb_acquisitions.maximise_acquisition.__code__ = mol_maximise_acquisition.__code__

###############################################################################
# now we have to make GPBandit use this `poisoned` module:                    #
# one option is to override it in child class                                 #
###############################################################################

# Sampling initial data from the domain ----------------------------------------------
def get_cp_domain_initial_qinfos(domain, num_samples, dom_mol_sample_type='rand'):
    """
    Get initial qinfos in Cartesian product domain.
    The difference to the original function is in addition 
    of a sampler to handle MolDomain sampling.
    """
    initial_pool = sample_mols_from_cartesian_domain(domain, num_samples)
    individual_domain_samples = [initial_pool]
    ret_dom_pts = transpose_list_of_lists(individual_domain_samples)
    ret_dom_pts = ret_dom_pts[:num_samples]
    return [Namespace(point=x) for x in ret_dom_pts]


class GPBandit(GPBandit_):
    def say_hi(self):
        # testing function to see if the tricks worked
        print("Hi")

    def _determine_next_query(self):
        """ Determine the next point for evaluation. """
        curr_acq = self._get_next_acq()
        anc_data = self._get_ancillary_data_for_acquisition(curr_acq)
        anc_data.capital_type = self.capital_type
        anc_data.acq_optimizer = self.acq_optimizer
        select_pt_func = getattr(gpb_acquisitions.asy, curr_acq)  # <---- here
        qinfo = Namespace(curr_acq=curr_acq,
                          hp_tune_method=self.gp_processor.hp_tune_method)
        # this line would call mol_maximise_acquisition, where self.gp.eval will be used as (part of) acq_fn
        next_eval_point = select_pt_func(self.gp, anc_data)
        qinfo.point = next_eval_point
        return qinfo

    def _set_up_for_acquisition(self):
        """
        set up the acquisition to use
        If `self.options.acq` is "default", then use ei, ucb and ttei
        Otherwise, use the `self.options.acq`
        """
        if self.options.acq == "default":
            acq = "ei-ucb-ttei"
        else:
            acq = self.options.acq
        self.acqs_to_use = [elem.lower() for elem in acq.split('-')]
        self.acqs_to_use_counter = {key: 0 for key in self.acqs_to_use}
        if self.options.acq_probs == 'uniform':
            self.acq_probs = np.ones(len(self.acqs_to_use)) / float(len(self.acqs_to_use))
        elif self.options.acq_probs == 'adaptive':
            self.acq_uniform_sampling_prob = 0.05
            self.acq_sampling_weights = {key: 1.0 for key in self.acqs_to_use}
            self.acq_probs = self._get_adaptive_ensemble_acq_probs()
        else:
            self.acq_probs = np.array([float(x) for x in self.options.acq_probs.split('-')])
        self.acq_probs = self.acq_probs / self.acq_probs.sum()
        assert len(self.acq_probs) == len(self.acqs_to_use)

    def _opt_method_optimise_initalise(self):
        """ Important setup: creating the optimization object """
        initial_pool = [mol_lst[0] for mol_lst in self.history.query_points]
        logging.info(f'Length of initial pool {len(initial_pool)}, should be equal to init_capital.')
        if self.acq_opt_method == 'rand_explorer':
            self.acq_optimizer = RandomExplorer(initial_pool=initial_pool,
                                                max_pool_size=self.options.max_pool_size) #### random initialize 
        else:
            raise NotImplementedError("Acq opt method {} not implemented.".format(self.acq_opt_method))


# CPGP Class to use------------------------------------------------------------

class CPGPBandit(GPBandit):
    """ A GP Bandit class on Cartesian product spaces. """
    def __init__(self, func_caller, worker_manager, 
                 is_mf=False, 
                 domain_dist_computers=None, options=None, reporter=None):
        """ Constructor. """ 
        if options is None:  # never gets called, otherwise imports would fail below:
            reporter = get_reporter(reporter)
            if is_mf:
                all_args = get_all_mf_euc_gp_bandit_args()
            else:
                all_args = get_all_cp_gp_bandit_args()
            options = load_options(all_args, reporter)
        self.domain_dist_computers = domain_dist_computers
        self.capital_type = options.capital_type  # store capital_type for Explorer
        super(CPGPBandit, self).__init__(func_caller, worker_manager, is_mf=is_mf,
                                         options=options, reporter=reporter)

    def _child_opt_method_set_up(self):
        """ Set up for child class. Override this method in child class. """
        self.domain_lists_of_dists = None
        if self.domain_dist_computers is None:
            self.domain_dist_computers = [None] * self.domain.num_domains
        self.kernel_params_for_each_domain = [{} for _ in range(self.domain.num_domains)]
        # Create a Dummy GP Fitter so that we can get the mislabel and struct coeffs for
        # otmann. 
        if self.is_an_mf_method():
            fs_orderings = self.func_caller.fidel_space_orderings
            d_orderings = self.func_caller.domain_orderings
            dummy_gp_fitter = CPMFGPFitter(
                [], [], [], config=None,
                fidel_space=self.func_caller.fidel_space,
                domain=self.func_caller.domain,
                fidel_space_kernel_ordering=fs_orderings.kernel_ordering,
                domain_kernel_ordering=d_orderings.kernel_ordering,
                fidel_space_lists_of_dists=None,
                domain_lists_of_dists=None,
                fidel_space_dist_computers=None,
                domain_dist_computers=None,
                options=self.options, reporter=self.reporter
            )
        else:  ##### use this 
            dummy_gp_fitter = MolCPGPFitter(
                [], [], self.func_caller.domain,
                domain_kernel_ordering=self.func_caller.domain_orderings.kernel_ordering,
                domain_lists_of_dists=None,
                domain_dist_computers=self.domain_dist_computers,  # TODO: is domain_dist_computers=None necessary?
                options=self.options, reporter=self.reporter
            )
        
        # Pre-compute distances for all sub-domains in domain - not doing for fidel_space
        # since we don't expect pre-computing distances will be necessary there.
        for idx, dom in enumerate(self.domain.list_of_domains):
            if dom.get_type() == 'neural_network' and self.domain_dist_computers[idx] is None:
                from dragonfly.nn.otmann import get_otmann_distance_computer_from_args
                otm_mislabel_coeffs =  \
                    dummy_gp_fitter.domain_kernel_params_for_each_domain[idx].otmann_mislabel_coeffs
                otm_struct_coeffs =  \
                    dummy_gp_fitter.domain_kernel_params_for_each_domain[idx].otmann_struct_coeffs
                self.domain_dist_computers[idx] = get_otmann_distance_computer_from_args(
                    dom.nn_type, self.options.otmann_non_assignment_penalty,
                    otm_mislabel_coeffs, otm_struct_coeffs, self.options.otmann_dist_type)
                self.kernel_params_for_each_domain[idx]['otmann_dist_type'] = \
                    self.options.otmann_dist_type
        # Report more frquently if Neural networks are present
        domain_types = [dom.get_type() for dom in self.domain.list_of_domains]
        if 'neural_network' in domain_types:
            self.options.report_results_every = self.options.nn_report_results_every

    def _domain_specific_acq_opt_set_up(self):
        """ Set up acquisition optimisation for the child class. """
        if self.acq_opt_method.lower() in ['direct']:
            self._set_up_cp_acq_opt_direct()
        elif self.acq_opt_method.lower() in ['pdoo']:
            self._set_up_cp_acq_opt_pdoo()
        elif self.acq_opt_method.lower() == 'rand':
            self._set_up_cp_acq_opt_rand()
        elif self.acq_opt_method.lower().startswith('ga'):
            self._set_up_cp_acq_opt_ga()
        elif self.acq_opt_method.lower().endswith('explorer'):
            self._set_up_cp_acq_opt_explorer()
        else:
            raise ValueError('Unrecognised acq_opt_method "%s".'%(self.acq_opt_method))

    # Any of these set up methods can be overridden by a child class -------------------
    def _set_up_cp_acq_opt_with_params(self, lead_const, min_iters, max_iters):
        """ Set up acquisition optimisation with params. """
        if self.get_acq_opt_max_evals is None:
            dim_factor = lead_const * min(5, self.domain.get_dim())**2
            self.get_acq_opt_max_evals = lambda t: np.clip(dim_factor * np.sqrt(min(t, 1000)),
                                                           min_iters, max_iters)

    def _set_up_cp_acq_opt_direct(self):
        """ Sets up optimisation for acquisition using direct/pdoo. """
        self._set_up_cp_acq_opt_with_params(2, 1000, 3e4)

    def _set_up_cp_acq_opt_pdoo(self):
        """ Sets up optimisation for acquisition using direct/pdoo. """
        self._set_up_cp_acq_opt_with_params(2, 2000, 6e4)

    def _set_up_cp_acq_opt_rand(self):
        """ Set up optimisation for acquisition using rand. """
        self._set_up_cp_acq_opt_with_params(1, 1000, 3e4)

    def _set_up_cp_acq_opt_ga(self):
        """ Set up optimisation for acquisition using rand. """
        domain_types = [dom.get_type() for dom in self.domain.list_of_domains]
        if 'neural_network' in domain_types:
            # Because Neural networks can be quite expensive
            self._set_up_cp_acq_opt_with_params(1, 300, 1e3)
        else:
            self._set_up_cp_acq_opt_with_params(1, 1000, 3e4)

    def _set_up_cp_acq_opt_explorer(self):
        """ If number of evaluations is not set (-1), it will be set here. """
        self._set_up_cp_acq_opt_with_params(1, 10, 1e3)

    def _compute_lists_of_dists(self, X1, X2):
        """ Computes lists of dists. """
        ret = [None] * self.domain.num_domains
        for idx, dist_comp in enumerate(self.domain_dist_computers):
            if dist_comp is not None:
                X1_idx = [X1_dim[idx] for X1_dim in X1]
                X2_idx = X1_idx if X1 is X2 else [X2_dim[idx] for X2_dim in X2]
                ret[idx] = dist_comp(X1_idx, X2_idx)
        return ret

    def _add_data_to_gp(self, new_data):
        """ Adds data to the GP. Overriding this method. """
        # First add it to the list of dists
        if self.is_an_mf_method():
            _, new_reg_X, _ = new_data
        else:
            new_reg_X, _ = new_data
        if self.domain_lists_of_dists is None:
            # First time, so use all the data
            self.domain_lists_of_dists = self._compute_lists_of_dists(new_reg_X, new_reg_X)
            self.already_evaluated_dists_for = new_reg_X
        else:
            domain_lists_of_dists_new_new = self._compute_lists_of_dists(new_reg_X, new_reg_X)
            domain_lists_of_dists_old_new = self._compute_lists_of_dists(
                                                                 self.already_evaluated_dists_for, new_reg_X)
            for i in range(self.domain.num_domains): # through each domain
                if self.domain_lists_of_dists[i] is None:
                    continue
                for j in range(len(domain_lists_of_dists_new_new[i])):
                    # iterate through each dist in curr dom
                    self.domain_lists_of_dists[i][j] = block_augment_array(
                        self.domain_lists_of_dists[i][j], domain_lists_of_dists_old_new[i][j],
                        domain_lists_of_dists_old_new[i][j].T, domain_lists_of_dists_new_new[i][j])
            self.already_evaluated_dists_for.extend(new_reg_X)
        # Add data to the GP as we will be repeating with the same GP.
        if hasattr(self, 'gp_processor') \
           and hasattr(self.gp_processor, 'fit_type') \
           and self.gp_processor.fit_type == 'fitted_gp':
            reg_data = self._get_gp_reg_data()
            if self.is_an_mf_method():
                self.gp.set_mf_data(reg_data[0], reg_data[1], reg_data[2], build_posterior=False)
                self.gp.set_domain_lists_of_dists(self.domain_lists_of_dists)
            else:
                self.gp.set_data(reg_data[0], reg_data[1], build_posterior=False)
                self.gp.set_domain_lists_of_dists(self.domain_lists_of_dists)
            # Build the posterior
            self.gp.build_posterior()

    def _get_initial_qinfos(self, num_init_evals):
        """ Returns initial qinfos. Currently this is not used """
        return get_cp_domain_initial_qinfos(self.domain, num_init_evals)

    def _get_mf_gp_fitter(self, reg_data, use_additive=False):
        """ Returns the Multi-fidelity GP Fitter. Can be overridded by a child class. """
        # We are not maintaining a list of distances for the domain or the fidelity space.
        raise NotImplementedError("Multi-fidelity currently unimplemented for molecule")
        fs_orderings = self.func_caller.fidel_space_orderings
        return CPMFGPFitter(reg_data[0], reg_data[1], reg_data[2], config=None,
                 fidel_space=self.func_caller.fidel_space,
                 domain=self.func_caller.domain,
                 fidel_space_kernel_ordering=fs_orderings.kernel_ordering,
                 domain_kernel_ordering=self.func_caller.domain_orderings.kernel_ordering,
                 fidel_space_lists_of_dists=None,
                 domain_lists_of_dists=self.domain_lists_of_dists,
                 fidel_space_dist_computers=None,
                 domain_dist_computers=self.domain_dist_computers,
                 options=self.options, reporter=self.reporter)

    def _get_non_mf_gp_fitter(self, reg_data, use_additive=False):
        """ Returns the NOn-Multi-fidelity GP Fitter. Can be overridded by a child class. """
        return MolCPGPFitter(reg_data[0], reg_data[1], self.func_caller.domain,
                 domain_kernel_ordering=self.func_caller.domain_orderings.kernel_ordering,
                 domain_lists_of_dists=self.domain_lists_of_dists,
                 domain_dist_computers=self.domain_dist_computers,
                 options=self.options, reporter=self.reporter)


