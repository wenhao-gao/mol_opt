"""
Class that performs molecule space traversal.

TODO:
* add better caching of evaluations
  (for the case of 'expensive' functions and enabled filtering)
* better handling of fitness_func arguments
  (lists vs args etc.)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

from collections import defaultdict
from time import time

from synth.forward_synth import RexgenForwardSynthesizer
from myrdkit import Chem
from rdkit_contrib.sascorer import calculateScore as calculateSAScore
from mols.molecule import Molecule, Reaction
from datasets.loaders import get_chembl_prop, get_initial_pool


# class Explorer:
#     def __init__(self):
#         pass

#     def run(self, capital):
#         """ Main method of Explorers.

#         Arguments:
#             capital - for how long to run
#         Returns:
#             opt_value, opt_point, history
#         """
#         pass

class RandomExplorer(object):
    """
    Implements a random evolutionary algorithm
    for exploring molecule space.
    """
    def __init__(self, fitness_func=None, capital_type='return_value',
                 initial_pool=None, max_pool_size=None,
                 n_outcomes=1):
        """
        Params:
            fitness_func {function} - objective to optimize over evolution,
                expected format: func(List[Molecule]) -> value
            capital_type {int/float} - number of steps or other cost of exploration
            initial_pool {list} - just what it says
            max_pool_size {int or None} - whether to keep the pool to top k most fit
            n_outcomes {int} - # of most likely reaction outcomes to keep and evaluate;
                currently this is used for selecting the actual primary outcome of the reaction
        """
        self.fitness_func = fitness_func
        self.capital_type = capital_type
        self.synth = RexgenForwardSynthesizer()
        # if initial_pool == 'default':
        #     initial_pool = get_initial_pool()
        self.pool = initial_pool
        self.max_pool_size = max_pool_size
        self.n_outcomes = n_outcomes
        # TODO: think whether to add additional *synthesized* pool

    def reset_params(self, fitness_func, capital_type, **kwargs):
        self.fitness_func = fitness_func
        self.capital_type = capital_type

    def select_product(self, outcomes, criterion="prob"):
        """ Select the candidate by:
            - highest probability ("prob")
            - highest fitness ("fit")
            - heuristic to determine the actual
              major product ("product")
        """
        supported_criteria = ["prob", "fit", "product"]
        if criterion == "prob":
            top_pt = outcomes[0]
            top_val = self.fitness_func([top_pt])
        elif criterion == "fit":
            top_pt = sorted(outcomes, key=lambda mol: self.fitness_func([mol]))[-1]
            top_val = self.fitness_func([top_pt])
        elif criterion == "product":
            # using a heuristic here, e.g. weight
            weight_fn = Chem.Descriptors.ExactMolWt
            top_pt = sorted(outcomes, key=lambda mol: weight_fn(mol.to_rdkit()))[-1]
            top_val = self.fitness_func([top_pt])
        else:
            raise ValueError(f"Argument {criterion} not supported, choose from {supported_criteria}")
        return top_pt, top_val

    def run_step(self):
        outcomes = []
        while not outcomes:
            # choose molecules to cross-over
            r_size = np.random.randint(2,3)
            mols = np.random.choice(self.pool, size=r_size)
            # evolve
            reaction = Reaction(mols)
            try:
                outcomes = self.synth.predict_outcome(reaction, k=self.n_outcomes)
            except RuntimeError as e:
                logging.info('Synthesizer failed, restarting with another subset.')
                outcomes = []
            else:
                if not outcomes:
                    logging.info('Synthesizer returned an empty set of results, restarting with another subset.')
        # print('outcomes', outcomes)

        top_pt, top_val = self.select_product(outcomes, criterion="product")
        # print("top_pt", top_pt)
        self.pool.append(top_pt)

        # filter
        if self.max_pool_size is not None:
            self.pool = sorted(self.pool, key=lambda mol: self.fitness_func([mol]))[-self.max_pool_size:]

        # update history
        if self.history['objective_vals']:
            top_value = max(top_val, self.history['objective_vals'][-1])
            self.history['objective_vals'].append(top_value)

    def run(self, capital):
        """
        Params:
        :data: start dataset (list of Molecules)
        :capital: number of steps or other cost of exploration
        """
        self._initialize_history()
        # for step-count capital
        if self.capital_type == 'return_value':
            capital = int(capital)
            for _ in range(capital):
                self.run_step()
        else:
            raise NotImplementedError(f"Capital type {self.capital_type} not implemented.")

        top_pt = self.get_best(k=1)[0]
        top_val = self.fitness_func([top_pt])
        return top_val, top_pt, self.history

    def get_best(self, k): #### used in run 
        top = sorted(self.pool, key=lambda mol: self.fitness_func([mol]))[-k:]
        return top

    def _initialize_history(self): #### used in run 
        n_init = len(self.pool)
        max_over_pool = np.max([self.fitness_func([mol]) for mol in self.pool])
        self.history = {
                        'objective_vals': [max_over_pool] * n_init
                        }

