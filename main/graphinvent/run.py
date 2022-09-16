from __future__ import print_function

import argparse
import yaml
import os
from copy import deepcopy
import numpy as np
from tdc import Oracle
from main.optimizer import BaseOptimizer, Objdict

from Workflow import Workflow


class GraphInvent_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "graphinvent"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        config = Objdict(config)

        workflow = Workflow(constants=config)
        util.suppress_warnings()
        workflow.learning_phase()
