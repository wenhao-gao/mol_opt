"""
Note you may encounter permission error in ray if you don't have access to the tmp directory.
One can solve this by runnning:
1. export TMPDIR="$HOME/tmp" (or other directory to store temp files)
2. If the dir doesn't exist, mkdir path_to_tmp
One can check your tmpdir by running:
python -c "import tempfile; print(tempfile.gettempdir())"
"""
from __future__ import print_function

import argparse
import yaml
import os
from tdc import Oracle
import ray 
import signal
import sys

from main.optimizer import BaseOptimizer
from main.molpal.molpal import args as molpal_args
from main.molpal.molpal import Explorer

def sigterm_handler(signum, frame):
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)


class MolPAL_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "mol_pal"
        print('''\
*********************************************************************
*  __    __     ______     __         ______   ______     __        *
* /\ "-./  \   /\  __ \   /\ \       /\  == \ /\  __ \   /\ \       *
* \ \ \-./\ \  \ \ \/\ \  \ \ \____  \ \  _-/ \ \  __ \  \ \ \____  *
*  \ \_\ \ \_\  \ \_____\  \ \_____\  \ \_\    \ \_\ \_\  \ \_____\ *
*   \/_/  \/_/   \/_____/   \/_____/   \/_/     \/_/\/_/   \/_____/ *
*********************************************************************''')
        print('Welcome to MolPAL!')

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        params = vars(molpal_args.gen_args())
        for kw in config.keys():
            params[kw] = config[kw]

        try:
            if 'redis_password' in os.environ:
                ray.init(
                    address=os.environ["ip_head"],
                    _redis_password=os.environ['redis_password']
                )
            else:
                ray.init()
                # ray.init(address='auto')
        except ConnectionError:
            ray.init()
        except PermissionError:
            print('Failed to create a temporary directory for ray')
            raise

        # import ipdb; ipdb.set_trace()
        
        path = params.pop("output_dir")
        explorer = Explorer(oracle=self.oracle, path=path, **params)

        print('Starting exploration...')
        print(f'{explorer.status}.', flush=True)
        explorer.explore_initial()

        while not self.finish:
            print(f'{explorer.status}. Continuing...', flush=True)
            explorer.explore_batch()
            if self.finish:
                break

        ray.shutdown()

