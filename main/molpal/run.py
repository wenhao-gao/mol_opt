from __future__ import print_function

import argparse
import yaml
import os
from tdc import Oracle
import ray 
import signal
import sys

from main.optimizer import BaseOptimizer
from molpal import args as molpal_args
from molpal import Explorer

def sigterm_handler(signum, frame):
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)


class MolPAL_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "mol_apl"
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
                ray.init(address='auto')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

    path_here = os.path.dirname(os.path.realpath(__file__))

    if args.output_dir is None:
        args.output_dir = os.path.join(path_here, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    for oracle_name in args.oracles:

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_here, args.config_default)))

        if args.task == "tune":
            try:
                config_tune = yaml.safe_load(open(args.config_tune))
            except:
                config_tune = yaml.safe_load(open(os.path.join(path_here, args.config_tune)))

        oracle = Oracle(name = oracle_name)
        optimizer = MolPAL_Optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main()

