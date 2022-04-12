from random import random
import sys
from timeit import default_timer as time

import optuna
from optuna.trial import Trial

from main.molpal.molpal.args import gen_args
from main.molpal.molpal.explorer import Explorer, IncompatibilityError

def objective(trial: Trial):
    args = gen_args()

    # acquisition hyperparam's
    args.cluster = bool(trial.suggest_int('cluster', 0, 1))
    if not args.cluster and random() > 0.5:
        args.epsilon = trial.suggest_float('epsilon', 0.00, 0.2, step=0.05)

    args.fps = None
    if args.model in {'rf', 'nn'} or args.cluster:
        args.encoder = trial.suggest_categorical(
            'encoder', {'morgan', 'pair', 'rdkit'})
    
    try:
        exp = Explorer(**vars(args))
    except (IncompatibilityError, NotImplementedError) as e:
        print(e)
        return float('-inf')
    
    start = time()
    exp.run()
    total = time() - start
    m, s = divmod(total, 60)
    h, m = divmod(int(m), 60)
    print(f'Total time for trial #{trial.number}: {h}h {m}m {s:0.2f}s\n')

    return exp.top_k_avg

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('#-----------------------------------------------------------------#')
    print()
    print('Best params:')
    print(study.best_params)
    print()
    print('Best trial:')
    print(study.best_trial)
    
    print('Done optimizing!')

if __name__ == '__main__':
    main()
