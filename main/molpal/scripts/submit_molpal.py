import subprocess as sp
import sys
import time

BATCH_SCRIPT = 'run_molpal.batch'

LIBRARY = sys.argv[1]
CONFIG = sys.argv[2]

if LIBRARY in ('AmpC', 'HTS'):
    SIZES = ['0.004', '0.002', '0.001']
else:
    SIZES = ['0.01']

MODELS = ['rf', 'nn', 'mpn']
METRICS = ['greedy', 'ei', 'pi', 'thompson', 'ucb']
REPS = range(3) if LIBRARY == 'AmpC' else range(5)

for model in MODELS:
    conf_method_ = {
        'rf': 'none',
        'mpn': 'mve',
        'nn': 'dropout'
    }[model]

    if LIBRARY == 'AmpC':
        if model == 'rf':
            sbatch_extra = ['-p', 'shared', '-c', '12']
        elif model == 'mpn':
            sbatch_extra = ['-p', 'gpu', '--gres', 'gpu:1',
                            '-c', '12', '--constraint', 'cc3.7']
        else:
            sbatch_extra = ['-p', 'gpu', '--gres', 'gpu:1',
                            '-c', '1', '--constraint', 'cc3.7']
    else:
        if model == 'rf':
            sbatch_extra = ['-p', 'serial_requeue', '-c', '8']
        elif model == 'mpn':
            sbatch_extra = ['-p', 'gpu_requeue', '--gres', 'gpu:1', 
                            '-c', '8', '--constraint', 'cc3.7']
        else:
            sbatch_extra = ['-p', 'gpu_requeue', '--gres', 'gpu:1',
                            '-c', '2', '--constraint', 'cc3.7']

    for size in SIZES:
        for metric in METRICS:
            if metric == 'greedy' and model == 'nn':
                conf_method = 'none'
            else:
                conf_method = conf_method_

            for rep in REPS:
                name = f'{LIBRARY}_{model}_{metric}_{size}_{rep}_online'

                sbatch_argv = [
                    '--output', f'molpal_{name}_%j.out',
                    '--error', f'molpal_{name}_%j.err',
                    *sbatch_extra
                ]

                python_argv = (
                    f'--name {name} --config {CONFIG} --metric {metric} ' 
                    + f'--init-size {size} --batch-size {size} ' 
                    + f'--model {model}  --conf-method {conf_method} -vvvv'
                ).split()

                print(sp.run(['sbatch', *sbatch_argv, 
                              BATCH_SCRIPT, *python_argv]))

                time.sleep(1)

# submit random jobs
if LIBRARY == 'AmpC':
    script = 'run_molpal_AmpC.batch'
else:
    script = 'run_molpal_rq.batch'

for size in SIZES:
    for rep in REPS:
        model = 'rf'
        conf_method = 'none'
        metric = 'random'
        name = f'{LIBRARY}_{metric}_{size}_{rep}'

        sbatch_argv = ['--output', f'molpal_{name}_%j.out',
                       '--error', f'molpal_{name}_%j.err',
                       '-p', 'serial_requeue', '-c', '8']

        script_argv = (
            f'--name {name} --config {CONFIG} --metric {metric} ' 
            + f'--init-size {size} --batch-size {size} ' 
            + f'--model {model}  --conf-method {conf_method}  -vvvv'
        ).split()

        print(sp.run(['sbatch', *sbatch_argv, BATCH_SCRIPT, *script_argv]))

        time.sleep(1)

exit(0)
