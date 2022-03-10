from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from itertools import islice
from operator import itemgetter
from pathlib import Path
import pickle
from typing import Iterable, List, Set, Tuple

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style='white', context='paper')

def recursive_conversion(nested_dict):
    if not isinstance(nested_dict, defaultdict):
        return nested_dict

    for k in nested_dict:
        sub_dict = nested_dict[k]
        nested_dict[k] = recursive_conversion(sub_dict)
    return dict(nested_dict)

def read_data(p_data, k, maximize: bool = False) -> List[Tuple]:
    c = 1 if maximize else -1
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)
        # the data files are always sorted
        data = [(row[0], c * float(row[1]))
                for row in islice(reader, k) if row[1]]
    
    return data

def get_smis_from_data(p_data) -> Set:
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)
        smis = {row[0] for row in reader}
    
    return smis

def boltzmann(xs: Iterable[float]) -> float:
    X = np.array(xs)
    E = np.exp(-X)
    Z = E.sum()
    return (X * E / Z).sum()

def calculate_rewards(found: List[Tuple], true: List[Tuple],
                      avg: bool = True, smis: bool = True, scores: bool = True
                      ) -> Tuple[float, float, float]:
    N = len(found)
    found_smis, found_scores = zip(*found)
    true_smis, true_scores = zip(*true)

    if avg:
        found_avg = np.mean(found_scores)
        true_avg = np.mean(true_scores)
        f_avg = found_avg / true_avg
    else:
        f_avg = None

    # if boltzmann:
    #     found_boltzmann = boltzmann(found_scores)
    #     true_boltzmann = boltzmann(true_scores)
    #     f_boltzmann = boltzmann(found_scores) / boltzmann(true_scores)
    # else:
    #     f_boltzmann = None

    if smis:
        found_smis = set(found_smis)
        true_smis = set(true_smis)
        correct_smis = len(found_smis & true_smis)
        f_smis = correct_smis / len(true_smis)
    else:
        f_smis = None

    if scores:
        missed_scores = Counter(true_scores)
        missed_scores.subtract(found_scores)
        n_missed_scores = sum(
            count if count > 0 else 0
            for count in missed_scores.values()
        )
        f_scores = (N - n_missed_scores) / N
    else:
        f_scores = None

    return f_avg, f_smis, f_scores

def gather_run_results(
        run, true_data, N, maximize: bool = False
    ) -> List[Tuple[float, float, float]]:
    data = run / 'data'

    d_it_results = {}
    for it_data in tqdm(data.iterdir(), 'Iters', None, False):
        try:
            it = int(it_data.stem.split('_')[-1])
        except ValueError:
            continue

        found = read_data(it_data, N, maximize)
        d_it_results[it] = calculate_rewards(found, true_data)

    return [(d_it_results[it]) for it in sorted(d_it_results.keys())]

def gather_metric_results(metric, true_data, N, maximize: bool = False):
    rep_results = np.array([
        gather_run_results(rep, true_data, N, maximize)
        for rep in tqdm(metric.iterdir(), 'Reps', None, False)
    ])

    means = np.mean(rep_results, axis=0)
    sds = np.sqrt(np.var(rep_results, axis=0))

    return {
        'avg': list(zip(means[:, 0].tolist(), sds[:, 0].tolist())),
        'smis': list(zip(means[:, 1].tolist(), sds[:, 1].tolist())),
        'scores': list(zip(means[:, 2].tolist(), sds[:, 2].tolist()))
    }

def gather_all_rewards(parent_dir, true_data, N: int,
                       overwrite: bool = False, maximize: bool = False):
    nested_dict = lambda: defaultdict(nested_dict)
    results = nested_dict()

    parent_dir = Path(parent_dir)
    cached_rewards = parent_dir / f'.all_rewards_{N}.pkl'
    if cached_rewards.exists() and not overwrite:
        return pickle.load(open(cached_rewards, 'rb'))

    for training in tqdm(parent_dir.iterdir(), 'Training', None, False):
        if not training.is_dir():
            continue

        for split in tqdm(training.iterdir(), 'Splits', None, False):
            for model in tqdm(split.iterdir(), 'Models', None, False):
                if model.name == 'random':
                    results[
                        training.name][
                        float(split.name)][
                        model.name
                    ] = gather_metric_results(model, true_data, N, maximize)
                    continue

                for metric in tqdm(model.iterdir(), 'Metrics', None, False):
                    if metric.name == 'thompson':
                        metric_ = 'ts'
                    else:
                        metric_ = metric.name
                    
                    results[training.name][
                        float(split.name)][
                        model.name][
                        metric_
                    ] = gather_metric_results(metric, true_data, N, maximize)
    results = recursive_conversion(results)

    pickle.dump(results, open(cached_rewards, 'wb'))

    return results

################################################################################
#------------------------------------------------------------------------------#
################################################################################

METRICS = ['greedy', 'ucb', 'ts', 'ei', 'pi']
METRIC_NAMES = {'greedy': 'greedy', 'ucb': 'UCB', 'ts': 'TS',
                'ei': 'EI', 'pi': 'PI'}
METRIC_COLORS = dict(zip(METRICS, sns.color_palette('bright')))

MODELS = ['rf', 'nn', 'mpn']
MODEL_COLORS = dict(zip(MODELS, sns.color_palette('dark')))

SPLITS = [0.004, 0.002, 0.001]

DASHES = ['dash', 'dot', 'dashdot']
MARKERS = ['circle', 'square', 'diamond']

def style_axis(ax):
    ax.set_xlabel(f'Molecules explored')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=100)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
    ax.xaxis.set_tick_params(rotation=30)
    ax.grid(True)

def abbreviate_k_or_M(x: float, pos) -> str:
    if x >= 1e6:
        return f'{x*1e-6:0.1f}M'
    if x >= 1e3:
        return f'{x*1e-3:0.0f}k'

    return f'{x:0.0f}'

def plot_model_metrics(
        results, size: int, N: int,
        split: float = 0.010, reward='scores', si_fig: bool = False
    ):
    xs = [int(size*split * i) for i in range(1, 7)]

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                            figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2
        
    for i, (model, ax) in enumerate(zip(MODELS, axs)):
        for metric in METRICS:
            if metric == 'greedy':
                metric_ = metric
            elif metric == 'thompson':
                metric_ = 'TS'
            else:
                metric_ = metric.upper()

            if not si_fig:
                ys, y_sds = zip(
                    *results['retrain'][split][model][metric][reward]
                )
                ys = [y*100 for y in ys]
                y_sds = [y*100 for y in y_sds]
                ax.errorbar(
                    xs, ys, yerr=y_sds, color=METRIC_COLORS[metric], 
                    label=metric_, fmt=fmt, ms=ms, mec='black', capsize=capsize
                )
            else:
                ys, y_sds = zip(
                    *results['retrain'][split][model][metric][reward]
                )
                ys = [y*100 for y in ys]
                ax.plot(
                    xs, ys, fmt, color=METRIC_COLORS[metric], 
                    ms=ms, mec='black', alpha=0.33,
                )

                ys, y_sds = zip(
                    *results['online'][split][model][metric][reward]
                )
                ys = [y*100 for y in ys]
                y_sds = [y*100 for y in y_sds]
                ax.errorbar(
                    xs, ys, yerr=y_sds, color=METRIC_COLORS[metric],
                    fmt=fmt, ms=ms, mec='black', capsize=capsize,
                    label=metric_
                )

                formatter = ticker.FuncFormatter(abbreviate_k_or_M)
                ax.xaxis.set_major_formatter(formatter)
        
        add_random_trace(ax, results, split, reward, xs, fmt, ms, capsize)

        ax.set_title(model.upper())
        if i == 0:
            ax.set_ylabel(f'Percentage of Top-{N} {reward.capitalize()} Found')
            ax.legend(loc='upper left', title='Metric')
        
        style_axis(ax)
    
    fig.tight_layout()

    return fig

def plot_split_models(
        results, size: int, N: int, metric: str = 'greedy', reward='scores'
    ):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2
        
    for i, (split, ax) in enumerate(zip(SPLITS, axs)):
        xs = [int(size*split * i) for i in range(1, 7)]

        for model in MODELS:
            if model == 'random':
                continue

            ys, y_sds = zip(*results['retrain'][split][model][metric][reward])
            ys = [y*100 for y in ys]
            y_sds = [y*100 for y in y_sds]

            if len(xs) != len(ys):
                continue

            ax.errorbar(
                xs, ys, yerr=y_sds, color=MODEL_COLORS[model],
                label=model.upper(), fmt=fmt, ms=ms, mec='black', 
                capsize=capsize
            )
        
        add_random_trace(ax, results, split, reward, xs, fmt, ms, capsize)

        ax.set_title(f'{split*100:0.1f}%')
        if i == 0:
            ax.set_ylabel(f'Percentage of Top-{N} {reward.capitalize()} Found')
            ax.legend(loc='upper left', title='Model')

        style_axis(ax)

        formatter = ticker.FuncFormatter(abbreviate_k_or_M)
        ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    return fig

def plot_split_metrics(
        results, size: int, N: int,
        model: str = 'rf', reward='scores'
    ):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2

    for i, (split, ax) in enumerate(zip(SPLITS, axs)):
        if split not in results['retrain']:
            continue

        for metric in METRICS:
            if metric not in results['retrain'][split][model]:
                continue

            if metric == 'greedy':
                metric_ = metric
            elif metric == 'thompson':
                metric_ = 'TS'
            else:
                metric_ = metric.upper()
                
            ys, y_sds = zip(*results['retrain'][split][model][metric][reward])
            ys = [y*100 for y in ys]
            y_sds = [y*100 for y in y_sds]

            xs = [int(size*split * (i+1)) for i in range(len(ys))]

            ax.errorbar(
                xs, ys, yerr=y_sds, color=METRIC_COLORS[metric],
                fmt=fmt, ms=ms, mec='black', capsize=capsize,
                label=metric_
            )
        
        add_random_trace(ax, results, split, reward, xs, fmt, ms, capsize)

        ax.set_title(f'{split*100:0.1f}%')
        if i == 0:
            ax.set_ylabel(f'Percentage of Top-{N} {reward.capitalize()} Found')
            ax.legend(loc='upper left', title='Metric')

        style_axis(ax)
        formatter = ticker.FuncFormatter(abbreviate_k_or_M)
        ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()

    return fig

def add_random_trace(ax, results, split, reward, xs, fmt, ms, capsize):
    try:
        try:
            ys, y_sds = zip(*results['retrain'][split]['random'][reward])
        except KeyError:
            ys, y_sds = zip(*results['online'][split]['random'][reward])
    except KeyError:
        return

    ys = [y*100 for y in ys]
    y_sds = [y*100 for y in y_sds]
    ax.errorbar(
        xs, ys, yerr=y_sds, fmt=fmt, ms=ms, color='grey',
        mec='black', capsize=capsize, label='random'
    )

def plot_single_batch(
        full_results, single_batch_results, size: int, N: int,
        metric: str = 'greedy', reward='scores'
    ):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4/1.5, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2

    for model in MODELS:
        split = 0.004
        xs = [int(size*split * i) for i in range(1, 7)]

        ys, y_sds = zip(*full_results['retrain'][split][model][metric][reward])
        ys = [y*100 for y in ys]
        y_sds = [y*100 for y in y_sds]

        if len(xs) != len(ys):
            continue

        ax.plot(
            xs, ys, fmt, color=MODEL_COLORS[model],
            ms=ms, mec='black', alpha=0.33
        )

        split = 0.004
        xs = [int(size * split), int(size * 0.024)]
        ys, y_sds = zip(
            *single_batch_results['retrain'][split][model][metric][reward]
        )
        ys = [y*100 for y in ys]
        y_sds = [y*100 for y in y_sds]

        ax.errorbar(
            xs, ys, yerr=y_sds, fmt='o-', color=MODEL_COLORS[model], 
            ms=ms, mec='black', capsize=capsize, label=model.upper()
        )

        split = 0.02
        xs = [int(size * split), int(size * 0.024)]

        ys, y_sds = zip(
            *single_batch_results['retrain'][split][model][metric][reward]
        )
        ys = [y*100 for y in ys]
        y_sds = [y*100 for y in y_sds]

        ax.errorbar(
            xs, ys, yerr=y_sds, fmt='o--', color=MODEL_COLORS[model],
            ms=ms, mec='black', capsize=capsize,
        )

    ax.set_ylabel(f'Percentage of Top-{N} {reward.capitalize()} Found')
    ax.legend(loc='upper left', title='Model')

    style_axis(ax)

    formatter = ticker.FuncFormatter(abbreviate_k_or_M)
    ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    return fig

def plot_singles(
        results, size: int, N: int, split: float,
        model: str = 'mpn', metric: str = 'greedy', reward='scores'
    ):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4, 4))

    colors = sns.color_palette('Greens', len(results['retrain']))
    ms = 5
    capsize = 2
    
    for i, split_ in enumerate(results['retrain']):
        xs = [int(size*split_), int(size*split_)+int(size*split)]

        ys, y_sds = zip(*results['retrain'][split_][model][metric][reward])
        ys = [y*100 for y in ys]
        y_sds = [y*100 for y in y_sds]

        if len(xs) != len(ys):
            continue

        ax.errorbar(
            xs, ys, yerr=y_sds, fmt='o-', color=colors[i], 
            ms=ms, mec='black', capsize=capsize, label=split_
        )

    ax.set_ylabel(f'Percentage of Top-{N} {reward.capitalize()} Found')
    ax.legend(loc='upper left', title='Init fraction')

    style_axis(ax)

    formatter = ticker.FuncFormatter(abbreviate_k_or_M)
    ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    return fig

def plot_convergence(
        results, size: int, N: int, metric: str = 'greedy', reward='scores'
    ):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4/1.5, 4))

    fmt = 'o-'
    ms = 5
    
    split = 0.001        

    for model in MODELS:
        ys, y_sds = zip(*results['retrain'][split][model][metric][reward])
        ys = [y*100 for y in ys]
        y_sds = [y*100 for y in y_sds]

        xs = [int(size*split * (i+1)) for i in range(len(ys))]

        ax.plot(
            xs, ys, fmt, color=MODEL_COLORS[model],
            label=model.upper(), ms=ms, mec='black'
        )
    
    ax.set_ylabel(f'Percentage of Top-{N} {reward.capitalize()} Found')
    ax.legend(loc='upper left', title='Model')
    
    style_axis(ax)
    
    formatter = ticker.FuncFormatter(abbreviate_k_or_M)
    ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    return fig

def write_csv(rewards, split):
    results_df = []

    for training in ('online', 'retrain'):
        for model in MODELS:
            if model not in rewards[training][split]:
                continue

            for metric in METRICS:
                if metric not in rewards[training][split][model]:
                    continue

                if metric == 'greedy':
                    metric_ = metric
                elif metric == 'thompson':
                    metric_ = 'TS'
                else:
                    metric_ = metric.upper()

                scores = rewards[training][split][model][metric]['scores'][-1]
                smis = rewards[training][split][model][metric]['smis'][-1]
                avg = rewards[training][split][model][metric]['avg'][-1]

                results_df.append({
                    'Training': training,
                    'Model': model.upper(),
                    'Metric': metric_,
                    'Scores ($\pm$ s.d.)': f'{100*scores[0]:0.1f} ({100*scores[1]:0.1f})',
                    'SMILES ($\pm$ s.d.)': f'{100*smis[0]:0.1f} ({100*smis[1]:0.1f})',
                    'Average ($\pm$ s.d.)': f'{100*avg[0]:0.2f} ({100*avg[1]:0.2f})'
                })

    if 'random' in rewards['online'][split]:
        scores = rewards['online'][split]['random']['scores'][-1]
        smis = rewards['online'][split]['random']['smis'][-1]
        avg = rewards['online'][split]['random']['avg'][-1]

        random_results = {
            'Training': 'random',
            'Model': 'random',
            'Metric': 'random',
            'Scores ($\pm$ s.d.)': f'{100*scores[0]:0.1f} ({100*scores[1]:0.1f})',
            'SMILES ($\pm$ s.d.)': f'{100*smis[0]:0.1f} ({100*smis[1]:0.1f})',
            'Average ($\pm$ s.d.)': f'{100*avg[0]:0.2f} ({100*avg[1]:0.2f})'
        }
        results_df.append(random_results)
    elif 'random' in rewards['retrain'][split]:
        scores = rewards['retrain'][split]['random']['scores'][-1]
        smis = rewards['retrain'][split]['random']['smis'][-1]
        avg = rewards['retrain'][split]['random']['avg'][-1]

        random_results = {
            'Training': 'random',
            'Model': 'random',
            'Metric': 'random',
            'Scores ($\pm$ s.d.)': f'{100*scores[0]:0.1f} ({100*scores[1]:0.1f})',
            'SMILES ($\pm$ s.d.)': f'{100*smis[0]:0.1f} ({100*smis[1]:0.1f})',
            'Average ($\pm$ s.d.)': f'{100*avg[0]:0.2f} ({100*avg[1]:0.2f})'
        }
        results_df.append(random_results)

    df = pd.DataFrame(results_df).set_index(['Training', 'Model', 'Metric'])

    return df

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--true-pkl',
                        help='a pickle file containing a dictionary of the true scoring data')
    parser.add_argument('--size', type=int,
                        help='the size of the full library which was explored. You only need to specify this if you are using a truncated pickle file. I.e., your pickle file contains only the top 1000 scores because you only intend to calculate results of the top-k, where k <= 1000')
    parser.add_argument('--parent-dir',
                        help='the parent directory containing all of the results. NOTE: the directory must be organized in the folowing manner: <root>/<online,retrain>/<split_size>/<model>/<metric>/<repeat>/<run>. See the README for a visual description.')
    parser.add_argument('--parent-dir-sb',
                        help='the parent directory of the single batch data')
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('-N', type=int,
                        help='the number of top scores from which to calculate perforamnce')
    parser.add_argument('--split', type=float, default=0.004,
                        help='the split size to plot when using model-metrics mode')
    parser.add_argument('--model', default='mpn',
                        help='the model class to plot when using split-metrics mode')
    parser.add_argument('--metric', default='greedy',
                        help='the metric to plot when use split-models mode')
    parser.add_argument('--mode', required=True,
                        choices=('model-metrics', 'split-models', 
                                 'split-metrics', 'si', 'single-batch', 'singles',
                                 'convergence', 'csv', 'errors', 
                                 'diversity', 'intersection'),
                        help='what figure to generate. For "x-y" modes, this corresponds to the figure structure, where there will be a separate panel for each "x" and in each panel there will be traces corresponding to each independent "y". E.g., "model-metrics" makes a figure with three sepearate panels, one for each model and inside each panel a trace for each metric. "si" will make the trajectory plots present in the SI.')
    # parser.add_argument('--name', default='.')
    # parser.add_argument('--format', '--fmt', default='png',
    #                     choices=('png', 'pdf'))
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='whether to overwrite the hidden cache file. This is useful if there is new data in PARENT_DIR.')
    parser.add_argument('--maximize', action='store_true', default=False,
                        help='whether the objective for which you are calculating performance should be maximized.')

    args = parser.parse_args()

    if args.true_pkl:
        true_data = pickle.load(open(args.true_pkl, 'rb'))
        size = args.size or len(true_data)
        try:
            true_data = sorted(true_data.items(), key=itemgetter(1))
        except AttributeError:
            true_data = sorted(true_data, key=itemgetter(1))

        if args.maximize:
            true_data = true_data[::-1]
        true_data = true_data[:args.N]

    if args.mode in ('model-metrics', 'split-models',
                     'split-metrics', 'si', 'singles',
                     'single-batch', 'convergence', 'csv'):
        results = gather_all_rewards(
            args.parent_dir, true_data, args.N, args.overwrite, args.maximize
        )

    if args.mode == 'model-metrics':
        fig = plot_model_metrics(
            results, size, args.N, args.split, 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'split-models':
        fig = plot_split_models(
            results, size, args.N, args.metric, 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')
    
    elif args.mode == 'split-metrics':
        fig = plot_split_metrics(
            results, size, args.N, args.model, 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'si':
        fig = plot_model_metrics(
            results, size, args.N, args.split, 'scores', True
        )
        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'csv':
        df = write_csv(results, args.split)

        name = input('CSV name: ')
        df.to_csv(f'paper/csv/{name}.csv')

    elif args.mode == 'single-batch':
        single_batch_results = gather_all_rewards(
            args.parent_dir_sb, true_data, args.N,
            args.overwrite, args.maximize
        )

        fig = plot_single_batch(
            results, single_batch_results, size, args.N, 'greedy', 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')
    
    elif args.mode == 'singles':
        # single_batch_results = gather_all_rewards(
        #     args.parent_dir_sb, true_data, args.N,
        #     args.overwrite, args.maximize
        # )
        fig = plot_singles(
            results, size, args.N, args.split, 'mpn', 'greedy', 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'convergence':
        fig = plot_convergence(results, size, args.N, 'greedy', 'scores')

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    else:
        exit()