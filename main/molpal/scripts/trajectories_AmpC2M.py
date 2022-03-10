from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from itertools import islice
import math
from operator import itemgetter
from pathlib import Path
import pickle
import pprint
from timeit import default_timer
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style='white', context='paper')

METRICS = ['greedy', 'ucb', 'ts', 'ei', 'pi']
METRIC_NAMES = {'greedy': 'greedy', 'ucb': 'UCB', 'ts': 'TS',
                'ei': 'EI', 'pi': 'PI'}
METRIC_COLORS = dict(zip(METRICS, sns.color_palette('bright')))

MODELS = ['rf', 'nn', 'mpn']
MODEL_COLORS = dict(zip(MODELS, sns.color_palette('dark')))

PRUNE_METHODS = ['best', 'random', 'maxmin', 'leader']
PRUNE_COLORS = dict(zip(PRUNE_METHODS, sns.color_palette('dark')))

SPLITS = [0.004, 0.002, 0.001]

DASHES = ['dash', 'dot', 'dashdot']
MARKERS = ['circle', 'square', 'diamond']

class Timer:
    def __enter__(self):
        self.start = default_timer()
    def __exit__(self, type, value, traceback):
        self.stop = default_timer()
        print(f'{self.stop - self.start:0.4f}s')

class PrettyPct(float):
    def __init__(self, x):
        self.x = 100*x

    def __repr__(self):
        if self.x == 0:
            return '0.000'
        elif self.x >= 0.001:
            return f'{self.x:0.3f}'
        else:
            return f'{self.x:0.1e}'

def boltzmann(xs: Iterable[float]) -> float:
    Z = sum(math.exp(-x) for x in xs)
    return sum(x * math.exp(-x) / Z for x in xs)

def mean(xs: Iterable[float]) -> float:
    return sum(x for x in xs) / len(xs)

def var(xs: Iterable[float], x_mean: float) -> float:
    return sum((x-x_mean)**2 for x in xs) / len(xs)

def mean_and_sd(xs: Iterable[float]) -> Tuple[float, float]:
    x_mean = mean(xs)
    return x_mean, math.sqrt(var(xs, x_mean))

def recursive_conversion(nested_dict):
    if not isinstance(nested_dict, defaultdict):
        return nested_dict

    for k in nested_dict:
        sub_dict = nested_dict[k]
        nested_dict[k] = recursive_conversion(sub_dict)
    return dict(nested_dict)

def read_true(p_true, N: int) -> List[Tuple]:
    data = {}
    with open(p_true) as fid:
        reader = csv.reader(fid); next(reader)
        for row in tqdm(reader, leave=False):
            try:
               data[row[0]] = float(row[1])
            except ValueError:
                continue

    return sorted(data.items(), key=lambda kv: kv[1])[:N]

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

def calculate_rewards(found: List[Tuple], true: List[Tuple],
                      avg: bool = True, smis: bool = True, scores: bool = True
                      ) -> Tuple[float, float, float]:
    N = len(found)
    found_smis, found_scores = zip(*found)
    true_smis, true_scores = zip(*true)

    if avg:
        found_avg = mean(found_scores)
        true_avg = mean(true_scores)
        f_avg = found_avg / true_avg
    else:
        f_avg = None

    if smis:
        found_smis = set(found_smis)
        true_smis = set(true_smis)
        correct_smis = len(found_smis & true_smis)
        f_smis = correct_smis / len(true_smis)
    else:
        f_smis = None

    if scores:
        # true_scores = [round(x, 2) for x in true_scores]
        # found_scores = [round(x, 2) for x in found_scores]
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

    if 'ucb' in str(run) and len(d_it_results) != 6:
        print(run)
    return [(d_it_results[it]) for it in sorted(d_it_results.keys())]

def average_it_results(it_rep_results: Sequence[Sequence[Tuple]]):
    d_metric_results = {
        'avg': [],
        'smis': [],
        'scores': []
    }
    for it_reps in it_rep_results:
        it_avgs, it_smiss, it_scoress = zip(*it_reps)
        d_metric_results['avg'].append(mean_and_sd(it_avgs))
        d_metric_results['smis'].append(mean_and_sd(it_smiss))
        d_metric_results['scores'].append(mean_and_sd(it_scoress))

    return d_metric_results

def gather_all_rewards(parent_dir, true_data: List[Dict], N: int,
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
                    for rep in model.iterdir():
                        rep_ = int(rep.name.split('_')[-1])
                        results[
                            training.name][
                            float(split.name)][
                            model.name][
                            rep_
                        ] = gather_run_results(rep, true_data[rep_], N)
                    continue

                for metric in tqdm(model.iterdir(), 'Metrics', None, False):
                    if metric.name == 'thompson':
                        metric_ = 'ts'
                    else:
                        metric_ = metric.name
                    for rep in tqdm(metric.iterdir(), 'Reps', None, False):
                        rep_ = int(rep.name.split('_')[-1])
                        results[
                            training.name][
                            float(split.name)][
                            model.name][
                            metric_ ][
                            rep_
                        ] = gather_run_results(rep, true_data[rep_], N, maximize)
    results = recursive_conversion(results)

    for training in results:
        for split in results[training]:
            for model in results[training][split]:
                if model == 'random':
                    rep_it_results = list(
                        results[training][split][model].values()
                    )
                    it_rep_results = list(zip(*rep_it_results))
                    it_results = average_it_results(it_rep_results)
                    results[training][split][model] = it_results
                    continue

                for metric in results[training][split][model]:
                    rep_it_results = list(
                        results[training][split][model][metric].values()
                    )
                    it_rep_results = list(zip(*rep_it_results))
                    # print(it_rep_results)
                    it_results = average_it_results(it_rep_results)
                    results[training][split][model][metric] = it_results

    pickle.dump(results, open(cached_rewards, 'wb'))

    return results

def abbreviate_k_or_M(x: float, pos) -> str:
    if x >= 1e6:
        return f'{x*1e-6:0.1f}M'
    if x >= 1e3:
        return f'{x*1e-3:0.0f}k'

    return f'{x:0.0f}'

def plot_model_metrics_rewards(
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
        for metric in results['retrain'][split][model]:
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
                y_sds = [y*100 for y in y_sds]
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
                    fmt=fmt, ms=ms, mec='black', capsize=capsize, label=metric_
                )
        
        add_random_trace(ax, results, split, reward, xs, fmt, ms, capsize)

        ax.set_title(model.upper())
        if i == 0:
            ax.set_ylabel(f'Percentage of Top-{N} {reward.capitalize()} Found')
            ax.legend(loc='upper left', title='Metric')
        ax.set_ylim(bottom=0)

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.set_ylim(top=100)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        ax.grid(True)
    
    fig.tight_layout()

    return fig

def plot_split_models_rewards(
        results, size: int, N: int, metric: str = 'greedy', reward='scores'
    ):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2
        
    for i, (split, ax) in enumerate(zip(SPLITS, axs)):
        xs = [int(size*split * i) for i in range(1, 7)]

        for model in MODELS:
            if model not in results['retrain'][split]:
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
        ax.set_ylim(bottom=0)

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.set_ylim(top=100)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        ax.grid(True)
    
    fig.tight_layout()
    return fig

def plot_split_metrics_rewards(
        results, size: int, N: int,
        model: str = 'rf', reward='scores', si_fig: bool = False
    ):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2

    for i, (split, ax) in enumerate(zip(SPLITS, axs)):
        xs = [int(size*split * i) for i in range(1, 7)]

        for metric in results['retrain'][split][model]:
            if metric == 'greedy':
                metric_ = metric
            elif metric == 'thompson':
                metric_ = 'TS'
            else:
                metric_ = metric.upper()

            ys, y_sds = zip(*results['retrain'][split][model][metric][reward])
            ys = [y*100 for y in ys]
            y_sds = [y*100 for y in y_sds]

            if len(xs) != len(ys):
                continue

            ax.errorbar(
                xs, ys, yerr=y_sds, color=METRIC_COLORS[metric], label=metric_,
                fmt=fmt, ms=ms, mec='black', capsize=capsize
            )
        
        add_random_trace(ax, results, split, reward, xs, fmt, ms, capsize)

        ax.set_title(f'{split*100:0.1f}%')
        if i == 0:
            ax.set_ylabel(f'Percentage of Top-{N} {reward.capitalize()} Found')
            ax.legend(loc='upper left', title='Metric')
        ax.set_ylim(bottom=0)

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.set_ylim(top=100)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        formatter = ticker.FuncFormatter(abbreviate_k_or_M)
        ax.xaxis.set_major_formatter(formatter)
        
        ax.grid(True)
    
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

def write_reward_csv(rewards, split):
    results_df = []
    for training in rewards:
        for model in rewards[training][split]:
            if model == 'random':
                scores = rewards[training][split][model]['scores'][-1]
                smis = rewards[training][split][model]['smis'][-1]
                avg = rewards[training][split][model]['avg'][-1]

                random_results = {
                    'Training': training,
                    'Model': 'random',
                    'Metric': 'random',
                    'Scores ($\pm$ s.d.)': f'{100*scores[0]:0.2f} ({100*scores[1]:0.2f})',
                    'SMILES ($\pm$ s.d.)': f'{100*smis[0]:0.2f} ({100*smis[1]:0.2f})',
                    'Average ($\pm$ s.d.)': f'{100*avg[0]:0.2f} ({100*avg[1]:0.2f})'
                }
                continue

            for metric in rewards[training][split][model]:
                if metric == 'greedy':
                    metric_ = metric.capitalize()
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

    try:
        results_df.append(random_results)
    except UnboundLocalError:
        pass

    df = pd.DataFrame(results_df).set_index(['Training', 'Model', 'Metric'])
    return df

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--parent-dir')
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('-N', type=int, help='the number of top scores')
    parser.add_argument('--split', type=float)
    parser.add_argument('--model')
    parser.add_argument('--mode', required=True,
                        choices=('model-metrics', 'split-models', 
                                 'split-metrics', 'si', 'csv', 'errors', 
                                 'diversity', 'intersection',
                                 'union', 'union-10k50k'))
    # parser.add_argument('--name', default='.')
    # parser.add_argument('--format', '--fmt', default='png',
    #                     choices=('png', 'pdf'))
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--maximize', action='store_true', default=False)

    args = parser.parse_args()

    COLEY = Path('/nfs/ccoleylab001/dgraff/data')
    cached_true_data = COLEY / f'.AmpC_2M_true_data_{args.N}.csv'
    if cached_true_data.exists():
        true_data = pickle.load(open(cached_true_data, 'rb'))
    else:
        p_true_csvs = [COLEY / f'AmpC_2M_scores_{i}.csv' for i in range(5)]
        true_data = [read_true(p, args.N) for p in p_true_csvs]
        pickle.dump(true_data, open(cached_true_data, 'wb'))

    size = 2000000

    if args.mode in ('model-metrics', 'split-models',
                     'split-metrics', 'si', 'csv'):
        results = gather_all_rewards(
            args.parent_dir, true_data, args.N, args.overwrite, args.maximize
        )

    if args.mode == 'model-metrics':
        fig = plot_model_metrics_rewards(
            results, size, args.N, args.split, 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'split-models':
        fig = plot_split_models_rewards(
            results, size, args.N, 'greedy', 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')
    
    elif args.mode == 'split-metrics':
        fig = plot_split_metrics_rewards(
            results, size, args.N, args.model, 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'si':
        fig = plot_model_metrics_rewards(
            results, size, args.N, args.split, 'scores', True
        )
        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'csv':
        df = write_reward_csv(results, args.split)

        name = input('CSV name: ')
        df.to_csv(f'paper/csv/{name}.csv')

    else:
        exit()