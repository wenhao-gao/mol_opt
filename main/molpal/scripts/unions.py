from argparse import ArgumentParser
from collections import defaultdict
import csv
from operator import itemgetter
from pathlib import Path
import pickle
import pprint
from typing import List, Set

from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style='white', context='paper')

METRICS = ['greedy', 'ucb', 'ts', 'ei', 'pi']
METRIC_NAMES = {'greedy': 'greedy', 'ucb': 'UCB', 'ts': 'TS',
                'ei': 'EI', 'pi': 'PI'}
METRIC_COLORS = dict(zip(METRICS, sns.color_palette('bright')))

MODELS = ['rf', 'nn', 'mpn']
MODEL_COLORS = dict(zip(MODELS, sns.color_palette('dark')))

SPLITS = [0.004, 0.002, 0.001]

DASHES = ['dash', 'dot', 'dashdot']
MARKERS = ['circle', 'square', 'diamond']

def recursive_conversion(nested_dict):
    if not isinstance(nested_dict, defaultdict):
        return nested_dict

    for k in nested_dict:
        sub_dict = nested_dict[k]
        nested_dict[k] = recursive_conversion(sub_dict)
    return dict(nested_dict)

def get_smis_from_data(p_data) -> Set:
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)
        smis = {row[0] for row in reader}
    
    return smis

def gather_run_smis(run) -> List[Set[str]]:
    data = run / 'data'

    d_it_smis = {}
    for it_data in tqdm(data.iterdir(), desc='Analyzing iterations',
                        leave=False, disable=True):
        try:
            it = int(it_data.stem.split('_')[-1])
        except ValueError:
            continue

        d_it_smis[it] = get_smis_from_data(it_data)

    return [d_it_smis[it] for it in sorted(d_it_smis.keys())]

def gather_metric_smis(metric):
    reps_its_smis = [
        gather_run_smis(rep)
        for rep in tqdm(metric.iterdir(), 'Reps', None, False)
    ]

    it_reps_smis = zip(*reps_its_smis)

    return [
        len(set.union(*rep_smis))
        for rep_smis in it_reps_smis
    ]

def gather_smis_unions(parent_dir, overwrite: bool = False):
    nested_dict = lambda: defaultdict(nested_dict)
    total_smis = nested_dict()

    parent_dir = Path(parent_dir)
    cached_smis_unions = parent_dir / '.smis_unions.pkl'
    if cached_smis_unions.exists() and not overwrite:
        return pickle.load(open(cached_smis_unions, 'rb'))

    for training in tqdm(parent_dir.iterdir(), 'Training', None, False):
        if not training.is_dir():
            continue
        
        for split in tqdm(training.iterdir(), 'Splits', None, False):
            for model in tqdm(split.iterdir(), desc='Models', leave=False):
                if model.name == 'random':
                    total_smis[
                        training.name][
                        float(split.name)][
                        model.name
                    ] = gather_metric_smis(model)
                    # for rep in model.iterdir():
                    #     rep_ = int(rep.name.split('_')[-1])
                    #     total_smis[
                    #         training.name][
                    #         float(split.name)][
                    #         model.name][
                    #         rep_
                    #     ] = gather_run_smis(rep)
                    continue

                for metric in tqdm(model.iterdir(), 'Metrics', None, False):
                    if metric.name == 'thompson':
                        metric_ = 'ts'
                    else:
                        metric_ = metric.name

                    total_smis[
                        training.name][
                        float(split.name)][
                        model.name][
                        metric_
                    ] = gather_metric_smis(metric)
                    # for rep in metric.iterdir():
                    #     rep_ = int(rep.name.split('_')[-1])
                    #     total_smis[
                    #         training.name][
                    #         float(split.name)][
                    #         model.name][
                    #         metric_ ][
                    #         rep_
                    #     ] = gather_run_smis(rep)
    total_smis = recursive_conversion(total_smis)

    # for training in total_smis:
    #     for split in total_smis[training]:
    #         for model in total_smis[training][split]:
    #             if model == 'random':
    #                 rep_it_results = list(
    #                     total_smis[training][split][model].values()
    #                 )
    #                 it_rep_results = list(zip(*rep_it_results))
    #                 it_results = [
    #                     len(set.union(*rep_results))
    #                     for rep_results in it_rep_results
    #                 ]
    #                 total_smis[training][split][model] = it_results
    #                 continue

    #             for metric in total_smis[training][split][model]:
    #                 rep_it_results = list(
    #                     total_smis[training][split][model][metric].values()
    #                 )
    #                 it_rep_results = list(zip(*rep_it_results))
    #                 it_results = [
    #                     len(set.union(*rep_results))
    #                     for rep_results in it_rep_results
    #                 ]
    #                 total_smis[training][split][model][metric] = it_results

    pickle.dump(total_smis, open(cached_smis_unions, 'wb'))

    return total_smis

################################################################################
#------------------------------------------------------------------------------#
################################################################################

def abbreviate_k_or_M(x: float, pos) -> str:
    if x >= 1e6:
        return f'{x*1e-6:0.1f}M'
    if x >= 1e3:
        return f'{x*1e-3:0.0f}k'

    return f'{x:0.0f}'

def plot_unions_10k50k(results_10k, results_50k, metric: str = 'greedy'):
    fig, axs = plt.subplots(1, 2, figsize=(4/1.5 * 2, 4))

    fmt = 'o-'
    ms = 5
    split = 0.010

    resultss = [results_10k, results_50k]
    sizes = [10560, 50240]
    titles = ['10k', '50k']

    for i, results in enumerate(resultss):
        ax = axs[i]
        size = sizes[i]

        xs = [int(size*split * i) for i in range(1, 7)]

        for model in MODELS:
            if model not in results['retrain'][split]:
                continue
            ys = results['retrain'][split][model][metric]
            ax.plot(
                xs, ys, fmt, color=MODEL_COLORS[model],
                label=model.upper(), ms=ms, mec='black'
            )
        
        add_bounds(ax, xs)
        add_random(ax, xs, results, split, fmt, ms)
        
        ax.set_title(titles[i])
        if i == 0:
            ax.set_ylabel(f'Total Number of Unique SMILES')
            ax.legend(loc='upper left', title='Model')

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        ax.grid(True)
    
    fig.tight_layout()
    return fig

def plot_unions_HTS(results, size, metric: str = 'greedy'):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5

    for i, (split, ax) in enumerate(zip(SPLITS, axs)):
        xs = [int(size*split * i) for i in range(1, 7)]

        for model in MODELS:
            if model not in results['retrain'][split]:
                continue
            ys = results['retrain'][split][model][metric]
            ax.plot(
                xs, ys, fmt, color=MODEL_COLORS[model],
                label=model.upper(), ms=ms, mec='black'
            )

        add_bounds(ax, xs)
        add_random(ax, xs, results, split, fmt, ms)

        ax.set_title(f'{split*100:0.1f}%')
        if i == 0:
            ax.set_ylabel(f'Total Number of Unique SMILES')
            ax.legend(loc='upper left', title='Model')

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        formatter = ticker.FuncFormatter(abbreviate_k_or_M)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        ax.grid(True)
    
    fig.tight_layout()
    return fig

def plot_unions_single(results, size,
                       split: float = 0.004, metric: str = 'greedy'):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4/1.5, 4))

    fmt = 'o-'
    ms = 5

    xs = [int(size*split * i) for i in range(1, 7)]

    for model in MODELS:
        if model not in results['retrain'][split]:
            continue
        ys = results['retrain'][split][model][metric]
        ax.plot(
            xs, ys, fmt, color=MODEL_COLORS[model],
            label=model.upper(), ms=ms, mec='black'
        )

    add_bounds(ax, xs)
    add_random(ax, xs, results, split, fmt, ms)

    ax.set_ylabel(f'Total Number of Unique SMILES')
    ax.legend(loc='upper left', title='Model')

    ax.set_xlabel(f'Molecules explored')
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
    ax.xaxis.set_tick_params(rotation=30)

    formatter = ticker.FuncFormatter(abbreviate_k_or_M)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.grid(True)
    
    fig.tight_layout()
    return fig

def add_random(ax, xs, results, split, fmt, ms):
    try:
        ys_random = results['retrain'][split]['random']
    except KeyError:
        ys_random = results['online'][split]['random']
    ax.plot(
        xs, ys_random, fmt, color='grey',
        label='random', ms=ms, mec='black'
    )

def add_bounds(ax, xs):
    ys_upper = [5 * x for x in xs]
    ys_lower = [x + 4*xs[0] for x in xs]
    ax.plot( xs, ys_upper, '-', color='black')
    ax.plot(xs, ys_lower, '-', color='black')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--true-pkl',
                        help='a pickle file containing a dictionary of the true scoring data')
    parser.add_argument('--size', type=int,
                        help='the size of the full library which was explored. You only need to specify this if you are using a truncated pickle file. I.e., your pickle file contains only the top 1000 scores because you only intend to calculate results of the top-k, where k <= 1000')
    parser.add_argument('--parent-dir',
                        help='the parent directory containing all of the results. NOTE: the directory must be organized in the folowing manner: <root>/<online,retrain>/<split_size>/<model>/<metric>/<repeat>/<run>. See the README for a visual description.')
    parser.add_argument('--parent-dir-10k',
                        help='the parent directory of the 10k data to make the union plot of the 10k and 50k data')
    parser.add_argument('--parent-dir-50k',
                        help='the parent directory of the 50k data to make the union plot of the 10k and 50k data')
    parser.add_argument('--mode', required=True,
                        choices=('10k50k', 'HTS', 'HTS-single'),
                        help='what union figure to generate')
    parser.add_argument('--split', type=float, default=0.004,
                        help='which metric to plot union data for')
    parser.add_argument('--metric', choices=METRICS, default='greedy',
                        help='which metric to plot union data for')
    # parser.add_argument('--name', default='.')
    # parser.add_argument('--format', '--fmt', default='png',
    #                     choices=('png', 'pdf'))
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='whether to overwrite the hidden cache file. This is useful if there is new data in PARENT_DIR.')

    args = parser.parse_args()

    if args.true_pkl:
        true_data = pickle.load(open(args.true_pkl, 'rb'))
    size = args.size or len(true_data)

    if args.mode == '10k50k':
        results_10k = gather_smis_unions(args.parent_dir_10k, args.overwrite)
        results_50k = gather_smis_unions(args.parent_dir_50k, args.overwrite)
        fig = plot_unions_10k50k(results_10k, results_50k, args.metric)

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'HTS':
        results = gather_smis_unions(args.parent_dir, args.overwrite)
        fig = plot_unions_HTS(results, size, args.metric)

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')
    
    elif args.mode == 'HTS-single':
        results = gather_smis_unions(args.parent_dir, args.overwrite)
        fig = plot_unions_single(results, size, args.split, args.metric)

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')
