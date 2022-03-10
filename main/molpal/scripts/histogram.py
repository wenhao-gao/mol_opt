import argparse
import csv
from functools import partial
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style='white', context='paper')

def extract_scores(scores_csv, score_col=1, title_line=True):
    if Path(scores_csv).suffix == '.gz':
        open_ = partial(gzip.open, mode='rt')
    else:
        open_ = open
    
    with open_(scores_csv) as fid:
        reader = csv.reader(fid)
        if title_line:
            next(reader)
        scores = []
        for row in tqdm(reader):
            try:
                score = float(row[score_col])
            except ValueError:
                continue
            scores.append(score)
    return np.sort(np.array(scores))

def abbreviate_k_or_M(x: float, pos) -> str:
    if x >= 1e6:
        return f'{x*1e-6:0.1f}M'
    if x >= 1e3:
        return f'{x*1e-3:0.0f}k'

    return f'{x:0.0f}'

def plot_histogram(scores_csv: str, score_col: int, name: str, k: int,
                   log: bool = True, clip: bool = False,
                   maximize: bool = False):
    """Generate and plot the histogram of scores contained in scores_csv

    Parameters
    ----------
    scores_csv : str
        the path to a CSV containing the scores
    score_col : int
        the column of the CSV containing the scores (as floats)
    name : str
        the name of the output file containing the histogram plots
    k : int
        the cutoff to denote
    log : bool, optional
        whether to plot the log-scaled historam side-by-side, by default True
    clip : bool, default=False
        whether to clip positive score values if minimizing the objective or 
        negative score values if maximizing the objective
    maximize : bool, default=False
        whether the scores are to be maximized or minimized
    """
    scores = extract_scores(scores_csv, score_col)
    if clip:
        scores = scores[scores < 0] if not maximize else scores[scores >= 0]
    cutoff = scores[k] if not maximize else scores[-(k+1)]

    if log:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(10, 4))

        BINWIDTH = 0.1
        for ax in (ax1, ax2):
            hist, _, _ = ax.hist(scores, color='b', edgecolor='none',
                    bins=np.arange(min(scores), max(scores)+BINWIDTH, BINWIDTH))
            ax.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
            ax.grid(True, linewidth=1, color='whitesmoke')

        ax1.set_ylabel('Count')
        ax2.set_yscale('log')
        
        if max(hist) > 10e3:
            formatter = ticker.FuncFormatter(abbreviate_k_or_M)
            ax1.yaxis.set_major_formatter(formatter)
            
        ax = fig.add_subplot(111, frameon=False)
        ax.tick_params(labelcolor='none',
                    top=False, bottom=False, left=False, right=False)
        ax.set_xlabel('Score')
    else:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4))

        BINWIDTH = 0.1
        ax.hist(scores, color='b', edgecolor='none',
                bins=np.arange(min(scores), max(scores)+BINWIDTH, BINWIDTH))
        ax.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
        ax.grid(True, linewidth=1, color='whitesmoke')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        
    fig.tight_layout()
    fig.savefig(f'{name}_score_hist.pdf')
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+',
                        help='the paths containing the datasets')
    parser.add_argument('--names', nargs='+',
                        help='the name of each dataset')
    parser.add_argument('--score-cols', nargs='+', type=int,
                        help='the column in each dataset CSV containing the score')
    parser.add_argument('--top-ks', nargs='+', type=int,
                        help='the value of k to use for each dataset')
    parser.add_argument('--no-log', action='store_true', default=False,
                        help='whether to NOT plot the log-scaled histogram')
    parser.add_argument('--clip', action='store_true', default=False,
                        help='whether to clip values above or below 0 if minimizing or maximizing, respectively')
    parser.add_argument('--maximize', action='store_true', default=False,
                        help='whether the scores are attempting to be maximized')
    args = parser.parse_args()
    
    for path, score_col, name, k in zip(args.paths, args.score_cols,
                                        args.names, args.top_ks):
        plot_histogram(path, score_col, name, k, not args.no_log,
                       args.clip, args.maximize)
