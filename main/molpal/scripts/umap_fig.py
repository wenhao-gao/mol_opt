import argparse
import csv
from pathlib import Path
import pickle
from typing import Dict, List, Set, Union

import h5py
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse
import numpy as np
from numpy import random
import seaborn as sns
from tqdm import tqdm
import umap

sns.set_theme(style='white', context='paper')

NBINS=50

def reduce_fps(fps_h5: str, k: Union[int, float] = 0.1,
               overwrite: bool = False) -> np.ndarray:
    fps_h5 = Path(fps_h5)
    fps_reduced_npy = fps_h5.with_name(f'{fps_h5.stem}_reduced.npy')
    if fps_reduced_npy.exists() and not overwrite:
        return np.load(fps_reduced_npy)

    with h5py.File(str(fps_h5), 'r') as h5f:
        fps = h5f['fps'][:]

    if isinstance(k, float):
        k = int(k*len(fps))
    train_idxs = random.choice(len(fps), size=k, replace=False)
    X_train = fps[train_idxs]

    transform = umap.UMAP(
        n_neighbors=15, metric='jaccard', n_components=2, verbose=True
    ).fit(X_train)

    fps_reduced = transform.transform(fps)
    
    fps_reduced = np.empty((len(fps), 2))
    for i in tqdm(range(100000)):
        fps_reduced[i] = transform.tranform(fps[i])
    np.save(fps_reduced_npy, fps_reduced)
    print(f'Reduced fingerprints saved to {fps_reduced_npy}')

    return fps_reduced

def get_num_iters(expt_dir: str) -> int:
    data_dir = Path(expt_dir) / 'data'
    scores_csvs = [p_csv for p_csv in data_dir.iterdir()
                   if 'iter' in p_csv.stem]
    return len(scores_csvs)

def read_scores(scores_csv: str) -> Dict:
    """read the scores contained in the file located at scores_csv"""
    scores = {}
    failures = {}
    with open(scores_csv) as fid:
        reader = csv.reader(fid)
        next(reader)
        for row in reader:
            try:
                scores[row[0]] = float(row[1])
            except:
                failures[row[0]] = None
    
    return scores, failures

def get_new_smis_by_epoch(scores_csvs: List[str]) -> List[Dict]:
    """get the set of new points and associated scores acquired at each
    iteration in the list of scores_csvs that are already sorted by iteration"""
    all_smis = set()
    new_smiss = []
    for scores_csv in scores_csvs:
        scores, _ = read_scores(scores_csv)
        new_smis = {smi for smi in scores.keys()
                      if smi not in all_smis}
        new_smiss.append(new_smis)
        all_smis.update(new_smis)
    
    return new_smiss

def add_ellipses(ax, invert=False):
    kwargs = dict(fill=False, color='white' if invert else 'black', lw=1.)
    ax.add_patch(Ellipse(xy=(6.05, -6.0), width=2.9, height=1.2, **kwargs))
    ax.add_patch(Ellipse(xy=(16.05, 4.5), width=1.7, height=2.6, **kwargs))
    ax.add_patch(Ellipse(xy=(2.2, 12.7), width=1.7, height=2.3, **kwargs))

def add_model_data(fig, gs, expt_dir, i, model,
                   d_smi_fp_score, zmin, zmax,
                   portrait, n_models):
    scores_csvs = [p_csv for p_csv in (Path(expt_dir)/'data').iterdir()
                   if 'iter' in p_csv.stem]
    scores_csvs = sorted(scores_csvs, key=lambda p: int(p.stem.split('_')[4]))

    new_smiss = get_new_smis_by_epoch(scores_csvs)
    if portrait:
        MAX_ROW = len(new_smiss)
    else:
        MAX_ROW = n_models

    axs = []
    for j, new_smis in enumerate(new_smiss):
        if portrait:
            row, col = j, i
        else:
            row, col = i, j

        ax = fig.add_subplot(gs[row, col])
        fps_scores = sorted(
            [d_smi_fp_score[smi] for smi in new_smis],
            key=lambda fp_score: fp_score[1], reverse=True
        )
        fps, scores = zip(*fps_scores)
        fps, scores = np.array(fps), -1*np.array(scores)

        ax.scatter(
            fps[:, 0], fps[:, 1], 
            marker='.', c=scores, s=2, cmap='plasma', vmin=zmin, vmax=zmax
        )
        add_ellipses(ax)

        if row==0:
            if portrait:
                ax.set_title(model)

        if row==MAX_ROW:
            if not portrait:
                ax.set_xlabel(j)
        
        if col==0:
            if portrait:
                ax.set_ylabel(row)
            else:
                ax.set_ylabel(model)
        
        ax.set_xticks([])
        ax.set_yticks([])

        axs.append(ax)

    return fig, axs

def si_fig(d_smi_fp_score, expt_dirs, models, 
           portrait=True):
    zmin = -max(score for score in d_smi_score.values() if score < 0)
    zmax = -min(d_smi_score.values())
    zmin = 9 #round((zmin+zmax)/2)

    n_models = len(expt_dirs)
    n_iters = get_num_iters(expt_dirs[0])

    if portrait:
        fig = plt.figure(figsize=(10*1.15, 15), constrained_layout=True)
        gs = fig.add_gridspec(nrows=n_iters, ncols=n_models)
    else:
        fig = plt.figure(figsize=(15*1.15, 10), constrained_layout=True)
        gs = fig.add_gridspec(nrows=n_models, ncols=n_iters)

    axs = []
    for i, (expt_dir, model) in enumerate(zip(expt_dirs, models)):
        fig, axs_ = add_model_data(fig, gs, expt_dir, i, model,
                                   d_smi_fp_score, zmin, zmax,
                                   portrait, n_models)
        axs.extend(axs_)

    ticks = list(range(zmin, round(zmax)))

    colormap = ScalarMappable(cmap='plasma')
    colormap.set_clim(zmin, zmax)
    cbar = plt.colorbar(colormap, ax=axs, aspect=60, ticks=ticks)
    cbar.ax.set_title('Score')

    ticks[0] = f'≤{ticks[0]}'
    cbar.ax.set_yticklabels(ticks)

    if portrait:
        fig.text(0.01, 0.5, 'Iteration', ha='center', va='center', 
                 rotation='vertical', fontsize=14, fontweight='bold',)
        # fig.text(0.465, 1.01, 'Model', ha='center', va='top',
        #          fontsize=14, fontweight='bold',)
    else:
        # fig.text(0.01, 0.5, 'Model', ha='center', va='center',
        #          rotation='vertical', fontsize=16, fontweight='bold')
        fig.text(0.48, 0.01, 'Iteration', ha='center', va='center', 
                 fontsize=16, fontweight='bold',)

    return fig

def add_top1k_panel(fig, gs, d_smi_fp_score):
    top_1k_fps_scores = sorted(
        d_smi_fp_score.values(),
        key = lambda fp_score: fp_score[1]
    )[:1000]
    top_1k_fps_embedded, _ = zip(*top_1k_fps_scores)
    top_1k_fps_embedded = np.array(top_1k_fps_embedded)

    ax = fig.add_subplot(gs[0:2, 0:2])
    ax.scatter(top_1k_fps_embedded[:, 0], top_1k_fps_embedded[:, 1],
                c='grey', marker='.')
    add_ellipses(ax)

    return fig, ax

def add_density_panel(fig, gs, ax1, d_smi_fp_score):
    fps, _ = zip(*d_smi_fp_score.values())
    fps_embedded = np.array(fps)

    ax2 = fig.add_subplot(gs[0:2, 2:])
    _, _, _, im = ax2.hist2d(
        x=fps_embedded[:, 0], y=fps_embedded[:, 1],
        bins=NBINS, cmap='Purples_r'
    )
    ax2_cbar = plt.colorbar(im, ax=(ax1, ax2), aspect=20)
    ax2_cbar.ax.set_title('Points')
    
    ax2.set_yticks([])

    add_ellipses(ax2, True)

    return fig, ax2

def add_model_row(fig, gs, expt_dir, row, iters, model,
                  d_smi_fp_score, zmin, zmax, ):
    scores_csvs = [p_csv for p_csv in (Path(expt_dir)/'data').iterdir()
                   if 'iter' in p_csv.stem]
    scores_csvs = sorted(scores_csvs, key=lambda p: int(p.stem.split('_')[4]))

    col = 0
    axs = []
    for j, new_smis in enumerate(get_new_smis_by_epoch(scores_csvs)):
        if j not in iters:
            continue

        ax = fig.add_subplot(gs[row, col])
        fps_scores = sorted(
            [d_smi_fp_score[smi] for smi in new_smis],
            key=lambda fp_score: fp_score[1], reverse=True)
        fps, scores = zip(*fps_scores)
        fps, scores = np.array(fps), -1*np.array(scores)
        ax.scatter(
            fps[:, 0], fps[:, 1], alpha=0.75,
            marker='.', c=scores, s=2, cmap='plasma', vmin=zmin, vmax=zmax
        )
        add_ellipses(ax)

        if row==4:
            ax.set_xlabel(j)
        if col==0:
            ax.set_ylabel(model)


        ax.set_xticks([])
        ax.set_yticks([])

        axs.append(ax)
        col+=1

    return fig, axs

def main_fig(d_smi_fp_score, expt_dirs,
             models=None, iters=None,):
    models = ['RF', 'NN', 'MPN'] or models
    iters = [0, 1, 3, 5] or iters[:4]

    _, scores = d_smi_fp_score.values()
    zmax = -min(scores)
    zmin = -max(score for score in scores if score < 0)
    zmin = 9 #round(2*(zmin+zmax)/3)

    nrows = 2+len(expt_dirs)
    ncols = 4
    fig = plt.figure(figsize=(2*ncols*1.15, 2*nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows, ncols=4)

    fig, ax1 = add_top1k_panel(fig, gs, d_smi_fp_score)
    fig, ax2 = add_density_panel(fig, gs, ax1, d_smi_fp_score)
    
    axs = []
    for i, (expt_dir, model) in enumerate(zip(expt_dirs, models)):
        fig, axs_ = add_model_row(fig, gs, expt_dir, i+2, iters, model,
                                  d_smi_fp_score, zmin, zmax)
        axs.extend(axs_)
    
    colormap = ScalarMappable(cmap='plasma')
    colormap.set_clim(zmin, zmax)

    ticks = list(range(zmin, round(zmax)))

    cbar = plt.colorbar(colormap, ax=axs, aspect=30, ticks=ticks)
    cbar.ax.set_title('Score')

    ticks[0] = f'≤{ticks[0]}'
    cbar.ax.set_yticklabels(ticks)

    fig.text(-0.03, 1.03, 'A', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='center', ha='right')
    fig.text(-0.0, 1.03, 'B', transform=ax2.transAxes,
             fontsize=16, fontweight='bold', va='center', ha='left')
    fig.text(-0.03, -0.075, 'C', transform=ax1.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='right')

    fig.text(0.475, 0.005, 'Iteration', ha='center', va='center', 
             fontweight='bold')

    return fig

def restricted_float_or_int(arg: str) -> Union[float, int]:
    try:
        value = int(arg)
        if value < 0:
            raise argparse.ArgumentTypeError(f'{value} is less than 0')
    except ValueError:
        value = float(arg)
        if value < 0 or value > 1:
            raise argparse.ArgumentTypeError(f'{value} must be in [0,1]')
    
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps-h5',
                        help='an HDF5 file containing the full fingerprint of each molecule in the library.')
    parser.add_argument('-k', type=restricted_float_or_int, default=0.1,
                        help='the number or fraction of the library from which to train a UMAP embedding. Large libraries (e.g., HTS) are memory-limited and must be trained on a fraction.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='The reduced fingerprints are saved in a file after calculation for reproducibility/speed. Setting this to true will recalculate and overwrite the file, if it exists.')
    parser.add_argument('--fps-npy',
                        help='the filepath of a .npy file containing the reduced fingerprints')
    parser.add_argument('--scores-dict-pkl',
                        help='the filepath of a pickle file containing the scores dictionary')
    parser.add_argument('--smis-csv',
                        help='a csv file containing the SMILES string associated with each fingerprint in fps-h5. Must be in the same ordering as fps-h5')
    parser.add_argument('--expt-dirs', nargs='+',
                        help='the MolPAL output directories')
    parser.add_argument('--models', nargs='+',
                        help='the respective name of each model used in --expt-dirs')
    parser.add_argument('--iters', nargs=4, type=int, default=[0, 1, 3, 5],
                        help='the FOUR iterations of points to show in the main figure')
    parser.add_argument('--si-fig', action='store_true', default=False,
                        help='whether to produce generate the SI fig instead of the main fig')
    parser.add_argument('--landscape', action='store_true', default=False,
                        help='whether to produce a landscape SI figure')
    args = parser.parse_args()

    if args.fps_npy:
        fps_embedded = np.load(args.fps_npy)
    else:
        fps_embedded = reduce_fps(args.fps_h5, args.k, args.overwrite)

    d_smi_score = pickle.load(open(args.scores_dict_pkl, 'rb'))

    with open(args.smis_csv, 'r') as fid:
        reader = csv.reader(fid); next(reader)
        smis = [row[0] for row in tqdm(reader)]
        
    d_smi_idx = {smi: i for i, smi in enumerate(smis)}

    d_smi_fp_score = {
        smi: (fps_embedded[i], d_smi_score[smi])
        for i, smi in tqdm(enumerate(smis)) if smi in d_smi_score
    }

    if not args.si_fig:
        fig = main_fig(
            d_smi_fp_score, args.expt_dirs, args.models, args.iters
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')
    else:
        fig = si_fig(
            d_smi_fp_score, args.expt_dirs, args.models, not args.landscape
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')