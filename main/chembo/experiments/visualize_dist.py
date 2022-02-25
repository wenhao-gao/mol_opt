"""
Build t-SNE, prop-vs-dist and prop-vs-kernel visualizations.

NOTE:
* Decide whether we want to use titles,
  if we don't we can remove all padding borders,
  otherwise they are there in eps, too
"""

import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
font = {#'family' : 'normal',
        # 'weight' : 'bold',
        'size'   :  40}
plt.rc('font', **font)

from dist.ot_dist_computer import OTChemDistanceComputer
from datasets.loaders import get_chembl_prop, get_chembl
from mols.mol_functions import get_objective_by_name
from mols.mol_kernels import mol_kern_factory

import itertools
import os

# VIS_DIR = 'experiments/results/visualizations'
VIS_DIR = 'experiments/visualizations'

def make_tsne(func, as_subplots=False):
    """
    Plot TSNE embeddings colored with property
    for several distance computers.
    """
    n_mols = 200

    dist_computers = [
        OTChemDistanceComputer(mass_assignment_method='equal',
                                normalisation_method='none',
                                struct_pen_method='bond_frac'),
        OTChemDistanceComputer(mass_assignment_method='equal',
                                normalisation_method='total_mass',
                                struct_pen_method='bond_frac'),
        OTChemDistanceComputer(mass_assignment_method='molecular_mass',
                                normalisation_method='none',
                                struct_pen_method='bond_frac'),
        OTChemDistanceComputer(mass_assignment_method='molecular_mass',
                                normalisation_method='total_mass',
                                struct_pen_method='bond_frac')

    ]
    titles = ['Equal mass assign, no norm', 'Equal mass assign, total mass norm',
              'Mol mass assign, no norm',   'Mol mass assign, total mass norm']

    smile_strings, smiles_to_prop = get_chembl_prop(n_mols=n_mols)
    if func == 'prop':
        smile_strings, smiles_to_prop = get_chembl_prop(n_mols=n_mols)
        prop_list = [smiles_to_prop[sm] for sm in smile_strings]
    else:
        mols = get_chembl(max_size=n_mols)
        smile_strings = [mol.to_smiles() for mol in mols]
        func_ = get_objective_by_name(func)
        prop_list = [func_(mol) for mol in mols]


    f, ll_ax = plt.subplots(2, 2, figsize=(15, 15))
    axes = itertools.chain.from_iterable(ll_ax)
    for ind, (ax, dist_computer, title) in enumerate(zip(axes, dist_computers, titles)):
        distances_mat = dist_computer(smile_strings, smile_strings)[0]

        # plot them
        tsne = TSNE(metric='precomputed')
        points_to_plot = tsne.fit_transform(distances_mat)
        if as_subplots:
            ax.set_title(title)
            ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c=prop_list, cmap=plt.cm.Spectral, s=9, alpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # save separately:
            plt.clf()
            fig = plt.figure()  # figsize=fsize
            ax = fig.add_subplot(1,1,1)
            plt.title(title)
            plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c=prop_list, cmap=plt.cm.Spectral, s=9, alpha=0.8)
            plt.xticks([])
            plt.yticks([])
            # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(os.path.join(VIS_DIR, f'tsne_vis_{func}_{dist_computer}.eps'),
                        format='eps', dpi=1000)  # bbox_inches=extent, pad_inches=0
            plt.clf()

    if as_subplots:
        plt.savefig(os.path.join(VIS_DIR, f'tsne_vis_{func}.eps'), format='eps', dpi=1000)
        plt.clf()


def plot_tsne(func):
    n_mols = 250
    mols = get_chembl(max_size=n_mols, as_mols=True)
    smile_strings = [m.smiles for m in mols]

    title = f"{func} ot-dist"
    distance_computer = OTChemDistanceComputer(mass_assignment_method='molecular_mass',
        normalisation_method='total_mass',
        struct_pen_method='bond_frac')
    distances_mat = distance_computer(smile_strings, smile_strings)[0]

    # title = f"{func} similarity kernel"
    # kernel = mol_kern_factory('similarity_kernel')
    # kern_mat = kernel(mols, mols)
    # distances_mat = 1/kern_mat

    # title = f"{func} fingerprint dist"
    # distances_mat = np.zeros((len(smile_strings), len(smile_strings)))
    # for i in tqdm(range(len(smile_strings))):
    #     for j in range(len(smile_strings)):
    #         distances_mat[i, j] = np.sum((mols[i].to_fingerprint(ftype='numeric') -
    #             mols[j].to_fingerprint(ftype='numeric')) ** 2 )

    tsne = TSNE(metric='precomputed')
    points_to_plot = tsne.fit_transform(distances_mat)

    mols = get_chembl(max_size=n_mols)
    smile_strings = [mol.to_smiles() for mol in mols]
    func_ = get_objective_by_name(func)
    prop_list = [func_(mol) for mol in mols]

    plt.title(title, fontsize=22)
    plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c=prop_list, cmap=plt.cm.Spectral, s=15, alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(VIS_DIR, title.replace(" ", "_")+'.eps'),
                format='eps', dpi=1000)  # bbox_inches=extent, pad_inches=0


def make_pairwise(func, as_subplots=False):
    n_mols = 100

    if func == 'prop':
        smile_strings, smiles_to_prop = get_chembl_prop(n_mols=n_mols)
        prop_list = [smiles_to_prop[sm] for sm in smile_strings]
    else:
        mols = get_chembl(max_size=n_mols)
        smile_strings = [mol.to_smiles() for mol in mols]
        func_ = get_objective_by_name(func)
        prop_list = [func_(mol) for mol in mols]

    dist_computers = [
        OTChemDistanceComputer(mass_assignment_method='equal',
                                normalisation_method='none',
                                struct_pen_method='bond_frac'),
        OTChemDistanceComputer(mass_assignment_method='equal',
                                normalisation_method='total_mass',
                                struct_pen_method='bond_frac'),
        OTChemDistanceComputer(mass_assignment_method='molecular_mass',
                                normalisation_method='none',
                                struct_pen_method='bond_frac'),
        OTChemDistanceComputer(mass_assignment_method='molecular_mass',
                                normalisation_method='total_mass',
                                struct_pen_method='bond_frac')
    ]
    titles = ['Unit weight, Unnormalized', 'Unit weight, Normalized',
              'Molecular mass weight, Unnormalized',   'Molecular mass weight, Normalized']

    f, ll_ax = plt.subplots(2, 2, figsize=(15, 15))
    axes = itertools.chain.from_iterable(ll_ax)
    for ind, (ax, dist_computer, title) in enumerate(zip(axes, dist_computers, titles)):
        distmat = dist_computer(smile_strings, smile_strings)[0]
        xs, ys = [], []
        for i in range(n_mols):
            for j in range(n_mols):
                dist_in_dist = distmat[i, j]
                dist_in_val = np.abs(prop_list[i] - prop_list[j])
                xs.append(dist_in_dist)
                ys.append(dist_in_val)

        if as_subplots:
            ax.set_title(title)
            ax.scatter(xs, ys, s=2, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # save separately:
            plt.clf()
            fig = plt.figure()  # figsize=fsize
            ax = fig.add_subplot(1,1,1)
            plt.title(title, fontsize=22)
            plt.scatter(xs, ys, s=2, alpha=0.6)
            plt.xscale('log')
            plt.xticks([])
            plt.yticks([])
            plt.xlim([None, 1.03*max(xs)])
            plt.xlabel("OT-distance, log scale", fontsize=20)
            if ind == 0:
                plt.ylabel(f"Difference in SA score", fontsize=20)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                extent.x0 -= 0.5
                extent.x1 += 0.1
                extent.y0 -= 0.6
                extent.y1 += 0.7
            else:
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                extent.x0 -= 0.5
                extent.x1 += 0.1
                extent.y0 -= 0.6
                extent.y1 += 0.7
            plt.savefig(os.path.join(VIS_DIR, f"dist_vs_value_{func}_{ind+1}.pdf"),
                                    bbox_inches=extent, pad_inches=0) #bbox_inches=extent, pad_inches=0, format='eps', dpi=1000, 
            plt.clf()

    if as_subplots:
        plt.savefig(os.path.join(VIS_DIR, f"dist_vs_value_{func}.eps"),
                    format='eps', dpi=1000)
        plt.clf()


def make_pairwise_kernel(kernel_name, func, **kwargs):
    n_mols = 100

    mols = get_chembl(max_size=n_mols)
    # smile_strings = [mol.to_smiles() for mol in mols]
    func_ = get_objective_by_name(func)
    kernel = mol_kern_factory(kernel_name, **kwargs)
    kern_mat = kernel(mols, mols)
    prop_list = [func_(mol) for mol in mols]

    xs, ys = [], []
    for i in range(n_mols):
        for j in range(n_mols):
            if mode == "inverse_sim":
                dist_in_dist = 1 / kern_mat[i, j]
            elif mode == "scaled_kernel":
                dist_in_dist = 1 / kern_mat[i, j]
                dist_in_dist /= np.sqrt(kern_mat[i, i] * kern_mat[j, j])
            elif mode == "fps_distance":
                dist_in_dist = np.sum((mols[i].to_fingerprint(ftype='numeric') -
                    mols[j].to_fingerprint(ftype='numeric')) ** 2 )
            else:
                raise ValueError

            dist_in_val = np.abs(prop_list[i] - prop_list[j])
            xs.append(dist_in_dist)
            ys.append(dist_in_val)

    fig = plt.figure()  # figsize=fsize
    ax = fig.add_subplot(1,1,1)
    plt.scatter(xs, ys, s=2, alpha=0.6)
    # plt.yscale('log')
    plt.xscale('log')
    plt.xlim([11, 80])
    plt.xticks([])
    plt.yticks([])
    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(VIS_DIR, f"{kernel_name}_{func}.eps"),
                format='eps', dpi=1000)  # bbox_inches=extent, pad_inches=0
    plt.clf()


if __name__ == "__main__":
    os.makedirs(VIS_DIR, exist_ok=True)
    # make_tsne('qed')
    # make_tsne('sascore')

    # plot_tsne('qed')

    # make_pairwise('prop')
    make_pairwise('qed')
    make_pairwise('sascore')

    # mode = "fps_distance"  # Or: "inverse_sim", "scaled_kernel"
    # make_pairwise_kernel('similarity_kernel', 'qed', mode=mode)
    # make_pairwise_kernel('edgehist_kernel', 'qed', par=2, mode=mode)
    # make_pairwise_kernel('wl_kernel', 'qed',  par=2, mode=mode)
    # make_pairwise_kernel('distance_kernel_expsum', 'qed', dist_computer=OTChemDistanceComputer(), betas=[-0.5] * 4)

    # make_pairwise_kernel('similarity_kernel', 'plogp', mode=mode)
    # make_pairwise_kernel('edgehist_kernel', 'plogp', par=2)
    # make_pairwise_kernel('wl_kernel', 'plogp',  par=2)
    # make_pairwise_kernel('distance_kernel_expsum', 'plogp', dist_computer=OTChemDistanceComputer(), betas=[-0.5] * 4)
