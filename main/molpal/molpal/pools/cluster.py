from itertools import chain
from random import sample
import timeit
from typing import List, Optional

import h5py
import numpy as np
from scipy import sparse
from sklearn import cluster


def cluster_fps_h5(fps_h5: str, ncluster: int = 100) -> List[int]:
    """Cluster the inputs represented by the feature matrix in fps_h5

    Parameters
    ----------
    fps : str
        the filepath of an HDF5 file containing the feature matrix of the
        molecules, where N is the number of molecules and M is the length of
        the feature representation
    ncluster : int (Default = 100)
        the number of clusters to generate

    Returns
    -------
    cluster_ids : List[int]
        the cluster id corresponding to a given fingerprint
    """
    begin = timeit.default_timer()

    BATCH_SIZE = 1024
    n_iter = 1000

    clusterer = cluster.MiniBatchKMeans(ncluster, batch_size=BATCH_SIZE)

    with h5py.File(fps_h5, "r") as h5f:
        fps = h5f["fps"]

        for _ in range(n_iter):
            idxs = sorted(sample(range(len(fps)), BATCH_SIZE))
            clusterer.partial_fit(fps[idxs])

        cluster_ids = [
            clusterer.predict(fps[i : i + BATCH_SIZE])
            for i in range(0, len(fps), BATCH_SIZE)
        ]

    elapsed = timeit.default_timer() - begin
    print(f"Clustering took: {elapsed:0.3f}s")

    return list(chain(*cluster_ids))


def cluster_fps(
    fps: List[np.ndarray],
    ncluster: int = 100,
    method: str = "minibatch",
    ncpu: Optional[int] = None,
) -> np.ndarray:
    """Cluster the molecular fingerprints, fps, by a given method

    Parameters
    ----------
    fps : List[np.ndarray]
        a list of bit vectors corresponding to a given molecule's Morgan
        fingerprint (radius=2, length=1024)
    ncluster : int (Default = 100)
        the number of clusters to form with the given fingerprints (if the
        input method requires this parameter)
    method : str (Default = 'kmeans')
        the clusering method to use.
        Choices include:
        - k-means clustering: 'kmeans'
        - mini-batch k-means clustering: 'minibatch'
        - OPTICS clustering 'optics'
    ncpu : Optional[int]
        the number of cores to parallelize clustering over, if possible

    Returns
    -------
    cluster_ids : np.ndarray
        the cluster id corresponding to a given fingerprint
    """
    begin = timeit.default_timer()

    fps = sparse.vstack(fps, format="csr")

    if method == "kmeans":
        clusterer = cluster.KMeans(n_clusters=ncluster, n_jobs=ncpu)
    elif method == "minibatch":
        clusterer = cluster.MiniBatchKMeans(
            n_clusters=ncluster, n_init=10, batch_size=100, init_size=1000
        )
    elif method == "optics":
        clusterer = cluster.OPTICS(min_samples=0.01, metric="jaccard", n_jobs=ncpu)
    else:
        raise ValueError(f"{method} is not a supported clustering method")

    cluster_ids = clusterer.fit_predict(fps)

    elapsed = timeit.default_timer() - begin
    print(f"Clustering and predictions took: {elapsed:0.3f}s")

    return cluster_ids
