from typing import Iterator, Optional, Sequence

import numpy as np
import ray

from main.molpal.molpal.featurizer import Featurizer, feature_matrix
from main.molpal.molpal.pools.base import MoleculePool
from main.molpal.molpal.utils import batches


class LazyMoleculePool(MoleculePool):
    """A LazyMoleculePool does not precompute fingerprints for the pool

    Attributes (only differences with EagerMoleculePool are shown)
    ----------
    featurizer : Featurizer
        an Featurizer to generate uncompressed representations on the fly
    fps : None
        no fingerprint file is stored for a LazyMoleculePool
    chunk_size : int
        the buffer size of calculated fingerprints during pool iteration
    cluster_ids : None
        no clustering can be performed for a LazyMoleculePool
    cluster_sizes : None
        no clustering can be performed for a LazyMoleculePool
    """

    def get_fp(self, idx: int) -> np.ndarray:
        smi = self.get_smi(idx)
        return self.featurizer(smi)

    def get_fps(self, idxs: Sequence[int]) -> np.ndarray:
        smis = self.get_smis(idxs)
        return np.array([self.featurizer(smi) for smi in smis])

    def fps(self) -> Iterator[np.ndarray]:
        for fps_batch in self.fps_batches():
            for fp in fps_batch:
                yield fp

    def fps_batches(self) -> Iterator[np.ndarray]:
        for smis in batches(self.smis(), self.chunk_size):
            yield np.array(feature_matrix(smis, self.featurizer, True))

    def _encode_mols(self, featurizer: Featurizer, path: Optional[str] = None):
        """Do not precompute any feature representations"""
        self.featurizer = featurizer
        self.chunk_size = int(ray.cluster_resources()["CPU"] * 4096)
        self.fps_ = None

    def _cluster_mols(self, *args, **kwargs) -> None:
        """A LazyMoleculePool can't cluster molecules

        Doing so would require precalculating all uncompressed representations,
        which is what a LazyMoleculePool is designed to avoid. If clustering
        is desired, use the base (Eager)MoleculePool
        """
        print(
            "WARNING: Clustering is not possible for a LazyMoleculePool.",
            "No clustering will be performed.",
        )
