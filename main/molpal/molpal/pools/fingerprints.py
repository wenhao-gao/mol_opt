from pathlib import Path
from typing import Iterable, Set, Tuple, TypeVar

import h5py
import ray
from tqdm import tqdm

from main.molpal.molpal.featurizer import Featurizer, feature_matrix
from main.molpal.molpal.utils import batches

T = TypeVar("T")


def feature_matrix_hdf5(
    smis: Iterable[str],
    size: int,
    *,
    featurizer: Featurizer = Featurizer(),
    name: str = "fps.h5",
    path: str = "."
) -> Tuple[str, Set[int]]:
    """Precalculate the fature matrix of xs with the given featurizer and store
    the matrix in an HDF5 file

    Parameters
    ----------
    xs: Iterable[T]
        the inputs for which to generate the feature matrix
    size : int
        the length of the iterable
    ncpu : int (Default = 0)
        the number of cores to parallelize feature matrix generation over
    featurizer : Featurizer, default=Featurizer()
        an object that encodes inputs from an identifier representation to
        a feature representation
    name : str (Default = 'fps.h5')
        the name of the output HDF5 file with or without the extension
    path : str (Default = '.')
        the path under which the HDF5 file should be written

    Returns
    -------
    fps_h5 : str
        the filename of an hdf5 file containing the feature matrix of the
        representations generated from the molecules in the input file.
        The row ordering corresponds to the ordering of smis
    invalid_idxs : Set[int]
        the set of indices in xs containing invalid inputs
    """
    fps_h5 = str((Path(path) / name).with_suffix(".h5"))

    ncpu = int(ray.cluster_resources()["CPU"])
    with h5py.File(fps_h5, "w") as h5f:
        CHUNKSIZE = 512

        fps_dset = h5f.create_dataset(
            "fps",
            (size, len(featurizer)),
            chunks=(CHUNKSIZE, len(featurizer)),
            dtype="int8",
        )

        batch_size = CHUNKSIZE * 2 * ncpu
        n_batches = size // batch_size + 1

        invalid_idxs = set()
        i = 0
        offset = 0

        for smis_batch in tqdm(
            batches(smis, batch_size),
            "Precalculating fps",
            n_batches,
            unit="batch",
        ):
            fps = feature_matrix(smis_batch, featurizer, disable=True)
            for fp in tqdm(
                fps, "Writing", batch_size, False, unit="smi", smoothing=0.0
            ):
                if fp is None:
                    invalid_idxs.add(i + offset)
                    offset += 1
                    continue

                fps_dset[i] = fp
                i += 1

        valid_size = size - len(invalid_idxs)
        if valid_size != size:
            fps_dset.resize(valid_size, axis=0)

    return fps_h5, invalid_idxs
