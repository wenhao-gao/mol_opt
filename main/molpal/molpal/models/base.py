"""This module contains the Model abstract base class. All custom models must
implement this interface in order to interact properly with an Explorer"""
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Sequence, Set, Tuple, TypeVar

import numpy as np
from tqdm import tqdm

from main.molpal.molpal.utils import batches

T = TypeVar('T')
T_feat = TypeVar('T_feat')

class Model(ABC):
    """A Model can be trained on input data to predict the values for inputs
    that have not yet been evaluated.

    This is an abstract base class and cannot be instantiated by itself.

    Properties
    ----------
    provides : Set[str]
        the types of values this class of model provides
        - 'means': this model provides a mean predicted value for an input
        - 'vars': this model provides a variance for the predicted mean
        - 'stochastic': this model generates predicted values sthocastically
    type_ : str
        the underlying architecture of model.
        E.g., 'nn' for all models that use the NN class

    Attributes (instance)
    ----------
    model(s)
        the model(s) used to calculate prediction values
    test_batch_size : int
        the size of the batch to split prediction inputs into if not
        already batched
    ncpu : int
        the total number of cores available to parallelize computation over
    additional, class-specific instance attributes

    Parameters
    ----------
    test_batch_size : int
    ncpu : int (Default = 1)
    """
    def __init__(self, test_batch_size: int, **kwargs):
        self.test_batch_size = test_batch_size

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.apply(*args, **kwargs)

    @property
    @abstractmethod
    def provides(self) -> Set[str]:
        """The types of values this model provides"""

    @property
    @abstractmethod
    def type_(self) -> str:
        """The underlying architecture of the Model"""

    @abstractmethod
    def train(self, xs: Iterable[T], ys: Sequence[float], *,
              featurizer: Callable[[T], T_feat], retrain: bool = False) -> bool:
        """Train the model on the input data
        
        Parameters
        ----------
        xs : Iterable[T]
            an iterable of inputs in their identifier representation
        ys : Sequence[float]
            a parallel sequence of scalars that correspond to the regression
            target for each x
        featurize : Callable[[T], T_feat]
            a function that transforms an input from its identifier to its
            feature representation
        retrain : bool (Deafult = False)
            whether the model should be completely retrained
        """
        # TODO: hyperparameter optimizations in inner loop?

    @abstractmethod
    def get_means(self, xs: Sequence) -> np.ndarray:
        """Get the mean predicted values for a sequence of inputs"""

    @abstractmethod
    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        """Get both the predicted mean and variance for a sequence of inputs"""

    def apply(
        self, x_ids: Iterable[T], x_feats: Iterable[T_feat],
        batched_size: Optional[int] = None,
        size: Optional[int] = None, mean_only: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the model to the inputs

        Parameters
        ----------
        x_ids : Iterable[T]
            an iterable of input identifiers that correspond to the
            uncompressed input representations
        x_feats : Iterable[T_feat]
            an iterable of either batches or individual uncompressed feature
            representations corresponding to the input identifiers
        batched_size : Optional[int] (Default = None)
            the size of the batches if xs is an iterable of batches
        size : Optional[int] (Default = None)
            the length of the iterable, if known
        mean_only : bool (Default = True)
            whether to generate the predicted variance in addition to the mean

        Returns
        -------
        means : np.ndarray
            the mean predicted values
        variances: np.ndarray
            the variance in the predicted means, empty if mean_only is True
        """
        if self.type_ == 'mpn':
            xs = x_ids
            batched_size = None
        else:
            xs = x_feats

        if batched_size:
            n_batches = (size//batched_size) + 1 if size else None
        else:
            xs = batches(xs, self.test_batch_size)
            n_batches = (size//self.test_batch_size) + 1 if size else None

        meanss = []
        variancess = []

        if mean_only:
            for batch_xs in tqdm(
                xs, total=n_batches, desc='Inference',
                smoothing=0., unit='smi'
            ):
                means = self.get_means(batch_xs)
                meanss.append(means)
                variancess.append([])
        else:
            for batch_xs in tqdm(
                xs, total=n_batches, desc='Inference',
                smoothing=0., unit='smi'
            ):
                means, variances = self.get_means_and_vars(batch_xs)
                meanss.append(means)
                variancess.append(variances)

        return np.concatenate(meanss), np.concatenate(variancess)
    
    @abstractmethod
    def save(self, path) -> str:
        """Save the model under path"""
    
    @abstractmethod
    def load(self, path):
        """load the model from path"""
