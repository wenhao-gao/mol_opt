"""This module contains Model implementations that utilize the sklearn models 
as their underlying model"""
import logging
from molpal.acquirer.metrics import random
from pathlib import Path
import pickle
from typing import Callable, Iterable, Optional, Sequence, Tuple, TypeVar

import joblib
import numpy as np
from ray.util.joblib import register_ray
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from main.molpal.molpal.featurizer import feature_matrix
from main.molpal.molpal.models.base import Model

T = TypeVar('T')

register_ray()

class RFModel(Model):
    """A Random Forest model ensemble for estimating mean and variance

    Attributes (instance)
    ----------
    n_jobs : int
        the number of jobs to parallelize training and prediction over
    model : RandomForestRegressor
        the underlying model on which to train and perform inference
    
    Parameters
    ----------
    test_batch_size : Optional[int] (Default = 65536)
        the size into which testing data should be batched
    """
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 8,
                 min_samples_leaf: int = 1,
                 test_batch_size: Optional[int] = 65536,
                 model_seed: Optional[int] = None,
                 **kwargs):
        test_batch_size = test_batch_size or 65536

        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, n_jobs=-1,
            random_state=model_seed
        )

        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {'means', 'vars'}

    @property
    def type_(self):
        return 'rf'

    def train(self, xs: Iterable[T], ys: Iterable[float], *,
              featurizer: Callable[[T], np.ndarray], retrain: bool = True):
        X = np.array(feature_matrix(xs, featurizer))
        Y = np.array(ys)

        with joblib.parallel_backend('ray'):
            self.model.fit(X, Y)
            Y_pred = self.model.predict(X)

        errors = Y_pred - Y
        logging.info(f'  training MAE: {np.mean(np.abs(errors)):.2f},'
                     f'MSE: {np.mean(errors**2):.2f}')
        return True

    def get_means(self, xs: Sequence) -> np.ndarray:
        X = np.vstack(xs)
        with joblib.parallel_backend('ray'):
            return self.model.predict(X)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack(xs)
        preds = np.zeros((len(X), len(self.model.estimators_)))

        with joblib.parallel_backend('ray'):
            for j, submodel in enumerate(self.model.estimators_):
                preds[:, j] = submodel.predict(xs)

        return np.mean(preds, axis=1), np.var(preds, axis=1)

    def save(self, path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_path = str(path / 'model.pkl')
        pickle.dump(self.model, open(model_path, 'wb'))

        return model_path
    
    def load(self, path):
        self.model = pickle.load(open(path, 'rb'))

class GPModel(Model):
    """Gaussian process model
    
    Attributes
    ----------
    model : GaussianProcessRegressor
    kernel : type[kernels.Kernel]
        the GP kernel that will be used

    Parameters
    ----------
    gp_kernel : str (Default = 'dot')
    test_batch_size : Optional[int] (Default = 1000)
    """
    def __init__(self, gp_kernel: str = 'dot',
                 test_batch_size: Optional[int] = 1024,
                 model_seed: Optional[int] = None,
                 **kwargs):
        test_batch_size = test_batch_size or 1024

        kernel = {
            'dot': kernels.DotProduct,
            'matern': kernels.Matern,
            'rbf': kernels.RBF,
        }[gp_kernel]()
        
        self.model = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=model_seed
        )
        super().__init__(test_batch_size, **kwargs)
        
    @property
    def provides(self):
        return {'means', 'vars'}

    @property
    def type_(self):
        return 'gp'

    def train(self, xs: Iterable[T], ys: Iterable[float], *,
              featurizer, retrain: bool = False) -> bool:
        X = np.array(feature_matrix(xs, featurizer))
        Y = np.array(list(ys))

        self.model.fit(X, Y)
        Y_pred = self.model.predict(X)
        errors = Y_pred - Y
        logging.info('  training MAE: {:.2f}, MSE: {:.2f}'.format(
            np.mean(np.abs(errors)), np.mean(np.power(errors, 2))
        ))
        return True

    def get_means(self, xs: Sequence) -> np.ndarray:
        X = np.vstack(xs)
        
        return self.model.predict(X)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack(xs)
        Y_mean_pred, Y_sd_pred = self.model.predict(X, return_std=True)

        return Y_mean_pred, Y_sd_pred**2
    
    def save(self, path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        model_path = str(path / 'model.pkl')
        pickle.dump(self.model, open(model_path, 'wb'))

        return model_path
    
    def load(self, path):
        self.model = pickle.load(open(path, 'rb'))
