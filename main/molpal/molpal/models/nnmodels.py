"""This module contains Model implementations that utilize an NN model as their 
underlying model"""
from functools import partial
import logging
import json
import os
from pathlib import Path
from typing import (Callable, Iterable, List, NoReturn,
                    Optional, Sequence, Tuple, TypeVar)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
from numpy import ndarray
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from main.molpal.molpal.featurizer import Featurizer, feature_matrix
from main.molpal.molpal.models.base import Model

T = TypeVar('T')
T_feat = TypeVar('T_feat')
Dataset = tf.data.Dataset

def mve_loss(y_true, y_pred):
    mu = y_pred[:, 0]
    var = tf.math.softplus(y_pred[:, 1])

    return tf.reduce_mean(
        tf.math.log(2*3.141592)/2
        + tf.math.log(var)/2
        + tf.math.square(mu-y_true)/(2*var)
    )

class NN:
    """A feed-forward neural network model

    Attributes
    ----------
    model : keras.Sequential
        the underlying model on which to train and perform inference with
    optimizer : keras.optimizers.Adam
        the model optimizer
    loss : Callable
        the loss function to use
    input_size : int
        the dimension of the model input
    output_size : int
        the dimension of the model output
    batch_size : int
        the size to batch training into        
    uncertainty : Optional[str]
        the uncertainty method this model is using (if at all)
    uncertainty : bool
       Whether the model directly predicts its own uncertainty
    mean : float
        the mean of the unnormalized data
    std : float
        the standard deviation of the unnormalized data

    Parameters
    ----------
    input_size : int
    num_tasks : int
    batch_size : int, default=4096
    layer_sizes : Optional[Sequence[int]], default=None
        the sizes of the hidden layers in the network. If None, default to
        two hidden layers with 100 neurons each.
    dropout : Optional[float], default=None
        if specified, add a dropout hidden layer with the specified dropout
        rate after each hidden layer
    activation : Optional[str], default='relu'
        the name of the activation function to use
    uncertainty : Optional[str], default=None
    """
    def __init__(self, input_size: int, num_tasks: int,
                 batch_size: int = 4096,
                 layer_sizes: Optional[Sequence[int]] = None,
                 dropout: Optional[float] = None,
                 activation: Optional[str] = 'relu',
                 uncertainty: Optional[str] = None,
                 model_seed: Optional[int] = None):
        self.input_size = input_size
        self.batch_size = batch_size

        self.uncertainty = uncertainty

        layer_sizes = layer_sizes or [100, 100]
        self.model, self.optimizer, self.loss = self.build(
            input_size, num_tasks, layer_sizes, dropout,
            self.uncertainty, activation
        )

        self.mean = 0
        self.std = 0

        tf.random.set_seed(model_seed)
        
    def build(self, input_size, num_tasks, layer_sizes, dropout, 
              uncertainty, activation):
        """Build the model, optimizer, and loss function"""
        # self.model = keras.Sequential()

        dropout_at_predict = uncertainty == 'dropout'
        output_size = 2*num_tasks if self.uncertainty else num_tasks
        
        inputs = keras.layers.Input(shape=(input_size,))
        
        hidden = inputs
        for layer_size in layer_sizes:
            hidden = keras.layers.Dense(
                units=layer_size,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(0.01),
            )(hidden)

            if dropout:
                hidden = keras.layers.Dropout(
                    dropout
                )(hidden, training=dropout_at_predict)

        outputs = keras.layers.Dense(
            output_size, activation='linear'
        )(hidden)

        model = keras.Model(inputs, outputs)

        if uncertainty not in {'mve'}:
            optimizer = keras.optimizers.Adam(lr=0.01)
            loss = keras.losses.mse
        elif uncertainty == 'mve':
            optimizer = keras.optimizers.Adam(lr=0.05)
            loss = mve_loss
        else:
            raise ValueError(
                f'Unrecognized uncertainty method: "{uncertainty}"')

        return model, optimizer, loss

    def train(self, xs: Iterable[T], ys: Iterable[float],
              featurizer: Callable[[T], ndarray]) -> bool:
        """Train the model on xs and ys with the given featurizer

        Parameters
        ----------
        xs : Sequence[T]
            an sequence of inputs in their identifier representations
        ys : Sequence[float]
            a parallel sequence of target values for these inputs
        featurize : Callable[[T], ndarray]
            a function that transforms an identifier into its uncompressed
            feature representation
        
        Returns
        -------
        True
        """
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        X = np.array(feature_matrix(xs, featurizer))
        Y = self._normalize(ys)

        self.model.fit(
            X, Y, batch_size=self.batch_size, validation_split=0.2,
            epochs=50, validation_freq=2, verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5,
                    restore_best_weights=True, verbose=0
                ),
                tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False)
            ]
        )

        return True

    def predict(self, xs: Sequence[ndarray]) -> ndarray:
        X = np.stack(xs, axis=0)
        Y_pred = self.model.predict(X)

        if self.uncertainty == 'mve':
            Y_pred[:, 0::2] = Y_pred[:, 0::2] * self.std + self.mean
            Y_pred[:, 1::2] = Y_pred[:, 1::2] * self.std**2
        else:
            Y_pred = Y_pred * self.std + self.mean

        return Y_pred
    
    def save(self, path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_path = f'{path}/model'
        self.model.save(model_path, include_optimizer=True)

        state_path = f'{path}/state.json'
        state = {
            'std': self.std,
            'mean': self.mean,
            'model_path': model_path
        }
        json.dump(state, open(state_path, 'w'), indent=4)

        return state_path

    def load(self, path):
        state = json.load(open(path, 'r'))
        
        model_path = state['model_path']
        self.std = state['std']
        self.mean = state['mean']

        if self.uncertainty == 'mve':
            custom_objects = {'mve_loss': mve_loss}
        else:
            custom_objects = {}

        self.model = keras.models.load_model(
            model_path, custom_objects=custom_objects
        )
    
    def _normalize(self, ys: Sequence[float]) -> ndarray:
        Y = np.stack(list(ys))
        self.mean = np.nanmean(Y, axis=0)
        self.std = np.nanstd(Y, axis=0)

        return (Y - self.mean) / self.std

class NNModel(Model):
    """A simple feed-forward neural network model

    Attributes
    ----------
    model : Type[NN]
        the underlying neural net on which to train and perform inference
    
    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    
    See also
    --------
    NNDropoutModel
    NNEnsembleModel
    NNTwoOutputModel
    """
    def __init__(self, input_size: int, test_batch_size: Optional[int] = 4096,
                 dropout: Optional[float] = 0.0,
                 model_seed: Optional[int] = None,
                 **kwargs):
        test_batch_size = test_batch_size or 4096

        self.build_model = partial(
            NN, input_size=input_size, num_tasks=1,
            batch_size=test_batch_size, dropout=dropout,
            model_seed=model_seed
        )
        self.model = self.build_model()

        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {'means'}

    @property
    def type_(self):
        return 'nn'

    def train(self, xs: Iterable[T], ys: Sequence[Optional[float]], *,
              featurizer: Featurizer, retrain: bool = False) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys, featurizer)

    def get_means(self, xs: List) -> ndarray:
        return self.model.predict(xs)[:, 0]

    def get_means_and_vars(self, xs: List) -> NoReturn:
        raise TypeError('NNModel can\'t predict variances!')

    def save(self, path) -> str:
        return self.model.save(path)
    
    def load(self, path):
        self.model.load(path)

class NNEnsembleModel(Model):
    """A feed-forward neural network ensemble model for estimating mean
    and variance.
    
    Attributes
    ----------
    models : List[Type[NN]]
        the underlying neural nets on which to train and perform inference
    
    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    ensemble_size : int (Default = 5)
        the number of separate models to train
    bootstrap_ensemble : bool
        NOTE: UNUSED
    """
    def __init__(self, input_size: int, test_batch_size: Optional[int] = 4096,
                 dropout: Optional[float] = 0.0, ensemble_size: int = 5,
                 bootstrap_ensemble: Optional[bool] = False,
                 model_seed: Optional[int] = None, **kwargs):
        test_batch_size = test_batch_size or 4096
        self.build_model = partial(
            NN, input_size=input_size, num_tasks=1,
            batch_size=test_batch_size, dropout=dropout,
            model_seed=model_seed
        )

        self.ensemble_size = ensemble_size
        self.models = [self.build_model() for _ in range(self.ensemble_size)]

        self.bootstrap_ensemble = bootstrap_ensemble # TODO: Actually use this

        super().__init__(test_batch_size=test_batch_size, **kwargs)

    @property
    def type_(self):
        return 'nn'

    @property
    def provides(self):
        return {'means', 'vars'}

    def train(self, xs: Iterable[T], ys: Sequence[Optional[float]], *,
              featurizer: Featurizer, retrain: bool = False):
        if retrain:
            self.models = [
                self.build_model() for _ in range(self.ensemble_size)
            ]

        return all([model.train(xs, ys, featurizer) for model in self.models])

    def get_means(self, xs: Sequence) -> np.ndarray:
        preds = np.zeros((len(xs), len(self.models)))
        for j, model in tqdm(enumerate(self.models), leave=False,
                             desc='ensemble prediction', unit='model'):
            preds[:, j] = model.predict(xs)[:, 0]

        return np.mean(preds, axis=1)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.zeros((len(xs), len(self.models)))
        for j, model in tqdm(enumerate(self.models), leave=False,
                             desc='ensemble prediction', unit='model'):
            preds[:, j] = model.predict(xs)[:, 0]

        return np.mean(preds, axis=1), np.var(preds, axis=1)

    def save(self, path) -> str:
        for i, model in enumerate(self.models):
            model.save(path, f'model_{i}')

        return path
    
    def load(self, path):
        for model, model_path in zip(self.models, path.iterdir()):
            model.load(model_path)

class NNTwoOutputModel(Model):
    """Feed forward neural network with two outputs so it learns to predict
    its own uncertainty at the same time
    
    Attributes
    ----------
    model : Type[NN]
        the underlying neural net on which to train and perform inference
    
    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    """
    def __init__(self, input_size: int,
                 test_batch_size: Optional[int] = 4096,
                 dropout: Optional[float] = 0.0,
                 model_seed: Optional[int] = None,
                 **kwargs):
        test_batch_size = test_batch_size or 4096

        self.build_model = partial(
            NN, input_size=input_size, num_tasks=1,
            batch_size=test_batch_size, dropout=dropout,
            uncertainty='mve', model_seed=model_seed
        )
        self.model = self.build_model()

        super().__init__(test_batch_size=test_batch_size, **kwargs)

    @property
    def type_(self):
        return 'nn'

    @property
    def provides(self):
        return {'means', 'vars'}

    def train(self, xs: Iterable[T], ys: Sequence[Optional[float]], *,
              featurizer: Featurizer, retrain: bool = False) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys, featurizer)

    def get_means(self, xs: Sequence) -> np.ndarray:
        preds = self.model.predict(xs)
        return preds[:, 0]

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        preds = self.model.predict(xs)
        return preds[:, 0], self._safe_softplus(preds[:, 1])

    def save(self, path) -> str:
        return self.model.save(path)
    
    def load(self, path):
        self.model.load(path)

    @classmethod
    def _safe_softplus(cls, xs):
        in_range = xs < 100
        return np.log(1 + np.exp(xs*in_range))*in_range + xs*(1 - in_range)

class NNDropoutModel(Model):
    """Feed forward neural network that uses MC dropout for UQ
    
    Attributes
    ----------
    model : Type[NN]
        the underlying neural net on which to train and perform inference
    dropout_size : int

    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    dropout_size : int (Default = 10)
        the number of passes to make through the network during inference
    """
    def __init__(self, input_size: int, test_batch_size: Optional[int] = 4096,
                 dropout: Optional[float] = 0.2, dropout_size: int = 10,
                 model_seed: Optional[int] = None, **kwargs):
        test_batch_size = test_batch_size or 4096

        self.build_model = partial(
            NN, input_size=input_size, num_tasks=1,
            batch_size=test_batch_size, dropout=dropout,
            uncertainty='dropout', model_seed=model_seed
        )
        self.model = self.build_model()
        self.dropout_size = dropout_size

        super().__init__(test_batch_size=test_batch_size, **kwargs)

    @property
    def type_(self):
        return 'nn'

    @property
    def provides(self):
        return {'means', 'vars', 'stochastic'}

    def train(self, xs: Iterable[T], ys: Sequence[Optional[float]], *,
              featurizer: Featurizer, retrain: bool = False) -> bool:
        if retrain:
            self.model = self.build_model()
        
        return self.model.train(xs, ys, featurizer)

    def get_means(self, xs: Sequence) -> ndarray:
        predss = self._get_predss(xs)
        return np.mean(predss, axis=1)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        predss = self._get_predss(xs)
        return np.mean(predss, axis=1), np.var(predss, axis=1)

    def _get_predss(self, xs: Sequence) -> ndarray:
        """Get the predictions for each dropout pass"""
        predss = np.zeros((len(xs), self.dropout_size))
        for j in tqdm(range(self.dropout_size), leave=False,
                      desc='bootstrap prediction', unit='pass'):
            predss[:, j] = self.model.predict(xs)[:, 0]

        return predss

    def save(self, path) -> str:
        return self.model.save(path)
    
    def load(self, path):
        self.model.load(path)
