from argparse import Namespace
import csv
from datetime import timedelta
from functools import wraps
import logging
import math
import os
import pickle
from time import time
from typing import Any, Callable, List, Tuple, Type, Union

from sklearn.metrics import (auc, mean_absolute_error, mean_squared_error, 
                             precision_recall_curve, r2_score,
                             roc_auc_score, accuracy_score, log_loss)
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .data import StandardScaler, MoleculeDataset
from .nn_utils import NoamLR

def makedirs(*args, **kwargs):
    pass

def get_loss_func(args: Namespace) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Arguments containing the dataset type ("classification", "regression", or "multiclass").
    :return: A PyTorch loss function.
    """
    if args.dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')

    if args.dataset_type == 'regression':
        return nn.MSELoss(reduction='none')

    if args.dataset_type == 'multiclass':
        return nn.CrossEntropyLoss(reduction='none')

    raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)

def bce(targets: List[int], preds: List[float]) -> float:
    """
    Computes the binary cross entropy loss.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed binary cross entropy.
    """
    # Don't use logits because the sigmoid is added in all places except training itself
    bce_func = nn.BCELoss(reduction='mean')
    loss = bce_func(target=torch.Tensor(targets), input=torch.Tensor(preds)).item()

    return loss

def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))

def mse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    return mean_squared_error(targets, preds)

def accuracy(targets: List[int], preds: Union[List[float], List[List[float]]], 
             threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction

    return accuracy_score(targets, hard_preds)

def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], 
                                                    List[float]], float]:
    r"""
    Gets the metric function corresponding to a given metric name.

    Supports:

    * :code:`auc`: Area under the receiver operating characteristic curve
    * :code:`prc-auc`: Area under the precision recall curve
    * :code:`rmse`: Root mean squared error
    * :code:`mse`: Mean squared error
    * :code:`mae`: Mean absolute error
    * :code:`r2`: Coefficient of determination R\ :superscript:`2`
    * :code:`accuracy`: Accuracy (using a threshold to binarize predictions)
    * :code:`cross_entropy`: Cross entropy
    * :code:`binary_cross_entropy`: Binary cross entropy

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'cross_entropy':
        return log_loss

    if metric == 'binary_cross_entropy':
        return bce

    raise ValueError(f'Metric "{metric}" not supported.')

def build_optimizer(model: nn.Module, init_lr: float) -> Optimizer:
    """
    Builds a PyTorch Optimizer.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    return Adam([
        {'params': model.parameters(), 'lr': init_lr, 'weight_decay': 0}
    ])

def build_lr_scheduler(
        optimizer: Optimizer, warmup_epochs: Union[float, int],
        epochs: int, num_lrs: int, train_data_size: int, batch_size: int,
        init_lr: float, max_lr: float, final_lr: float
    ) -> Type[_LRScheduler]:
    """
    Builds a PyTorch learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose learning rate will be scheduled.
    warmup_epochs : Union[float, int]
        The number of epochs during which to linearly increase the learning rate.
    epochs : int
        The total number of epochs
    num_lrs : int
    train_data_size : int
    batch_size : int
    init_lr : float
        The initial learning rate.
    max_lr : float
        The maximum learning rate (achieved after :code:`warmup_epochs`).
    final_lr : float
        The final learning rate (achieved after :code:`total_epochs`)

    Returns
    --------
    Type[_LRScheduelr]
        An initialized learning rate scheduler.
    """
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[warmup_epochs],
        total_epochs=[epochs] * num_lrs,
        steps_per_epoch=train_data_size // batch_size,
        init_lr=[init_lr],
        max_lr=[max_lr],
        final_lr=[final_lr]
    )

def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    If a logger with that name already exists, simply returns that logger.
    Otherwise, creates a new logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of :code:`quiet`.
    One file handler (:code:`verbose.log`) saves all logs, the other (:code:`quiet.log`) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e., print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)

    if logging.getLogger().hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger

def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    """
    Creates a decorator which wraps a function with a timer that prints the elapsed time.

    :param logger_name: The name of the logger used to record output. If None, uses :code:`print` instead.
    :return: A decorator which wraps a function with a timer that prints the elapsed time.
    """
    def timeit_decorator(func: Callable) -> Callable:
        """
        A decorator which wraps a function with a timer that prints the elapsed time.

        :param func: The function to wrap with the timer.
        :return: The function wrapped with the timer.
        """
        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name is not None else print
            info(f'Elapsed time = {delta}')

            return result

        return wrap

    return timeit_decorator

def save_smiles_splits(data_path: str,
                       save_dir: str,
                       train_data: MoleculeDataset = None,
                       val_data: MoleculeDataset = None,
                       test_data: MoleculeDataset = None,
                       smiles_column: str = None) -> None:
    """
    Saves indices of train/val/test split as a pickle file.

    :param data_path: Path to data CSV file.
    :param save_dir: Path where pickle files will be saved.
    :param train_data: Train :class:`~chemprop.data.data.MoleculeDataset`.
    :param val_data: Validation :class:`~chemprop.data.data.MoleculeDataset`.
    :param test_data: Test :class:`~chemprop.data.data.MoleculeDataset`.
    :param smiles_column: The name of the column containing SMILES. By default, uses the first column.
    """
    makedirs(save_dir)

    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)

        if smiles_column is None:
            smiles_column_index = 0
        else:
            smiles_column_index = header.index(smiles_column)

        lines_by_smiles = {}
        indices_by_smiles = {}
        for i, line in enumerate(reader):
            smiles = line[smiles_column_index]
            lines_by_smiles[smiles] = line
            indices_by_smiles[smiles] = i

    all_split_indices = []
    for dataset, name in [(train_data, 'train'), (val_data, 'val'),
                          (test_data, 'test')]:
        if dataset is None:
            continue

        with open(os.path.join(save_dir, f'{name}_smiles.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles'])
            for smiles in dataset.smiles():
                writer.writerow([smiles])

        with open(os.path.join(save_dir, f'{name}_full.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for smiles in dataset.smiles():
                writer.writerow(lines_by_smiles[smiles])

        split_indices = []
        for smiles in dataset.smiles():
            split_indices.append(indices_by_smiles[smiles])
            split_indices = sorted(split_indices)
        all_split_indices.append(split_indices)

    with open(os.path.join(save_dir, 'split_indices.pckl'), 'wb') as f:
        pickle.dump(all_split_indices, f)
