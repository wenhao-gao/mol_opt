from itertools import chain
import logging
from typing import Callable, List

from torch import nn

from main.molpal.molpal.models.mpnn.predict import predict
from ..chemprop.data import MoleculeDataLoader, StandardScaler

def evaluate_predictions(
    preds: List[List[float]], targets: List[List[float]],
    num_tasks: int, metric_func: Callable, dataset_type: str,
    logger: logging.Logger = None) -> List[float]:
    """Evaluates predictions using a metric function and filtering out invalid targets.

    Paramaters
    ----------
    preds : List[List[float]]
        a 2D list of shape (data_size, num_tasks) with model predictions.
    targets : List[List[float]]
        a 2D list of shape (data_size, num_tasks) with targets.
    num_tasks : int
        the number of tasks.
    metric_func : Callable
        a function which takes in a list of targets and a list of predictions
        as arguments and returns the list of scores for each task
    dataset_type : str
        the dataset type.
    logger : logging.Logger
    
    Returns
    -------
    List[float]
        a list with the score for each task based on metric_func
    """
    info = logger.info if logger else print

    if len(preds) == 0:
        return [float('nan')] * num_tasks

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[]] * num_tasks
    valid_targets = [[]] * num_tasks

    for j in range(num_tasks):
        for i in range(len(preds)):
            if targets[i][j] is None:
                continue

            valid_preds[j].append(preds[i][j])
            valid_targets[j].append(targets[i][j])

    # Compute metric
    results = []
    for preds, targets in zip(valid_preds, valid_targets):
        # if all targets or preds are identical classification will crash
        if dataset_type == 'classification':
            if all(t == 0 for t in targets) or all(targets):
                info('Warning: Found a task with targets all 0s or all 1s')
                results.append(float('nan'))
                continue
            if all(p == 0 for p in preds) or all(preds):
                info('Warning: Found a task with predictions all 0s or all 1s')
                results.append(float('nan'))
                continue

        if len(targets) == 0:
            continue

        results.append(metric_func(targets, preds))

    return results

def evaluate(model: nn.Module, data_loader: MoleculeDataLoader, num_tasks: int,
             uncertainty: bool, metric_func: Callable, dataset_type: str,
             scaler: StandardScaler = None,
             logger: logging.Logger = None) -> List[float]:
    """Evaluates a model on a dataset.

    Parameters
    ----------
    model : nn.Module
        a pytorch model.
    data_loader : MoleculeDataLoader
        a MoleculeDataLoader.
    num_tasks : int
        the number of tasks.
    metric_func : Callable
        a function which takes in a list of targets and a list of predictions
        as arguments and returns the list of scores for each task
    dataset_type : str
        the dataset type
    scaler : StandardScaler
        a StandardScaler object fit on the training targets.
    logger : logging.Logger
    
    Returns
    -------
    List[float]
        A list with the score for each task based on metric_func
    """
    preds = predict(model, data_loader, uncertainty,
                    disable=True, scaler=scaler)

    if uncertainty:
        preds = preds[0]

    targets = list(chain(*[dataset.targets() for dataset in data_loader]))
    results = evaluate_predictions(
        preds=preds, targets=targets, num_tasks=num_tasks,
        metric_func=metric_func, dataset_type=dataset_type, logger=logger
    )

    return results
    
