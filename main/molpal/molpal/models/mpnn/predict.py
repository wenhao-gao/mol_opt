from typing import Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm

from main.molpal.molpal.models.mpnn.model import MoleculeModel
from main.molpal.molpal.models.chemprop.data import (
    StandardScaler,
    MoleculeDataLoader,
    MoleculeDataset,
    MoleculeDatapoint,
)

@torch.inference_mode()
def predict(
    model: MoleculeModel,
    smis: Iterable[str],
    batch_size: int = 50,
    ncpu: int = 1,
    uncertainty: Optional[str] = None,
    scaler: Optional[StandardScaler] = None,
    use_gpu: bool = False,
    disable: bool = False,
) -> np.ndarray:
    """Predict the target values of the given SMILES strings with the
    input model

    Parameters
    ----------
    model : mpnn.MoleculeModel
        the model to use
    smis : Iterable[str]
        the SMILES strings to perform inference on
    batch_size : int, default=50
        the size of each minibatch (impacts performance)
    ncpu : int, default=1
        the number of cores over which to parallelize input preparation
    uncertainty : Optional[str], default=None
        the uncertainty quantifiacation method the model uses. None if it
        does not use any uncertainty quantifiacation
    scaler : StandardScaler, default=None
        A StandardScaler object fit on the training targets. If none,
        prediction values will not be transformed to original dataset
    use_gpu : bool, default=False
        whether to use the GPU during inference
    disable : bool, default=False
        whether to disable the progress bar

    Returns
    -------
    predictions : np.ndarray
        an NxM array where N is the number of inputs for which to produce
        predictions and M is the number of prediction tasks
    """
    model.eval()

    device = "cuda" if use_gpu else "cpu"
    model.to(device)

    dataset = MoleculeDataset([MoleculeDatapoint([smi]) for smi in smis])
    data_loader = MoleculeDataLoader(dataset, batch_size, ncpu)

    pred_batches = []

    for batch in tqdm(
        data_loader, desc="Inference", unit="batch", leave=False, disable=disable
    ):
        componentss, _ = batch
        componentss = [
            [
                X.to(device) if isinstance(X, torch.Tensor) else X
                for X in components
            ]
            for components in componentss
        ]
        pred_batch = model(componentss)
        pred_batches.append(pred_batch)

    preds = torch.cat(pred_batches)
    preds = preds.cpu().numpy()

    if uncertainty == "mve":
        if scaler:
            preds[:, 0::2] *= scaler.stds
            preds[:, 0::2] += scaler.means
            preds[:, 1::2] *= scaler.stds**2

        return preds

    if scaler:
        preds *= scaler.stds
        preds += scaler.means

    return preds