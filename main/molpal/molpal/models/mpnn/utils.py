from typing import Optional

from torch import clamp, nn


def get_loss_func(
    dataset_type: str, uncertainty_method: Optional[str] = None
) -> nn.Module:
    """Get the loss function corresponding to a given dataset type

    Parameters
    ----------
    dataset_type : str
        the type of dataset
    uncertainty_method : Optional[str]
        the uncertainty method being used

    Returns
    -------
    loss_function : nn.Module
        a PyTorch loss function

    Raises
    ------
    ValueError
        if is dataset_type is neither "classification" nor "regression"
    """
    if dataset_type == "classification":
        return nn.BCEWithLogitsLoss(reduction="none")

    elif dataset_type == "regression":
        if uncertainty_method == "mve":
            return negative_log_likelihood

        return nn.MSELoss(reduction="none")

    raise ValueError(f'Unsupported dataset type: "{dataset_type}."')


def negative_log_likelihood(means, variances, targets):
    """The NLL loss function as defined in:
    Nix, D.; Weigend, A. ICNN’94. 1994; pp 55–60 vol.1"""
    variances = clamp(variances, min=1e-5)
    return (variances.log() + (means - targets) ** 2 / variances) / 2
