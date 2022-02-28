""" Common code for dockstring regression baselines """

import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats
import sklearn.metrics as metrics


def get_regression_parser():
    """Return ArgumentParser object for regression."""
    parser = argparse.ArgumentParser(add_help=False)

    # Important/mandatory arguments
    parser.add_argument(
        "--target", type=str, required=True, help="Protein target to fit to."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset tsv file."
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default=None,
        help="Path to tsv file defining train/test split. Default is no split.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output file (json)."
    )

    # Optional arguments
    parser.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="Max number of training points to use. Default is to use entire training set.",
    )
    parser.add_argument(
        "--max_docking_score",
        type=float,
        default=math.inf,
        help="Maximum allowable docking score (positive scores can be argued to be unphysical).",
    )
    parser.add_argument(
        "--full_preds",
        action="store_true",
        help="Flag to save the full set of predictions in the output.",
    )
    parser.add_argument(
        "--model_save_dir", type=str, default=None, help="Directory to save model in."
    )

    return parser


def split_dataframe_train_test(
    dataset_path,
    data_split_path,
    n_train: int = None,
):

    # Load dataset
    data = pd.read_csv(dataset_path, sep="\t", header=0).set_index("inchikey")

    # Split
    splits = (
        pd.read_csv(data_split_path, sep="\t", header=0)[
            ["inchikey", "smiles", "split"]
        ]
        .set_index("inchikey")
        .loc[data.index]
    )
    mask_train = splits["split"] == "train"
    mask_test = splits["split"] == "test"
    df_train = data[mask_train]
    df_test = data[mask_test]

    # Potentially subsample dataset
    if n_train is not None and n_train < len(data):
        df_train = df_train.sample(n=n_train, replace=False)

    return df_train, df_test


def eval_regression(
    y_pred: np.array,
    y_true: np.array,
    y_pred_std: np.array = None,
    n_subsample: int = None,
):

    # Potentially subsample train set
    assert len(y_pred) == len(y_true)
    if n_subsample is not None and n_subsample < len(y_pred):
        idxs = np.random.choice(len(y_pred), size=n_subsample, replace=False)
        y_pred = y_pred[idxs]
        y_true = y_true[idxs]
        if y_pred_std is not None:
            y_pred_std = y_pred_std[idxs]

    metrics_dict = dict(
        R2=float(metrics.r2_score(y_true=y_true, y_pred=y_pred)),
        mse=float(metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)),
        mae=float(metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)),
    )
    if y_pred_std is not None:
        metrics_dict["mean_logp"] = float(
            stats.norm(y_pred, y_pred_std).logpdf(y_true).mean()
        )
    return metrics_dict
