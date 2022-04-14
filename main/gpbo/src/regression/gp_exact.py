import argparse
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dockstring_data
from regression.regression_utils import (
    get_regression_parser,
    split_dataframe_train_test,
    eval_regression,
)
from gp import (
    TanimotoGP,
    fit_gp_hyperparameters,
    batch_predict_mu_var_numpy,
)


DATA_SAVE_NAME = "data.npz"
MODEL_SAVE_NAME = "model.pt"


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    return parser


def get_dataset(df: pd.DataFrame, target=None):
    X = np.stack(df["fp"].values)
    if target is None:
        y = np.zeros((len(X), 1))
    else:
        y = df[target].values.reshape(-1)
    return X.astype(np.float32), y.astype(np.float32)


def get_trained_model(train_dataset):

    X_train, y_train = train_dataset

    model = TanimotoGP(
        train_x=torch.as_tensor(X_train), train_y=torch.as_tensor(y_train.flatten())
    )
    fit_gp_hyperparameters(model)
    return model


def get_predictions(model, dataset, include_var=False):
    X, _ = dataset
    mu, var = batch_predict_mu_var_numpy(
        model, torch.as_tensor(X), include_var=include_var, batch_size=2 ** 16
    )
    if include_var:
        return mu, var
    else:
        return mu


def save_model(model: TanimotoGP, save_dir):
    # Save data
    np.savez_compressed(
        Path(save_dir) / DATA_SAVE_NAME,
        x=model.train_inputs[0].numpy(),
        y=model.train_targets.numpy(),
    )

    # Save model
    torch.save(
        model.state_dict(),
        Path(save_dir) / MODEL_SAVE_NAME,
    )


def load_model(save_dir):
    with np.load(Path(save_dir) / DATA_SAVE_NAME) as npz:
        x = npz["x"]
        y = npz["y"]
    model = TanimotoGP(torch.as_tensor(x), torch.as_tensor(y))
    state_dict = torch.load(
        Path(save_dir) / MODEL_SAVE_NAME,
    )
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(parents=[get_parser(), get_regression_parser()])
    args = parser.parse_args()

    # Load and process dataframes
    process_df = functools.partial(
        dockstring_data.process_dataframe,
        targets=[args.target],
        fp=True,
        max_docking_score=args.max_docking_score,
    )
    if args.data_split is None:
        df_train = pd.read_csv(args.dataset, sep="\t", header=0)
        if args.n_train is not None and args.n_train < len(df_train):
            df_train = df_train.sample(args.n_train)
        df_test = None
    else:
        df_train, df_test = split_dataframe_train_test(
            args.dataset, args.data_split, n_train=args.n_train
        )
        df_test = process_df(df_test)
    df_train = process_df(df_train)

    # Train model with train dataset
    dataset_train = get_dataset(df_train, target=args.target)
    model = get_trained_model(dataset_train)

    # Save weights
    if args.model_save_dir is not None:
        Path(args.model_save_dir).mkdir(exist_ok=True)
        save_model(model, args.model_save_dir)

    # Test on test dataset
    if df_test is not None:
        dataset_test = get_dataset(df_test, target=args.target)

        # Get predictions from best estimator
        y_train_pred = get_predictions(model, dataset_train)
        y_test_pred = get_predictions(model, dataset_test)

        # Save results
        result_dict = dict(
            metrics_train=eval_regression(y_train_pred, dataset_train[1]),
            metrics_test=eval_regression(y_test_pred, dataset_test[1]),
            model_params=model.hparam_dict,
        )
        if args.full_preds:
            result_dict["full_preds"] = dict(
                smiles=list(map(str, df_test.smiles)),
                y_true=list(map(float, dataset_test[1].flatten())),
                y_pred=list(map(float, y_test_pred.flatten())),
            )
        with open(args.output_path, "w") as f:
            json.dump(result_dict, f, indent=4)
