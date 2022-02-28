""" Constants relating to the dataset. """

from typing import List

import pandas as pd

from fingerprints import smiles_to_fp_array

DATASET_PATH = "./data/dockstring-excape/dockstring-dataset.tsv"
CLUSTER_SPLIT_PATH = "./data/dockstring-excape/cluster_split.tsv"


def process_dataframe(
    df: pd.DataFrame,
    targets: List[str] = None,
    drop_nan: bool = True,
    fp=False,
    max_docking_score: float = None,
):
    df = df.copy()

    # Filter out targets if given
    if targets is not None:
        df = df[["smiles"] + targets]

    # Filter out NaNs
    if drop_nan:
        df = df.dropna()

    # Potentially compute fingerprints
    if fp:
        df["fp"] = df["smiles"].map(smiles_to_fp_array)

    # Potentially clip scores
    if max_docking_score is not None:
        df[targets] = df[targets].clip(upper=max_docking_score)

    return df
