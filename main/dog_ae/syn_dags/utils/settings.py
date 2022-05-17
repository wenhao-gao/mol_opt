

import configparser
from os import path
import os

import numpy as np
import torch

_config = None

TORCH_INT = torch.int64
NP_INT = np.int64
TORCH_FLT = torch.float32


PAD_VALUE = -10000

TOTAL_LOSS_TB_STRING = "total-loss"


def get_repo_path():
    return path.join(path.dirname(__file__), '../../')


def get_config() -> configparser.ConfigParser:
    global _config
    if _config is None:
        _config = configparser.ConfigParser()
        _config.read(path.join(get_repo_path(), 'synthesis-dags-config.ini'))
    return _config


def torch_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
