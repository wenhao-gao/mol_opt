import os
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from util.smiles.char_dict import SmilesCharDictionary


def load_dataset(char_dict, smi_path):
    processed_dataset_path = str(Path(smi_path).with_suffix("")) + "_processed.smiles"
    if False and os.path.exists(processed_dataset_path):
        with open(processed_dataset_path, "r") as f:
            processed_dataset = f.readlines()

    else:
        with open(smi_path, "r") as f:
            dataset = f.read().splitlines()

        processed_dataset = list(filter(char_dict.allowed, dataset))
        with open(processed_dataset_path, "w") as f:
            f.write("\n".join(processed_dataset))

    return processed_dataset
