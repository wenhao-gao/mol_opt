import csv
from functools import partial
import gzip
from pathlib import Path
from typing import Collection, Dict, Optional

from configargparse import ArgumentParser
from tqdm import tqdm

from molpal.objectives.base import Objective


class LookupObjective(Objective):
    """A LookupObjective calculates the objective function by looking the
    value up in an input file.

    Useful for retrospective studies.

    Attributes
    ----------
    self.data : Dict[str, Optional[float]]
        a dictionary containing the objective function value of each molecule

    Parameters
    ----------
    objective_config : str
        the configuration file for a LookupObjective
    **kwargs
        unused and addditional keyword arguments
    """
    def __init__(self, objective_config: str, minimize: bool = True, **kwargs):
        path, delimiter, title_line, smiles_col, score_col = parse_config(
            objective_config
        )

        if Path(path).suffix == ".gz":
            open_ = partial(gzip.open, mode="rt")
        else:
            open_ = open

        self.data = {}
        with open_(path) as fid:
            reader = csv.reader(fid, delimiter=delimiter)
            if title_line:
                next(fid)

            for row in tqdm(reader, desc="Building oracle", leave=False):
                key = row[smiles_col]
                val = row[score_col]
                try:
                    self.data[key] = float(val)
                except ValueError:
                    pass

        super().__init__(minimize=minimize)

    def forward(self, smis: Collection[str], *args, **kwargs) -> Dict[str, Optional[float]]:
        return {smi: self.c * self.data[smi] if smi in self.data else None for smi in smis}


def parse_config(config: str):
    """parse a LookupObjective configuration file

    Parameters
    ----------
    config : str
        the config file to parse

    Returns
    -------
    path : str
        the filepath of the lookup CSV file
    sep : str
        the CSV separator
    title_line : bool
        is there a title in in the lookup file?
    smiles_col : int
        the column containing the SMILES string in the lookup file
    data_col : int
        the column containing the desired data in the lookup file
    """
    parser = ArgumentParser()
    parser.add_argument("config", is_config_file=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--no-title-line", action="store_true", default=False)
    parser.add_argument("--smiles-col", type=int, default=0)
    parser.add_argument("--score-col", type=int, default=1)

    args = parser.parse_args(config)
    return (
        args.path,
        args.sep,
        not args.no_title_line,
        args.smiles_col,
        args.score_col,
    )
