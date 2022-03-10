import csv
from functools import partial
import gzip
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from matplotlib import ticker

def extract_smis(library, smiles_col=0, title_line=True) -> List[str]:
    if Path(library).suffix == '.gz':
        open_ = partial(gzip.open, mode='rt')
    else:
        open_ = open
    
    with open_(library) as fid:
        reader = csv.reader(fid)
        if title_line:
            next(reader)

        smis = []
        for row in reader:
            try:
                smis.append(row[smiles_col])
            except ValueError:
                continue

    return smis

def build_true_dict(true_csv, smiles_col: int = 0, score_col: int = 1,
                    title_line: bool = True,
                    maximize: bool = False) -> Dict[str, float]:
    if Path(true_csv).suffix == '.gz':
        open_ = partial(gzip.open, mode='rt')
    else:
        open_ = open
    
    c = 1 if maximize else -1

    with open_(true_csv) as fid:
        reader = csv.reader(fid)
        if title_line:
            next(reader)

        d_smi_score = {}
        for row in reader:
            try:
                d_smi_score[row[smiles_col]] = c * float(row[score_col])
            except ValueError:
                continue

    return d_smi_score

def read_scores(scores_csv: str) -> Tuple[Dict, Dict]:
    """read the scores contained in the file located at scores_csv"""
    scores = {}
    failures = {}
    with open(scores_csv) as fid:
        reader = csv.reader(fid)
        next(reader)
        for row in reader:
            try:
                scores[row[0]] = float(row[1])
            except:
                failures[row[0]] = None
    
    return scores, failures

def chunk(xs: Iterable, chunks: Iterable[int]):
    xs = iter(xs)
    xss = [list(islice(xs, 0, chunk)) for chunk in chunks]
    xss = [xs for xs in xss if len(xs) > 0]
    for x in xs:
        xss.append([x])

    return xss

def style_axis(ax):
    ax.set_xlabel(f'Molecules sampled')
    ax.set_xlim(left=0)
    # ax.set_ylim(bottom=0)#, top=100)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
    ax.xaxis.set_tick_params(rotation=30)
    ax.grid(True)

def abbreviate_k_or_M(x: float, pos) -> str:
    if x >= 1e6:
        return f'{x*1e-6:0.1f}M'
    if x >= 1e4:
        return f'{x*1e-3:0.0f}k'
    if x >= 1e3:
        return f'{x*1e-3:0.1f}k'

    return f'{x:0.0f}'
