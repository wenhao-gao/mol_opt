import argparse
import csv
from functools import partial
import gzip
from pathlib import Path
import pickle
import sys

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--scores-csv', required=True)
parser.add_argument('--smiles-col', type=int, default=0)
parser.add_argument('--score-col', type=int, default=1)
parser.add_argument('--no-title-line', action='store_true', default=False)

def main():
    args = parser.parse_args()
    title_line = not args.no_title_line

    if Path(args.scores_csv).suffix == '.gz':
        open_ = partial(gzip.open, mode='rt')
    else:
        open_ = open
    
    with open_(args.scores_csv) as fid:
        reader = csv.reader(fid)
        if title_line:
            next(reader)

        d_smi_score = {}
        for row in tqdm(reader):
            smi = row[args.smiles_col]
            try:
                score = float(row[args.score_col])
            except ValueError:
                continue
            d_smi_score[smi] = score

    pickle.dump(d_smi_score,
                open(str(Path(args.scores_csv).with_suffix('.pkl')), 'wb'))
                
if __name__ == "__main__":
    main()
