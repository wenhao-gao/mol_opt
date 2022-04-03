"""
Filters out purchasable building blocks which don't match one of the 3 reaction
templates used for unit testing.
"""
import pandas as pd
from tqdm import tqdm
from syn_net.utils.data_utils import *


if __name__ == '__main__':
    r_path = './data/ref/rxns_hb.json.gz'
    bb_path = '/home/whgao/scGen/synth_net/data/enamine_us.csv.gz'
    r_set = ReactionSet()
    r_set.load(r_path)
    matched_mols = set()
    for r in tqdm(r_set.rxns):
        for a_list in r.available_reactants:
            matched_mols = matched_mols | set(a_list)

    original_mols = pd.read_csv(bb_path, compression='gzip')['SMILES'].tolist()

    print('Total building blocks number:', len(original_mols))
    print('Matched building blocks number:', len(matched_mols))

    df = pd.DataFrame({'SMILES': list(matched_mols)})
    df.to_csv('./data/building_blocks_matched.csv.gz', compression='gzip')
