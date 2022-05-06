from myrdkit import Chem
import numpy as np
from collections import defaultdict

'''
This script looks at how many bond changes actually occur in forward reactions (in this dataset)
'''


def process_file(fpath):
    counter = defaultdict(int)
    with open(fpath, 'r') as fid_in, open(fpath + '.proc', 'w') as fid_out:
        for line in fid_in:
            edits = line.strip().split(' ')[1]
            counter[edits.count(';')] += 1
    # Convert to percentages
    total = sum(counter.values())
    counter = dict(counter)
    for k in counter.keys():
        counter[k] = float(counter[k]) / total
    print('Finished processing {}'.format(fpath))
    print(counter)

if __name__ == '__main__':

    # Process files
    process_file('../data/train.txt')
    process_file('../data/valid.txt')
    process_file('../data/test.txt')