import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import rdkit
import argparse
from fuseprop import merge_rationales, unique_rationales
from multiprocessing import Pool

def join_func(tup):
    x, ylist = tup
    joined_list = [(x, y, merge_rationales(x, y)) for y in ylist]
    return [(x,y,z) for x,y,z in joined_list if len(z) > 0]


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--rationale1', required=True)
    parser.add_argument('--rationale2', required=True)
    parser.add_argument('--ncpu', type=int, default=15)
    args = parser.parse_args()

    with open(args.rationale1) as f:
        rationale1 = [line.split()[1] for line in f] 

    with open(args.rationale2) as f:
        rationale2 = [line.split()[1] for line in f] 

    rationale1 = unique_rationales(rationale1)
    rationale2 = unique_rationales(rationale2)
    print('unique rationales:', len(rationale1), len(rationale2), file=sys.stderr)
    
    pool = Pool(args.ncpu)

    batches = [(x, rationale2) for x in rationale1]
    all_joined = pool.map(join_func, batches)
    all_joined = [tup for tlist in all_joined for tup in tlist]

    for x, y, joined in all_joined:
        for z in joined:
            print(x + '.' + y, z)
