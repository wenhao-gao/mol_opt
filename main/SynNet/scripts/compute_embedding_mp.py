"""
Computes the molecular embeddings of the purchasable building blocks.
"""
import multiprocessing as mp
from scripts.compute_embedding import *
from rdkit import RDLogger
RDLogger.DisableLog('*')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default="gin",
                        help="Objective function to optimize")
    parser.add_argument("--ncpu", type=int, default=16,
                        help="Number of cpus")
    args = parser.parse_args()

    # define the path to which data will be saved
    path = '/pool001/whgao/data/synth_net/st_hb/'

    # load the building blocks
    data = pd.read_csv(path + 'enamine_us_matched.csv.gz', compression='gzip')['SMILES'].tolist()
    print('Total data: ', len(data))

    if args.feature == 'gin':
        with mp.Pool(processes=args.ncpu) as pool:
            embeddings = pool.map(gin_embedding, data)
    elif args.feature == 'fp_4096':
        with mp.Pool(processes=args.ncpu) as pool:
            embeddings = pool.map(fp_4096, data)
    elif args.feature == 'fp_2048':
        with mp.Pool(processes=args.ncpu) as pool:
            embeddings = pool.map(fp_2048, data)
    elif args.feature == 'fp_1024':
        with mp.Pool(processes=args.ncpu) as pool:
            embeddings = pool.map(fp_1024, data)
    elif args.feature == 'fp_512':
        with mp.Pool(processes=args.ncpu) as pool:
            embeddings = pool.map(fp_512, data)
    elif args.feature == 'fp_256':
        with mp.Pool(processes=args.ncpu) as pool:
            embeddings = pool.map(fp_256, data)
    elif args.feature == 'rdkit2d':
        with mp.Pool(processes=args.ncpu) as pool:
            embeddings = pool.map(rdkit2d_embedding, data)

    embedding = np.array(embeddings)

    # import ipdb; ipdb.set_trace(context=9)
    np.save(path + 'enamine_us_emb_' + args.feature + '.npy', embeddings)

    print('Finish!')
