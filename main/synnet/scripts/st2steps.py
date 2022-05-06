"""
Splits a synthetic tree into states and steps.
"""
import os
from tqdm import tqdm
from scipy import sparse
from syn_net.utils.data_utils import *
from syn_net.utils.prep_utils import organize



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numbersave", type=int, default=999999999999,
                        help="Save number")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Increase output verbosity")
    parser.add_argument("-e", "--targetembedding", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-o", "--outputembedding", type=str, default='gin',
                        help="Choose from ['fp_4096', 'fp_256', 'gin', 'rdkit2d']")
    parser.add_argument("-r", "--radius", type=int, default=2,
                        help="Radius for Morgan Fingerprint")
    parser.add_argument("-b", "--nbits", type=int, default=4096,
                        help="Number of Bits for Morgan Fingerprint")
    parser.add_argument("-d", "--datasettype", type=str, default='train',
                        help="Choose from ['train', 'valid', 'test']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    args = parser.parse_args()

    dataset_type = args.datasettype
    embedding = args.targetembedding
    path_st = '/pool001/whgao/data/synth_net/st_hb/st_' + dataset_type + '.json.gz'
    save_dir = '/pool001/whgao/data/synth_net/hb_' + embedding + '_' + str(args.radius) + '_' + str(args.nbits) + '_' + str(args.outputembedding) + '/'

    st_set = SyntheticTreeSet()
    st_set.load(path_st)
    print('Original length: ', len(st_set.sts))
    data = st_set.sts
    del st_set
    print('Working length: ', len(data))

    states = []
    steps = []

    num_save = args.numbersave
    idx = 0
    save_idx = 0
    for st in tqdm(data):
        try:
            state, step = organize(st, target_embedding=embedding, radius=args.radius, nBits=args.nbits, output_embedding=args.outputembedding)
        except Exception as e:
            print(e)
            continue
        states.append(state)
        steps.append(step)
        idx += 1
        if idx % num_save == 0:
            print('Saving......')
            states = sparse.vstack(states)
            steps = sparse.vstack(steps)
            sparse.save_npz(save_dir + 'states_' + str(save_idx) + '_' + dataset_type + '.npz', states)
            sparse.save_npz(save_dir + 'steps_' + str(save_idx) + '_' + dataset_type + '.npz', steps)
            save_idx += 1
            del states
            del steps
            states = []
            steps = []

    del data

    if len(steps) != 0:
        states = sparse.vstack(states)
        steps = sparse.vstack(steps)

        print('Saving......')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        sparse.save_npz(save_dir + 'states_' + str(save_idx) + '_' + dataset_type + '.npz', states)
        sparse.save_npz(save_dir + 'steps_' + str(save_idx) + '_' + dataset_type + '.npz', steps)

    print('Finish!')
