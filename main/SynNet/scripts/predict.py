"""
This file contains the code to decode synthetic trees using a greedy search at
every sampling step.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
from syn_net.utils.data_utils import ReactionSet, SyntheticTreeSet
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from syn_net.utils.predict_utils import mol_fp, get_mol_embedding
from syn_net.utils.predict_utils import synthetic_tree_decoder, load_modules_from_checkpoint

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featurize", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    parser.add_argument("-v", "--version", type=int, default=0,
                        help="Version")
    parser.add_argument("--param", type=str, default='hb_fp_2_4096',
                        help="Name of directory with parameters in it.")
    parser.add_argument("--radius", type=int, default=2,
                        help="Radius for Morgan fingerprint.")
    parser.add_argument("--nbits", type=int, default=4096,
                        help="Number of Bits for Morgan fingerprint.")
    parser.add_argument("--out_dim", type=int, default=300,
                        help="Output dimension.")
    parser.add_argument("--ncpu", type=int, default=16,
                        help="Number of cpus")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("-n", "--num", type=int, default=-1,
                        help="Number of molecules to decode.")
    parser.add_argument("-d", "--data", type=str, default='test',
                        help="Choose from ['train', 'valid', 'test']")
    args = parser.parse_args()

    # define model to use for molecular embedding
    readout = AvgPooling()
    model_type = 'gin_supervised_contextpred'
    device = 'cuda:0'
    mol_embedder = load_pretrained(model_type).to(device)
    mol_embedder.eval()

    # load the purchasable building block embeddings
    bb_emb = np.load('/pool001/whgao/data/synth_net/st_' + args.rxn_template + '/enamine_us_emb.npy')

    # define path to the reaction templates and purchasable building blocks
    path_to_reaction_file   = ('/pool001/whgao/data/synth_net/st_' + args.rxn_template
                               + '/reactions_' + args.rxn_template + '.json.gz')
    path_to_building_blocks = ('/pool001/whgao/data/synth_net/st_' + args.rxn_template
                               + '/enamine_us_matched.csv.gz')

    # define paths to pretrained modules
    param_path = '/home/whgao/scGen/synth_net/synth_net/params/' + args.param + '/'
    path_to_act = param_path + 'act.ckpt'
    path_to_rt1 = param_path + 'rt1.ckpt'
    path_to_rxn = param_path + 'rxn.ckpt'
    path_to_rt2 = param_path + 'rt2.ckpt'

    np.random.seed(6)

    # load the purchasable building block SMILES to a dictionary
    building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
    bb_dict = {building_blocks[i]: i for i in range(len(building_blocks))}

    # load the reaction templates as a ReactionSet object
    rxn_set = ReactionSet()
    rxn_set.load(path_to_reaction_file)
    rxns = rxn_set.rxns

    # load the pre-trained modules
    act_net, rt1_net, rxn_net, rt2_net = load_modules_from_checkpoint(
        path_to_act=path_to_act,
        path_to_rt1=path_to_rt1,
        path_to_rxn=path_to_rxn,
        path_to_rt2=path_to_rt2,
        featurize=args.featurize,
        rxn_template=args.rxn_template,
        out_dim=args.out_dim,
        nbits=args.nbits,
        ncpu=args.ncpu,
    )

    def decode_one_molecule(query_smi):
        """
        Generate a synthetic tree from a given query SMILES.

        Args:
            query_smi (str): SMILES for molecule to decode.

        Returns:
            tree (SyntheticTree): The final synthetic tree
            act (int): The final action (to know if the tree was "properly" terminated)
        """
        if args.featurize == 'fp':
            z_target = mol_fp(query_smi, args.radius, args.nbits)
        elif args.featurize == 'gin':
            z_target = get_mol_embedding(query_smi)
        tree, action = synthetic_tree_decoder(z_target,
                                              building_blocks,
                                              bb_dict,
                                              rxns,
                                              mol_embedder,
                                              act_net,
                                              rt1_net,
                                              rxn_net,
                                              rt2_net,
                                              bb_emb=bb_emb,
                                              rxn_template=args.rxn_template,
                                              n_bits=args.nbits,
                                              max_step=15)
        return tree, action


    path_to_data = '/pool001/whgao/data/synth_net/st_' + args.rxn_template + '/st_' + args.data +'.json.gz'
    print('Reading data from ', path_to_data)
    sts = SyntheticTreeSet()
    sts.load(path_to_data)
    query_smis = [st.root.smiles for st in sts.sts]
    if args.num == -1:
        pass
    else:
        query_smis = query_smis[:args.num]

    output_smis = []
    similaritys = []
    trees = []
    num_finish = 0
    num_unfinish = 0

    print('Start to decode!')
    for smi in tqdm(query_smis):

        try:
            tree, action = decode_one_molecule(smi)
        except Exception as e:
            print(e)
            action = -1
            tree = None

        if action != 3:
            num_unfinish += 1
            output_smis.append(None)
            similaritys.append(None)
            trees.append(None)
        else:
            num_finish += 1
            output_smis.append(tree.root.smiles)
            ms = [Chem.MolFromSmiles(sm) for sm in [smi, tree.root.smiles]]
            fps = [Chem.RDKFingerprint(x) for x in ms]
            similaritys.append(DataStructs.FingerprintSimilarity(fps[0],fps[1]))
            trees.append(tree)

    print('Saving ......')
    save_path = '../results/' + args.rxn_template + '_' + args.featurize + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame({'query SMILES': query_smis, 'decode SMILES': output_smis, 'similarity': similaritys})
    print("mean similarities", df['similarity'].mean(), df['similarity'].std())
    print("NAs", df.isna().sum())
    df.to_csv(save_path + 'decode_result_' + args.data + '.csv.gz', compression='gzip', index=False)

    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save(save_path + 'decoded_st_' + args.data + '.json.gz')

    print('Finish!')
