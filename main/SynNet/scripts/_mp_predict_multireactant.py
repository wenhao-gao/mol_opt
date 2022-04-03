"""
This file contains a function to decode a single synthetic tree.
"""
import pandas as pd
import numpy as np
from syn_net.utils.data_utils import ReactionSet
from syn_net.utils.predict_utils import synthetic_tree_decoder_multireactant, load_modules_from_checkpoint, mol_fp


# define some constants (here, for the Hartenfeller-Button test set)
nbits        = 4096
out_dim      = 256
rxn_template = 'hb'
featurize    = 'fp'
param_dir    = 'hb_fp_2_4096_256'
ncpu         = 1

# load the purchasable building block embeddings
bb_emb = np.load('/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_256.npy')

# define path to the reaction templates and purchasable building blocks
path_to_reaction_file   = f'/pool001/whgao/data/synth_net/st_{rxn_template}/reactions_{rxn_template}.json.gz'
path_to_building_blocks = f'/pool001/whgao/data/synth_net/st_{rxn_template}/enamine_us_matched.csv.gz'

# define paths to pretrained modules
param_path  = f'/home/whgao/synth_net/synth_net/params/{param_dir}/'
path_to_act = f'{param_path}act.ckpt'
path_to_rt1 = f'{param_path}rt1.ckpt'
path_to_rxn = f'{param_path}rxn.ckpt'
path_to_rt2 = f'{param_path}rt2.ckpt'

# load the purchasable building block SMILES to a dictionary
building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
bb_dict         = {building_blocks[i]: i for i in range(len(building_blocks))}

# load the reaction templates as a ReactionSet object
rxn_set = ReactionSet()
rxn_set.load(path_to_reaction_file)
rxns    = rxn_set.rxns

# load the pre-trained modules
act_net, rt1_net, rxn_net, rt2_net = load_modules_from_checkpoint(
    path_to_act=path_to_act,
    path_to_rt1=path_to_rt1,
    path_to_rxn=path_to_rxn,
    path_to_rt2=path_to_rt2,
    featurize=featurize,
    rxn_template=rxn_template,
    out_dim=out_dim,
    nbits=nbits,
    ncpu=ncpu,
)

def func(smi):
    """
    Generates the synthetic tree for the input molecular embedding.

    Args:
        smi (str): SMILES string corresponding to the molecule to decode.

    Returns:
        smi (str): SMILES for the final chemical node in the tree.
        similarity (float): Similarity measure between the final chemical node
            and the input molecule.
        tree (SyntheticTree): The generated synthetic tree.
    """
    emb = mol_fp(smi)
    try:
        smi, similarity, tree, action = synthetic_tree_decoder_multireactant(
            z_target=emb,
            building_blocks=building_blocks,
            bb_dict=bb_dict,
            reaction_templates=rxns,
            mol_embedder=mol_fp,
            action_net=act_net,
            reactant1_net=rt1_net,
            rxn_net=rxn_net,
            reactant2_net=rt2_net,
            bb_emb=bb_emb,
            rxn_template=rxn_template,
            n_bits=nbits,
            beam_width=3,
            max_step=15)
    except Exception as e:
        print(e)
        action = -1

    if action != 3:
        return None, 0, None
    else:
        return smi, similarity, tree
