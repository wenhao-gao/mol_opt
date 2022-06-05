"""
This file contains a function to decode a single synthetic tree 
"""
import pandas as pd
import numpy as np
import rdkit
from tqdm import tqdm
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from main.synnet.syn_net.utils.data_utils import Reaction, ReactionSet, SyntheticTree, SyntheticTreeSet
from sklearn.neighbors import BallTree

from tdc.chem_utils import MolConvert
from main.synnet.syn_net.models.mlp import MLP
import os
path_here = os.path.dirname(os.path.realpath(__file__))
path_main = '/'.join(path_here.split('/')[:-1])

nbits = 4096
out_dim = 256
rxn_template = 'hb'
featurize = 'fp'
param_dir = 'hb_fp_2_4096_256'

def can_react(state, rxns):
    mol1 = state.pop()
    mol2 = state.pop()
    reaction_mask = [int(rxn.run_reaction([mol1, mol2]) is not None) for rxn in rxns]
    return sum(reaction_mask), reaction_mask

def get_action_mask(state, rxns):
    # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
    if len(state) == 0:
        return np.array([1, 0, 0, 0])
    elif len(state) == 1:
        return np.array([1, 1, 0, 1])
    elif len(state) == 2:
        can_react_, _ = can_react(state, rxns)
        if can_react_:
            return np.array([0, 1, 1, 0])
        else:
            return np.array([0, 1, 0, 0])
    else:
        raise ValueError('Problem with state.')

def get_reaction_mask(smi, rxns):
    # Return all available reaction templates
    # List of available building blocks if 2
    # Exclude the case of len(available_list) == 0
    reaction_mask = [int(rxn.is_reactant(smi)) for rxn in rxns]

    if sum(reaction_mask) == 0:
        return None, None
    available_list = []
    mol = rdkit.Chem.MolFromSmiles(smi)
    for i, rxn in enumerate(rxns):
        if reaction_mask[i] and rxn.num_reactant == 2:

            if rxn.is_reactant_first(mol):
                available_list.append(rxn.available_reactants[1])
            elif rxn.is_reactant_second(mol):
                available_list.append(rxn.available_reactants[0])
            else:
                raise ValueError('Check the reactants')

            if len(available_list[-1]) == 0:
                reaction_mask[i] = 0

        else:
            available_list.append([])

    return reaction_mask, available_list

def one_hot_encoder(dim, space):
    vec = np.zeros((1, space))
    vec[0, dim] = 1
    return vec

def mol_fp(smi, _radius=2, _nBits=4096):
    if smi is None:
        return np.zeros(_nBits)
    else:
        mol = Chem.MolFromSmiles(smi)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, _radius, _nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape((1, -1))

def cosine_distance(v1, v2, eps=1e-15):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1, ord=2) * np.linalg.norm(v2, ord=2) + eps)

bb_emb = np.load(os.path.join(path_main, 'data/enamine_us_emb_fp_256.npy'))
kdtree = BallTree(bb_emb, metric=cosine_distance)
def nn_search(_e, _tree=kdtree, _k=1):
    dist, ind = _tree.query(_e, k=_k)
    return dist[0][0], ind[0][0]

def set_embedding(z_target, state, _mol_embedding=mol_fp):
    if len(state) == 0:
        return np.concatenate([np.zeros((1, 2 * nbits)), z_target], axis=1)
    else:
        e1 = _mol_embedding(state[0])
        if len(state) == 1:
            e2 = np.zeros((1, nbits))
        else:
            e2 = _mol_embedding(state[1])
        return np.concatenate([e1, e2, z_target], axis=1)

def synthetic_tree_decoder(z_target, 
                            building_blocks, 
                            bb_dict,
                            reaction_templates, 
                            action_net, 
                            reactant1_net, 
                            rxn_net, 
                            reactant2_net, 
                            max_step=15):
    # Initialization
    tree = SyntheticTree()
    mol_recent = None

    # Start iteration
    # try:
    for i in range(max_step):
        # Encode current state
        # from ipdb import set_trace; set_trace(context=11)
        state = tree.get_state() # a set
        z_state = set_embedding(z_target, state, mol_fp)

        # Predict action type, masked selection
        # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
        action_proba = action_net(torch.Tensor(z_state)) 
        action_proba = action_proba.squeeze().detach().numpy() + 1e-10
        action_mask = get_action_mask(tree.get_state(), reaction_templates)
        act = np.argmax(action_proba * action_mask)

        # z_mol1 = reactant1_net(torch.Tensor(np.concatenate([z_state, one_hot_encoder(act, 4)], axis=1)))
        z_mol1 = reactant1_net(torch.Tensor(z_state))
        z_mol1 = z_mol1.detach().numpy()

        # Select first molecule
        if act == 3:
            # End
            break
        elif act == 0:
            # Add
            dist, ind = nn_search(z_mol1)
            mol1 = building_blocks[ind]
        else:
            # Expand or Merge
            mol1 = mol_recent

        # z_mol1 = get_mol_embedding(mol1, mol_embedder)
        z_mol1 = mol_fp(mol1)

        # import ipdb; ipdb.set_trace()

        # Select reaction
        reaction_proba = rxn_net(torch.Tensor(np.concatenate([z_state, z_mol1], axis=1)))
        reaction_proba = reaction_proba.squeeze().detach().numpy() + 1e-10

        if act != 2:
            reaction_mask, available_list = get_reaction_mask(mol1, reaction_templates)
        else:
            _, reaction_mask = can_react(tree.get_state(), reaction_templates)
            available_list = [[] for rxn in reaction_templates]

        if reaction_mask is None:
            if len(state) == 1:
                act = 3
                break
            else:
                break

        rxn_id = np.argmax(reaction_proba * reaction_mask)
        rxn = reaction_templates[rxn_id]

        if rxn.num_reactant == 2:
            # Select second molecule
            if act == 2:
                # Merge
                temp = set(state) - set([mol1])
                mol2 = temp.pop()
            else:
                # Add or Expand
                if rxn_template == 'hb':
                    z_mol2 = reactant2_net(torch.Tensor(np.concatenate([z_state, z_mol1, one_hot_encoder(rxn_id, 91)], axis=1)))
                elif rxn_template == 'pis':
                    z_mol2 = reactant2_net(torch.Tensor(np.concatenate([z_state, z_mol1, one_hot_encoder(rxn_id, 4700)], axis=1)))
                z_mol2 = z_mol2.detach().numpy()
                available = available_list[rxn_id]
                available = [bb_dict[available[i]] for i in range(len(available))]
                temp_emb = bb_emb[available]
                available_tree = BallTree(temp_emb, metric=cosine_distance)
                dist, ind = nn_search(z_mol2, _tree=available_tree)
                mol2 = building_blocks[available[ind]]
        else:
            mol2 = None

        # Run reaction
        mol_product = rxn.run_reaction([mol1, mol2])
        if mol_product is None or Chem.MolFromSmiles(mol_product) is None:
            act = 3
            break

        # Update
        tree.update(act, int(rxn_id), mol1, mol2, mol_product)
        mol_recent = mol_product

    if act != 3:
        tree = tree
    else:
        tree.update(act, None, None, None, None)

    return tree, act


path_to_reaction_file = os.path.join(path_main, f"data/reactions_{rxn_template}.json.gz")
path_to_building_blocks = os.path.join(path_main, "data/enamine_us_matched.csv.gz")

param_path = os.path.join(path_main, f"syn_net/params/{param_dir}/")
path_to_act = os.path.join(param_path, 'act.ckpt')
path_to_rt1 = os.path.join(param_path, 'rt1.ckpt')
path_to_rxn = os.path.join(param_path, 'rxn.ckpt')
path_to_rt2 = os.path.join(param_path, 'rt2.ckpt')

building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
bb_dict = {building_blocks[i]: i for i in range(len(building_blocks))}

rxn_set = ReactionSet()
rxn_set.load(path_to_reaction_file)
rxns = rxn_set.rxns

ncpu = 16

act_net = MLP.load_from_checkpoint(path_to_act,
                    input_dim=int(3 * nbits),
                    output_dim=4,
                    hidden_dim=1000,
                    num_layers=5,
                    dropout=0.5,
                    num_dropout_layers=1,
                    task='classification',
                    loss='cross_entropy',
                    valid_loss='accuracy',
                    optimizer='adam',
                    learning_rate=1e-4,
                    ncpu=ncpu)

rt1_net = MLP.load_from_checkpoint(path_to_rt1,
                    input_dim=int(3 * nbits),
                    output_dim=out_dim,
                    hidden_dim=1200,
                    num_layers=5,
                    dropout=0.5,
                    num_dropout_layers=1,
                    task='regression',
                    loss='mse',
                    valid_loss='mse',
                    optimizer='adam',
                    learning_rate=1e-4,
                    ncpu=ncpu)


rxn_net = MLP.load_from_checkpoint(path_to_rxn,
                    input_dim=int(4 * nbits),
                    output_dim=91,
                    hidden_dim=3000,
                    num_layers=5,
                    dropout=0.5,
                    num_dropout_layers=1,
                    task='classification',
                    loss='cross_entropy',
                    valid_loss='accuracy',
                    optimizer='adam',
                    learning_rate=1e-4,
                    ncpu=ncpu)

rt2_net = MLP.load_from_checkpoint(path_to_rt2,
                    input_dim=int(4 * nbits + 91),
                    output_dim=out_dim,
                    hidden_dim=3000,
                    num_layers=5,
                    dropout=0.5,
                    num_dropout_layers=1,
                    task='regression',
                    loss='mse',
                    valid_loss='mse',
                    optimizer='adam',
                    learning_rate=1e-4,
                    ncpu=ncpu)

act_net.eval()
rt1_net.eval()
rxn_net.eval()
rt2_net.eval()

def _tanimoto_similarity(fp1, fp2):
    return np.sum(fp1 * fp2) / (np.sum(fp1) + np.sum(fp2) - np.sum(fp1 * fp2))

def tanimoto_similarity(target_fp, smis):
    fps = [mol_fp(smi, 2, 4096) for smi in smis]
    return [_tanimoto_similarity(target_fp, fp) for fp in fps]

def func(emb):
    emb = emb.reshape((1, -1))
    try:
        tree, action = synthetic_tree_decoder(emb, building_blocks, bb_dict, rxns, act_net, rt1_net, rxn_net, rt2_net, max_step=30)
    except Exception as e:
        print(e)
        action = -1
    
    if action != 3:
        return None, None
    else:
        scores = np.array(tanimoto_similarity(emb, [node.smiles for node in tree.chemicals]))
        max_score_idx = np.where(scores == np.max(scores))[0][0]
        return tree.chemicals[max_score_idx].smiles, tree
