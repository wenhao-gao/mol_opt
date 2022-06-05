"""
This file contains various utils for decoding synthetic trees.
"""
import numpy as np
import rdkit
from tqdm import tqdm
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import BallTree
# from dgl.nn.pytorch.glob import AvgPooling
# from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from syn_net.models.mlp import MLP
from syn_net.utils.data_utils import SyntheticTree


# general functions
def can_react(state, rxns):
    """
    Determines if two molecules can react using any of the input reactions.

    Args:
        state (np.ndarray): The current state in the synthetic tree.
        rxns (list of Reaction objects): Contains available reaction templates.

    Returns:
        np.ndarray: The sum of the reaction mask tells us how many reactions are
             viable for the two molecules.
        np.ndarray: The reaction mask, which masks out reactions which are not
            viable for the two molecules.
    """
    mol1 = state.pop()
    mol2 = state.pop()
    reaction_mask = [int(rxn.run_reaction([mol1, mol2]) is not None) for rxn in rxns]
    return sum(reaction_mask), reaction_mask

def get_action_mask(state, rxns):
    """
    Determines which actions can apply to a given state in the synthetic tree
    and returns a mask for which actions can apply.

    Args:
        state (np.ndarray): The current state in the synthetic tree.
        rxns (list of Reaction objects): Contains available reaction templates.

    Raises:
        ValueError: There is an issue with the input state.

    Returns:
        np.ndarray: The action mask. Masks out unviable actions from the current
            state using 0s, with 1s at the positions corresponding to viable
            actions.
    """
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
    """
    Determines which reaction templates can apply to the input molecule.

    Args:
        smi (str): The SMILES string corresponding to the molecule in question.
        rxns (list of Reaction objects): Contains available reaction templates.

    Raises:
        ValueError: There is an issue with the reactants in the reaction.

    Returns:
        reaction_mask (list of ints, or None): The reaction template mask. Masks
            out reaction templates which are not viable for the input molecule.
            If there are no viable reaction templates identified, is simply None.
        available_list (list of lists, or None): Contains available reactants if
            at least one viable reaction template is identified. Else is simply
            None.
    """
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

# def graph_construction_and_featurization(smiles):
#     """
#     Constructs graphs from SMILES and featurizes them.

#     Args:
#         smiles (list of str): Contains SMILES of molecules to embed.

#     Returns:
#         graphs (list of DGLGraph): List of graphs constructed and featurized.
#         success (list of bool): Indicators for whether the SMILES string can be
#             parsed by RDKit.
#     """
#     graphs = []
#     success = []
#     for smi in tqdm(smiles):
#         try:
#             mol = Chem.MolFromSmiles(smi)
#             if mol is None:
#                 success.append(False)
#                 continue
#             g = mol_to_bigraph(mol, add_self_loop=True,
#                                 node_featurizer=PretrainAtomFeaturizer(),
#                                 edge_featurizer=PretrainBondFeaturizer(),
#                                 canonical_atom_order=False)
#             graphs.append(g)
#             success.append(True)
#         except:
#             success.append(False)

#     return graphs, success

def one_hot_encoder(dim, space):
    """
    Create a one-hot encoded vector of length=`space`, with a non-zero element
    at the index given by `dim`.

    Args:
        dim (int): Non-zero bit in one-hot vector.
        space (int): Length of one-hot encoded vector.

    Returns:
        vec (np.ndarray): One-hot encoded vector.
    """
    vec = np.zeros((1, space))
    vec[0, dim] = 1
    return vec

# def get_mol_embedding(smi, model, device='cpu', readout=AvgPooling()):
#     """
#     Computes the molecular graph embedding for the input SMILES.

#     Args:
#         smi (str): SMILES of molecule to embed.
#         model (dgllife.model, optional): Pre-trained NN model to use for
#             computing the embedding.
#         device (str, optional): Indicates the device to run on. Defaults to 'cpu'.
#         readout (dgl.nn.pytorch.glob, optional): Readout function to use for
#             computing the graph embedding. Defaults to readout.

#     Returns:
#         torch.Tensor: Learned embedding for the input molecule.
#     """
#     mol = Chem.MolFromSmiles(smi)
#     g = mol_to_bigraph(mol, add_self_loop=True,
#                        node_featurizer=PretrainAtomFeaturizer(),
#                        edge_featurizer=PretrainBondFeaturizer(),
#                        canonical_atom_order=False)
#     bg = g.to(device)
#     nfeats = [bg.ndata.pop('atomic_number').to(device),
#               bg.ndata.pop('chirality_type').to(device)]
#     efeats = [bg.edata.pop('bond_type').to(device),
#               bg.edata.pop('bond_direction_type').to(device)]
#     with torch.no_grad():
#         node_repr = model(bg, nfeats, efeats)
#     return readout(bg, node_repr).detach().cpu().numpy()[0]

def mol_fp(smi, _radius=2, _nBits=4096):
    """
    Computes the Morgan fingerprint for the input SMILES.

    Args:
        smi (str): SMILES for molecule to compute fingerprint for.
        _radius (int, optional): Fingerprint radius to use. Defaults to 2.
        _nBits (int, optional): Length of fingerprint. Defaults to 1024.

    Returns:
        features (np.ndarray): For valid SMILES, this is the fingerprint.
            Otherwise, if the input SMILES is bad, this will be a zero vector.
    """
    if smi is None:
        return np.zeros(_nBits)
    else:
        mol = Chem.MolFromSmiles(smi)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, _radius, _nBits)
        return np.array(features_vec)

def cosine_distance(v1, v2, eps=1e-15):
    """
    Computes the cosine similarity between two vectors.

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.
        eps (float, optional): Small value, for numerical stability. Defaults
            to 1e-15.

    Returns:
        float: The cosine similarity.
    """
    return (1 - np.dot(v1, v2)
            / (np.linalg.norm(v1, ord=2) * np.linalg.norm(v2, ord=2) + eps))

def ce_distance(y, y_pred, eps=1e-15):
    """
    Computes the cross-entropy between two vectors.

    Args:
        y (np.ndarray): First vector.
        y_pred (np.ndarray): Second vector.
        eps (float, optional): Small value, for numerical stability. Defaults
            to 1e-15.

    Returns:
        float: The cross-entropy.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return - np.sum((y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))


def nn_search(_e, _tree, _k=1):
    """
    Conducts a nearest neighbor search to find the molecule from the tree most
    simimilar to the input embedding.

    Args:
        _e (np.ndarray): A specific point in the dataset.
        _tree (sklearn.neighbors._kd_tree.KDTree, optional): A k-d tree.
        _k (int, optional): Indicates how many nearest neighbors to get.
            Defaults to 1.

    Returns:
        float: The distance to the nearest neighbor.
        int: The indices of the nearest neighbor.
    """
    dist, ind = _tree.query(_e, k=_k)
    return dist[0][0], ind[0][0]

# def graph_construction_and_featurization(smiles):
#     """
#     Constructs graphs from SMILES and featurizes them.

#     Args:
#         smiles (list of str): SMILES of molecules, for embedding computation.

#     Returns:
#         graphs (list of DGLGraph): List of graphs constructed and featurized.
#         success (list of bool): Indicators for whether the SMILES string can be
#             parsed by RDKit.
#     """
#     graphs  = []
#     success = []
#     for smi in tqdm(smiles):
#         try:
#             mol = Chem.MolFromSmiles(smi)
#             if mol is None:
#                 success.append(False)
#                 continue
#             g = mol_to_bigraph(mol, add_self_loop=True,
#                                node_featurizer=PretrainAtomFeaturizer(),
#                                edge_featurizer=PretrainBondFeaturizer(),
#                                canonical_atom_order=False)
#             graphs.append(g)
#             success.append(True)
#         except:
#             success.append(False)

#     return graphs, success

def set_embedding(z_target, state, nbits, _mol_embedding=mol_fp):
    """
    Computes embeddings for all molecules in the input space.

    Args:
        z_target (np.ndarray): Embedding for the target molecule.
        state (list): Contains molecules in the current state, if not the
            initial state.
        nbits (int): Length of fingerprint.
        _mol_embedding (Callable, optional): Function to use for computing the
            embeddings of the first and second molecules in the state.

    Returns:
        np.ndarray: Embedding consisting of the concatenation of the target
            molecule with the current molecules (if available) in the input state.
    """
    if len(state) == 0:
        return np.concatenate([np.zeros((1, 2 * nbits)), z_target], axis=1)
    else:
        e1 = np.expand_dims(_mol_embedding(state[0]), axis=0)
        if len(state) == 1:
            e2 = np.zeros((1, nbits))
        else:
            e2 = _mol_embedding(state[1])
        return np.concatenate([e1, e2, z_target], axis=1)

def synthetic_tree_decoder(z_target,
                           building_blocks,
                           bb_dict,
                           reaction_templates,
                           mol_embedder,
                           action_net,
                           reactant1_net,
                           rxn_net,
                           reactant2_net,
                           bb_emb,
                           rxn_template,
                           n_bits,
                           max_step=15):
    """
    Computes the synthetic tree given an input molecule embedding, using the
    Action, Reaction, Reactant1, and Reactant2 networks and a greedy search

    Args:
        z_target (np.ndarray): Embedding for the target molecule
        building_blocks (list of str): Contains available building blocks
        bb_dict (dict): Building block dictionary
        reaction_templates (list of Reactions): Contains reaction templates
        mol_embedder (dgllife.model.gnn.gin.GIN): GNN to use for obtaining
            molecular embeddings
        action_net (synth_net.models.mlp.MLP): The action network
        reactant1_net (synth_net.models.mlp.MLP): The reactant1 network
        rxn_net (synth_net.models.mlp.MLP): The reaction network
        reactant2_net (synth_net.models.mlp.MLP): The reactant2 network
        bb_emb (list): Contains purchasable building block embeddings.
        rxn_template (str): Specifies the set of reaction templates to use.
        n_bits (int): Length of fingerprint.
        max_step (int, optional): Maximum number of steps to include in the
            synthetic tree

    Returns:
        tree (SyntheticTree): The final synthetic tree.
        act (int): The final action (to know if the tree was "properly"
            terminated).
    """
    # Initialization
    print('synthetic_tree_decoder: 0')
    tree       = SyntheticTree()
    kdtree     = BallTree(bb_emb, metric=cosine_distance)
    mol_recent = None
    print('synthetic_tree_decoder: 1')

    # print("z_target", z_target.shape) ### (1,4096)


    # Start iteration
    # try:
    for i in tqdm(range(max_step)):
        # Encode current state
        print('synthetic_tree_decoder: for loop ', i)

        state = tree.get_state() # a set
        z_state = set_embedding(z_target, state, nbits=n_bits, _mol_embedding=mol_fp)  #### z_state [1,12288]
        # print("z_target, z_state", z_target.shape, z_state.shape)
        # Predict action type, masked selection
        # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
        action_proba = action_net(torch.Tensor(z_state))
        action_proba = action_proba.squeeze().detach().numpy() + 1e-10
        action_mask = get_action_mask(tree.get_state(), reaction_templates)
        act = np.argmax(action_proba * action_mask)

        # reactant1_net_input = torch.Tensor(
        #     np.concatenate([z_state, one_hot_encoder(act, 4)], axis=1)
        # ) ### original 
        reactant1_net_input = torch.Tensor(z_state) #### debug mode 
        z_mol1 = reactant1_net(reactant1_net_input)
        z_mol1 = z_mol1.detach().numpy()

        # Select first molecule
        if act == 3:
            # End
            break
        elif act == 0:
            # Add
            dist, ind = nn_search(z_mol1, _tree=kdtree)
            mol1 = building_blocks[ind]
        else:
            # Expand or Merge
            mol1 = mol_recent

        z_mol1 = mol_fp(mol1)

        z_mol1 = z_mol1.reshape(1,-1) #### [4096] -> [1,4096] #### debug 
        # Select reaction
        rxn_net_input  = torch.Tensor(np.concatenate([z_state, z_mol1], axis=1))
        reaction_proba = rxn_net(rxn_net_input)
        reaction_proba = reaction_proba.squeeze().detach().numpy() + 1e-10

        if act != 2:
            reaction_mask, available_list = get_reaction_mask(smi=mol1,
                                                              rxns=reaction_templates)
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
                    num_rxns = 91
                elif rxn_template == 'pis':
                    num_rxns = 4700
                reactant2_net_input = torch.Tensor(
                    np.concatenate([z_state, z_mol1, one_hot_encoder(rxn_id, num_rxns)],
                                   axis=1)
                )
                z_mol2 = reactant2_net(reactant2_net_input)
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
            if len(tree.get_state()) == 1:
                act = 3
                break
            else:
                break

        # Update
        tree.update(act, int(rxn_id), mol1, mol2, mol_product)
        mol_recent = mol_product

    if act != 3:
        tree = tree
    else:
        tree.update(act, None, None, None, None)

    return tree, act

def load_modules_from_checkpoint(path_to_act, path_to_rt1, path_to_rxn, path_to_rt2, featurize, rxn_template, out_dim, nbits, ncpu):

    if rxn_template == 'unittest':

        act_net = MLP.load_from_checkpoint(path_to_act,
                                           input_dim=int(3 * nbits),
                                           output_dim=4,
                                           hidden_dim=100,
                                           num_layers=3,
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
                                           hidden_dim=100,
                                           num_layers=3,
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
                                           output_dim=3,
                                           hidden_dim=100,
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
                                           input_dim=int(4 * nbits + 3),
                                           output_dim=out_dim,
                                           hidden_dim=100,
                                           num_layers=3,
                                           dropout=0.5,
                                           num_dropout_layers=1,
                                           task='regression',
                                           loss='mse',
                                           valid_loss='mse',
                                           optimizer='adam',
                                           learning_rate=1e-4,
                                           ncpu=ncpu)
    elif featurize == 'fp':

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

        if rxn_template == 'hb':

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

        elif rxn_template == 'pis':

            rxn_net = MLP.load_from_checkpoint(path_to_rxn,
                                               input_dim=int(4 * nbits),
                                               output_dim=4700,
                                               hidden_dim=4500,
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
                                               input_dim=int(4 * nbits + 4700),
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

    elif featurize == 'gin':

        act_net = MLP.load_from_checkpoint(path_to_act,
                                           input_dim=int(2 * nbits + out_dim),
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
                                           input_dim=int(2 * nbits + out_dim),
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

        if rxn_template == 'hb':

            rxn_net = MLP.load_from_checkpoint(path_to_rxn,
                                               input_dim=int(3 * nbits + out_dim),
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
                                               input_dim=int(3 * nbits + out_dim + 91),
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

        elif rxn_template == 'pis':

            rxn_net = MLP.load_from_checkpoint(path_to_rxn,
                                               input_dim=int(3 * nbits + out_dim),
                                               output_dim=4700,
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
                                               input_dim=int(3 * nbits + out_dim + 4700),
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

    return act_net, rt1_net, rxn_net, rt2_net

def _tanimoto_similarity(fp1, fp2):
    """
    Returns the Tanimoto similarity between two molecular fingerprints.

    Args:
        fp1 (np.ndarray): Molecular fingerprint 1.
        fp2 (np.ndarray): Molecular fingerprint 2.

    Returns:
        np.float: Tanimoto similarity.
    """
    return np.sum(fp1 * fp2) / (np.sum(fp1) + np.sum(fp2) - np.sum(fp1 * fp2))

def tanimoto_similarity(target_fp, smis):
    """
    Returns the Tanimoto similarities between a target fingerprint and molecules
    in an input list of SMILES.

    Args:
        target_fp (np.ndarray): Contains the reference (target) fingerprint.
        smis (list of str): Contains SMILES to compute similarity to.

    Returns:
        list of np.ndarray: Contains Tanimoto similarities.
    """
    fps = [mol_fp(smi, 2, 4096) for smi in smis]
    return [_tanimoto_similarity(target_fp, fp) for fp in fps]


# functions used in the *_multireactant.py
def nn_search_rt1(_e, _tree, _k=1):
    dist, ind = _tree.query(_e, k=_k)
    return dist[0], ind[0]

def synthetic_tree_decoder_rt1(z_target,
                                building_blocks,
                                bb_dict,
                                reaction_templates,
                                mol_embedder,
                                action_net,
                                reactant1_net,
                                rxn_net,
                                reactant2_net,
                                bb_emb,
                                rxn_template,
                                n_bits,
                                max_step=15,
                                rt1_index=0):
    """
    Computes the synthetic tree given an input molecule embedding, using the
    Action, Reaction, Reactant1, and Reactant2 networks and a greedy search.

    Args:
        z_target (np.ndarray): Embedding for the target molecule
        building_blocks (list of str): Contains available building blocks
        bb_dict (dict): Building block dictionary
        reaction_templates (list of Reactions): Contains reaction templates
        mol_embedder (dgllife.model.gnn.gin.GIN): GNN to use for obtaining
            molecular embeddings
        action_net (synth_net.models.mlp.MLP): The action network
        reactant1_net (synth_net.models.mlp.MLP): The reactant1 network
        rxn_net (synth_net.models.mlp.MLP): The reaction network
        reactant2_net (synth_net.models.mlp.MLP): The reactant2 network
        bb_emb (list): Contains purchasable building block embeddings.
        rxn_template (str): Specifies the set of reaction templates to use.
        n_bits (int): Length of fingerprint.
        beam_width (int): The beam width to use for Reactant 1 search. Defaults
            to 3.
        max_step (int, optional): Maximum number of steps to include in the
            synthetic tree
        rt1_index (int, optional): Index for molecule in the building blocks
            corresponding to reactant 1.

    Returns:
        tree (SyntheticTree): The final synthetic tree
        act (int): The final action (to know if the tree was "properly"
            terminated).
    """
    # Initialization
    tree = SyntheticTree()
    mol_recent = None
    kdtree = BallTree(bb_emb, metric=cosine_distance)

    # Start iteration
    for i in range(max_step):
        # Encode current state
        state = tree.get_state() # a set
        try:
            z_state = set_embedding(z_target, state, nbits=n_bits, _mol_embedding=mol_fp)
        except:
            z_target = np.expand_dims(z_target, axis=0)
            z_state = set_embedding(z_target, state, nbits=n_bits, _mol_embedding=mol_fp)

        # Predict action type, masked selection
        # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
        action_proba = action_net(torch.Tensor(z_state))
        action_proba = action_proba.squeeze().detach().numpy() + 1e-10
        action_mask  = get_action_mask(tree.get_state(), reaction_templates)
        act          = np.argmax(action_proba * action_mask)

        z_mol1 = reactant1_net(torch.Tensor(z_state))
        z_mol1 = z_mol1.detach().numpy()

        # Select first molecule
        if act == 3:
            # End
            break
        elif act == 0:
            # Add
            if mol_recent is not None:
                dist, ind = nn_search(z_mol1)
                mol1 = building_blocks[ind]
            else:
                dist, ind = nn_search_rt1(z_mol1, _tree=kdtree, _k=rt1_index+1)
                mol1 = building_blocks[ind[rt1_index]]
        else:
            # Expand or Merge
            mol1 = mol_recent

        # z_mol1 = get_mol_embedding(mol1, mol_embedder)
        z_mol1 = mol_fp(mol1)

        # Select reaction
        try:
            reaction_proba = rxn_net(torch.Tensor(np.concatenate([z_state, z_mol1], axis=1)))
        except:
            z_mol1 = np.expand_dims(z_mol1, axis=0)
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
        rxn    = reaction_templates[rxn_id]

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
                elif rxn_template == 'unittest':
                    z_mol2 = reactant2_net(torch.Tensor(np.concatenate([z_state, z_mol1, one_hot_encoder(rxn_id, 3)], axis=1)))
                z_mol2         = z_mol2.detach().numpy()
                available      = available_list[rxn_id]
                available      = [bb_dict[available[i]] for i in range(len(available))]
                temp_emb       = bb_emb[available]
                available_tree = BallTree(temp_emb, metric=cosine_distance)
                dist, ind      = nn_search(z_mol2, _tree=available_tree)
                mol2           = building_blocks[available[ind]]
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

def synthetic_tree_decoder_multireactant(z_target,
                                         building_blocks,
                                         bb_dict,
                                         reaction_templates,
                                         mol_embedder,
                                         action_net,
                                         reactant1_net,
                                         rxn_net,
                                         reactant2_net,
                                         bb_emb,
                                         rxn_template,
                                         n_bits,
                                         beam_width : int=3,
                                         max_step : int=15):
    """
    Computes the synthetic tree given an input molecule embedding, using the
    Action, Reaction, Reactant1, and Reactant2 networks and a greedy search.

    Args:
        z_target (np.ndarray): Embedding for the target molecule
        building_blocks (list of str): Contains available building blocks
        bb_dict (dict): Building block dictionary
        reaction_templates (list of Reactions): Contains reaction templates
        mol_embedder (dgllife.model.gnn.gin.GIN): GNN to use for obtaining molecular embeddings
        action_net (synth_net.models.mlp.MLP): The action network
        reactant1_net (synth_net.models.mlp.MLP): The reactant1 network
        rxn_net (synth_net.models.mlp.MLP): The reaction network
        reactant2_net (synth_net.models.mlp.MLP): The reactant2 network
        bb_emb (list): Contains purchasable building block embeddings.
        rxn_template (str): Specifies the set of reaction templates to use.
        n_bits (int): Length of fingerprint.
        beam_width (int): The beam width to use for Reactant 1 search. Defaults to 3.
        max_step (int, optional): Maximum number of steps to include in the synthetic tree

    Returns:
        tree (SyntheticTree): The final synthetic tree
        act (int): The final action (to know if the tree was "properly" terminated)
    """
    trees = []
    smiles = []
    similarities = []
    acts = []

    for i in range(beam_width):
        tree, act = synthetic_tree_decoder_rt1(z_target=z_target,
                                               building_blocks=building_blocks,
                                               bb_dict=bb_dict,
                                               reaction_templates=reaction_templates,
                                               mol_embedder=mol_embedder,
                                               action_net=action_net,
                                               reactant1_net=reactant1_net,
                                               rxn_net=rxn_net,
                                               reactant2_net=reactant2_net,
                                               bb_emb=bb_emb,
                                               rxn_template=rxn_template,
                                               n_bits=n_bits,
                                               max_step=max_step,
                                               rt1_index=i)


        similarities_ = np.array(tanimoto_similarity(z_target, [node.smiles for node in tree.chemicals]))
        max_simi_idx  = np.where(similarities_ == np.max(similarities_))[0][0]

        similarities.append(np.max(similarities_))
        smiles.append(tree.chemicals[max_simi_idx].smiles)
        trees.append(tree)
        acts.append(act)

    max_simi_idx = np.where(similarities == np.max(similarities))[0][0]
    similarity   = similarities[max_simi_idx]
    tree         = trees[max_simi_idx]
    smi          = smiles[max_simi_idx]
    act          = acts[max_simi_idx]

    return smi, similarity, tree, act
