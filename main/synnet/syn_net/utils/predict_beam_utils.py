"""
This file contains various utils for decoding synthetic trees using beam search.
"""
import numpy as np
from rdkit import Chem
from syn_net.utils.data_utils import SyntheticTree
from sklearn.neighbors import BallTree, KDTree
from syn_net.utils.predict_utils import *


np.random.seed(6)


def softmax(x):
    """
    Computes softmax values for each sets of scores in x.

    Args:
        x (np.ndarray or list): Values to normalize.
    Returns:
        (np.ndarray): Softmaxed values.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

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
    return dist[0], ind[0]

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
                           beam_width,
                           rxn_template,
                           n_bits,
                           max_step=15):
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
        beam_width (int): The beam width to use for Reactant 1 search.
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
    tree = SyntheticTree()
    kdtree = BallTree(bb_emb, metric=cosine_distance)
    mol_recent = None

    # Start iteration
    # try:
    for i in range(max_step):
        # Encode current state
        state = tree.get_state() # a set
        z_state = set_embedding(z_target, state, nbits=n_bits, mol_fp=mol_fp)

        # Predict action type, masked selection
        # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
        action_proba = action_net(torch.Tensor(z_state))
        action_proba = action_proba.squeeze().detach().numpy() + 1e-10
        action_mask = get_action_mask(tree.get_state(), reaction_templates)
        act = np.argmax(action_proba * action_mask)

        reactant1_net_input = torch.Tensor(
            np.concatenate([z_state, one_hot_encoder(act, 4)], axis=1)
        )
        z_mol1 = reactant1_net(reactant1_net_input)
        z_mol1 = z_mol1.detach().numpy()

        # Select first molecule
        if act == 3:
            # End
            nlls = [0.0]
            break
        elif act == 0:
            # Add
            # **don't try to sample more points than there are in the tree
            # beam search for mol1 candidates
            dist, ind = nn_search(z_mol1, _tree=kdtree, _k=min(len(bb_emb), beam_width))
            try:
                mol1_probas = softmax(- 0.1 * dist)
                mol1_nlls = -np.log(mol1_probas)
            except:  # exception for beam search of length 1
                mol1_nlls = [-np.log(0.5)]
            mol1_list = [building_blocks[idx] for idx in ind]
            nlls = mol1_nlls
        else:
            # Expand or Merge
            mol1_list = [mol_recent]
            nlls = [-np.log(0.5)]

        rxn_list    = []
        rxn_id_list = []
        mol2_list   = []
        act_list    = [act] * beam_width
        for mol1_idx, mol1 in enumerate(mol1_list):

            z_mol1 = mol_fp(mol1)
            act = act_list[mol1_idx]

            # Select reaction
            z_mol1 = np.expand_dims(z_mol1, axis=0)
            reaction_proba = rxn_net(torch.Tensor(np.concatenate([z_state, z_mol1], axis=1)))
            reaction_proba = reaction_proba.squeeze().detach().numpy()

            if act != 2:
                reaction_mask, available_list = get_reaction_mask(mol1, reaction_templates)
            else:
                _, reaction_mask = can_react(tree.get_state(), reaction_templates)
                available_list = [[] for rxn in reaction_templates]

            if reaction_mask is None:
                if len(state) == 1:
                    act = 3
                    nlls[mol1_idx] += -np.log(action_proba * reaction_mask)[act]  # correct the NLL
                    act_list[mol1_idx] = act
                    rxn_list.append(None)
                    rxn_id_list.append(None)
                    mol2_list.append(None)
                    continue
                else:
                    act_list[mol1_idx] = act
                    rxn_list.append(None)
                    rxn_id_list.append(None)
                    mol2_list.append(None)
                    continue

            rxn_id = np.argmax(reaction_proba * reaction_mask)
            rxn = reaction_templates[rxn_id]
            rxn_nll = -np.log(reaction_proba * reaction_mask)[rxn_id]

            rxn_list.append(rxn)
            rxn_id_list.append(rxn_id)
            nlls[mol1_idx] += rxn_nll

            if np.isinf(rxn_nll):
                mol2_list.append(None)
                continue
            elif rxn.num_reactant == 2:
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
                    dist, ind = nn_search(z_mol2, _tree=available_tree, _k=min(len(temp_emb), beam_width))
                    try:
                        mol2_probas = softmax(-dist)
                        mol2_nll = -np.log(mol2_probas)[0]
                    except:
                        mol2_nll = 0.0
                    mol2 = building_blocks[available[ind[0]]]
                    nlls[mol1_idx] += mol2_nll
            else:
                mol2 = None

            mol2_list.append(mol2)

        # Run reaction until get a valid (non-None) product
        for i in range(0, len(nlls)):
            best_idx = np.argsort(nlls)[i]
            rxn      = rxn_list[best_idx]
            rxn_id   = rxn_id_list[best_idx]
            mol2     = mol2_list[best_idx]
            act      = act_list[best_idx]
            try:
                mol_product = rxn.run_reaction([mol1, mol2])
            except:
                mol_product = None
            else:
                if mol_product is None:
                    continue
                else:
                    break

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

def set_embedding_fullbeam(z_target, state, _mol_embedding, nbits):
    """
    Computes embeddings for all molecules in input state.

    Args:
        z_target (np.ndarray): Embedding for the target molecule.
        state (list): Contains molecules in the current state, if not the
            initial state.
        _mol_embedding (Callable): Function to use for computing the embeddings
            of the first and second molecules in the state (e.g. Morgan fingerprint).
        nbits (int): Number of bits to use for the embedding.

    Returns:
        np.ndarray: Embedding consisting of the concatenation of the target
            molecule with the current molecules (if available) in the input
            state.
    """
    if len(state) == 0:
        z_target = np.expand_dims(z_target, axis=0)
        return np.concatenate([np.zeros((1, 2 * nbits)), z_target], axis=1)
    else:
        e1 = _mol_embedding(state[0])
        e1 = np.expand_dims(e1, axis=0)
        if len(state) == 1:
            e2 = np.zeros((1, nbits))
        else:
            e2 = _mol_embedding(state[1])
            e2 = np.expand_dims(e2, axis=0)
        z_target = np.expand_dims(z_target, axis=0)
        return np.concatenate([e1, e2, z_target], axis=1)

def synthetic_tree_decoder_fullbeam(z_target,
                                    building_blocks,
                                    bb_dict,
                                    reaction_templates,
                                    mol_embedder,
                                    action_net,
                                    reactant1_net,
                                    rxn_net,
                                    reactant2_net,
                                    bb_emb,
                                    beam_width,
                                    rxn_template,
                                    n_bits,
                                    max_step=15):
    """
    Computes the synthetic tree given an input molecule embedding, using the
    Action, Reaction, Reactant1, and Reactant2 networks and a beam search.

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
        beam_width (int): The beam width to use for Reactant 1 search.
        rxn_template (str): Specifies the set of reaction templates to use.
        n_bits (int): Length of fingerprint.
        max_step (int, optional): Maximum number of steps to include in the synthetic tree

    Returns:
        tree (SyntheticTree): The final synthetic tree
        act (int): The final action (to know if the tree was "properly" terminated)
    """
    # Initialization
    tree = SyntheticTree()
    mol_recent = None
    kdtree = KDTree(bb_emb, metric='euclidean')

    # Start iteration
    # try:
    for i in range(max_step):
        # Encode current state
        state = tree.get_state() # a set
        z_state = set_embedding_fullbeam(z_target, state, mol_fp, nbits=n_bits)

        # Predict action type, masked selection
        # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
        action_proba = action_net(torch.Tensor(z_state))
        action_proba = action_proba.squeeze().detach().numpy()
        action_mask = get_action_mask(tree.get_state(), reaction_templates)
        act = np.argmax(action_proba * action_mask)

        z_mol1 = reactant1_net(torch.Tensor(np.concatenate([z_state, one_hot_encoder(act, 4)], axis=1)))
        z_mol1 = z_mol1.detach().numpy()

        # Select first molecule
        if act == 3:
            # End
            mol1_nlls = [0.0]
            break
        elif act == 0:
            # Add
            # **don't try to sample more points than there are in the tree
            # beam search for mol1 candidates
            dist, ind = nn_search(z_mol1, _tree=kdtree, _k=min(len(bb_emb), beam_width))
            try:
                mol1_probas = softmax(- 0.1 * dist)
                mol1_nlls = -np.log(mol1_probas)
            except:  # exception for beam search of length 1
                mol1_nlls = [-np.log(0.5)]
            mol1_list = [building_blocks[idx] for idx in ind]
        else:
            # Expand or Merge
            mol1_list = [mol_recent]
            mol1_nlls = [-np.log(0.5)]

        action_tuples = []  # list of action tuples created by beam search
        act_list      = [act] * beam_width
        for mol1_idx, mol1 in enumerate(mol1_list):

            z_mol1 = mol_fp(mol1, nBits=n_bits)
            act = act_list[mol1_idx]

            # Select reaction
            z_mol1 = np.expand_dims(z_mol1, axis=0)
            reaction_proba = rxn_net(torch.Tensor(np.concatenate([z_state, z_mol1], axis=1)))
            reaction_proba = reaction_proba.squeeze().detach().numpy()

            if act != 2:
                reaction_mask, available_list = get_reaction_mask(mol1, reaction_templates)
            else:
                _, reaction_mask = can_react(tree.get_state(), reaction_templates)
                available_list = [[] for rxn in reaction_templates]

            if reaction_mask is None:
                if len(state) == 1:
                    act = 3
                    mol1_nlls[mol1_idx] += -np.log(action_proba * reaction_mask)[act]  # correct the NLL
                    act_list[mol1_idx] = act
                    #                     nll,                 act, mol1, rxn, rxn_id, mol2
                    action_tuples.append([mol1_nlls[mol1_idx], act, mol1, None, None, None])
                    continue
                else:
                    act_list[mol1_idx] = act
                    #                     nll,                 act, mol1, rxn, rxn_id, mol2
                    action_tuples.append([mol1_nlls[mol1_idx], act, mol1, None, None, None])
                    continue

            rxn_ids = np.argsort(-reaction_proba * reaction_mask)[:beam_width]
            rxn_nlls = mol1_nlls[mol1_idx] - np.log(reaction_proba * reaction_mask)

            for rxn_id in rxn_ids:
                rxn = reaction_templates[rxn_id]
                rxn_nll = rxn_nlls[rxn_id]

                if np.isinf(rxn_nll):
                    #                     nll,     act, mol1, rxn, rxn_id, mol2
                    action_tuples.append([rxn_nll, act, mol1, rxn, rxn_id, None])
                    continue
                elif rxn.num_reactant == 2:
                    # Select second molecule
                    if act == 2:
                        # Merge
                        temp = set(state) - set([mol1])
                        mol2 = temp.pop()
                        #                     nll,     act, mol1, rxn, rxn_id, mol2
                        action_tuples.append([rxn_nll, act, mol1, rxn, rxn_id, mol2])
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
                        available_tree = KDTree(temp_emb, metric='euclidean')
                        dist, ind = nn_search(z_mol2, _tree=available_tree, _k=min(len(temp_emb), beam_width))
                        try:
                            mol2_probas = softmax(-dist)
                            mol2_nlls = rxn_nll - np.log(mol2_probas)
                        except:
                            mol2_nlls = [rxn_nll + 0.0]
                        mol2_list = [building_blocks[available[idc]] for idc in ind]
                        for mol2_idx, mol2 in enumerate(mol2_list):
                            #                     nll,                 act, mol1, rxn, rxn_id, mol2
                            action_tuples.append([mol2_nlls[mol2_idx], act, mol1, rxn, rxn_id, mol2])
                else:
                    #                     nll,     act, mol1, rxn, rxn_id, mol2
                    action_tuples.append([rxn_nll, act, mol1, rxn, rxn_id, None])

        # Run reaction until get a valid (non-None) product
        for i in range(0, len(action_tuples)):
            nlls     = list(zip(*action_tuples))[0]
            best_idx = np.argsort(nlls)[i]
            act      = action_tuples[best_idx][1]
            mol1     = action_tuples[best_idx][2]
            rxn      = action_tuples[best_idx][3]
            rxn_id   = action_tuples[best_idx][4]
            mol2     = action_tuples[best_idx][5]
            try:
                mol_product = rxn.run_reaction([mol1, mol2])
            except:
                mol_product = None
            else:
                if mol_product is None:
                    continue
                else:
                    break

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
