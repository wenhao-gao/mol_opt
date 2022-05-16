
import collections

import torch
import numpy as np
import networkx as nx
from torch.nn.utils import rnn


from syn_dags.data import synthesis_trees
from syn_dags.data import smiles_to_feats


SAMPLE_TUPLE_TREE = ('CC(=O)Nc1ccc(Cl)c(I)c1F',
   [('CC(=O)Nc1ccc(Cl)cc1F', [('CC(=O)OC(C)=O', []), ('Nc1ccc(Cl)cc1F', [])]),
    ('FC(F)(F)CI', [])])

SAMPLE_TUPLE_TREE2 = ('CC(O)c1cc(F)cc(C#N)c1F',
 [('N#C[Cu]', []),
  ('CC(O)c1cc(F)cc(Br)c1F',
   [('O=Cc1cc(F)cc(Br)c1F',
     [('Nc1c(F)cc(C=O)c(F)c1Br',
       [('O=CO', []),
        ('N#Cc1cc(F)c(N)c(Br)c1F',
         [('N#Cc1cc(F)c(N)cc1F',
           [('N#C[Cu]', []),
            ('Nc1cc(F)c(Br)cc1F', [('BrBr', []), ('Nc1cc(F)ccc1F', [])])]),
          ('BrBr', [])])])]),
    ('C1CCOC1', [])])])


def test_tuple_tree_to_nx():

    expected_tree = nx.DiGraph()
    expected_tree.add_node(('CC(=O)Nc1ccc(Cl)c(I)c1F',))
    expected_tree.add_node(('CC(=O)Nc1ccc(Cl)cc1F',))
    expected_tree.add_node(('CC(=O)OC(C)=O',))
    expected_tree.add_node(('Nc1ccc(Cl)cc1F',))
    expected_tree.add_node(('FC(F)(F)CI',))
    expected_tree.add_edge(('FC(F)(F)CI',), ('CC(=O)Nc1ccc(Cl)c(I)c1F',))
    expected_tree.add_edge(('CC(=O)Nc1ccc(Cl)cc1F',), ('CC(=O)Nc1ccc(Cl)c(I)c1F',))
    expected_tree.add_edge(('CC(=O)OC(C)=O',), ('CC(=O)Nc1ccc(Cl)cc1F',))
    expected_tree.add_edge(('Nc1ccc(Cl)cc1F',), ('CC(=O)Nc1ccc(Cl)cc1F',))

    computed_tree = synthesis_trees.SynthesisTree.tuple_tree_to_nx(SAMPLE_TUPLE_TREE)

    assert set(map(lambda x: x[:-1], list(nx.to_edgelist(expected_tree)))) ==  \
           set(map(lambda x: x[:-1], list(nx.to_edgelist(computed_tree))))
    # ^ nx isomorphism test does not work for this as does not take account of node names. Hence we just check the edges
    # are correct and the same overall set of nodes

    assert set(expected_tree.nodes) == set(computed_tree.nodes)


def test_to_and_back_tuple_tree():
    computed_tree = synthesis_trees.SynthesisTree.tuple_tree_to_nx(SAMPLE_TUPLE_TREE)
    back = synthesis_trees.SynthesisTree.nx_to_tuple_tree(computed_tree, 'CC(=O)Nc1ccc(Cl)c(I)c1F')

    def make_invariant(tuple_tree):
        if isinstance(tuple_tree, str):
            return str
        elif isinstance(tuple_tree, list):
            return frozenset([make_invariant(e) for e in tuple_tree])
        elif isinstance(tuple_tree, tuple):
            return tuple([make_invariant(e) for e in tuple_tree])
        else:
            raise RuntimeError

    assert make_invariant(SAMPLE_TUPLE_TREE) == make_invariant(back)


def test_data_ordering():
    computed_tree = synthesis_trees.SynthesisTree.tuple_tree_to_nx(SAMPLE_TUPLE_TREE)

    for i in range(20):
        syn_tree = synthesis_trees.SynthesisTree(computed_tree, "CC(=O)Nc1ccc(Cl)c(I)c1F", rng=np.random.RandomState(i))

        # Note by convention we shall start as far from root node as possible but can choose nodes that are on the same
        # level in either possible orders.
        possible_orders = {
            (('CC(=O)OC(C)=O',), ('Nc1ccc(Cl)cc1F',), ('FC(F)(F)CI',), ('CC(=O)Nc1ccc(Cl)cc1F',), ('CC(=O)Nc1ccc(Cl)c(I)c1F',)),
            (('CC(=O)OC(C)=O',), ('Nc1ccc(Cl)cc1F',),  ('CC(=O)Nc1ccc(Cl)cc1F',), ('FC(F)(F)CI',), ('CC(=O)Nc1ccc(Cl)c(I)c1F',)),
            (('Nc1ccc(Cl)cc1F',), ('CC(=O)OC(C)=O',),   ('CC(=O)Nc1ccc(Cl)cc1F',), ('FC(F)(F)CI',), ('CC(=O)Nc1ccc(Cl)c(I)c1F',)),
            (('Nc1ccc(Cl)cc1F',), ('CC(=O)OC(C)=O',),   ('FC(F)(F)CI',), ('CC(=O)Nc1ccc(Cl)cc1F',),  ('CC(=O)Nc1ccc(Cl)c(I)c1F',)),
        }

        order_chosen = syn_tree.order_for_construction
        assert tuple(order_chosen) in possible_orders


def test_feature_extraction():
    """ Not an exhaustive test but checks some of the features out from get_features_required_for_training_model
    to check that
    """
    computed_tree = synthesis_trees.SynthesisTree.tuple_tree_to_nx(SAMPLE_TUPLE_TREE)
    order = [('CC(=O)OC(C)=O',), ('Nc1ccc(Cl)cc1F',), ('FC(F)(F)CI',), ('CC(=O)Nc1ccc(Cl)cc1F',), ('CC(=O)Nc1ccc(Cl)c(I)c1F',)]
    mol_to_graph_id = collections.OrderedDict([
        ('CC(=O)OC(C)=O',0), ('FC(F)(F)CI', 1), ('Nc1ccc(Cl)cc1F', 2), ('CC',3), ('CC(=O)C', 4),
        ('CC(=O)Nc1ccc(Cl)c(I)c1F', 5), ('CC(=O)Nc1ccc(Cl)cc1F',6)
    ])
    num_initial_reactants = 5
    # ^ nb that the initial reactant molecules need to come at beginning but then it can be anything in any order.

    syn_tree = synthesis_trees.SynthesisTree(computed_tree, "CC(=O)Nc1ccc(Cl)c(I)c1F", order)

    # Features
    feats = syn_tree.get_features_required_for_training_model(mol_to_graph_id, num_initial_reactants)

    # Check edges of final DAG
    expected_node_feats = np.array([0,2,1,6,5])[:, None]
    expected_edges = set([(3,0),(3,1),(4,2),(4,3)])
    expected_graph_ids = np.zeros(5)
    np.testing.assert_array_equal(expected_graph_ids, feats.dag_for_input.node_to_graph_id)
    np.testing.assert_array_equal(expected_node_feats, feats.dag_for_input.node_features)

    actual_edges = set([tuple(e.tolist()) for e in feats.dag_for_input.edge_type_to_adjacency_list_map['reactions'].T])
    assert actual_edges == expected_edges


    # Check the number of construction steps and their types & correct value.
    expected_choice_types = [0,1,0,1,0,1,0,2,2,2,0,2]
    np.testing.assert_array_equal(np.array(expected_choice_types), feats.sequence_action_kinds)

    expected_construction_choices = [0,0,0,2,0,1,1,0,2,7,1,8]
    np.testing.assert_array_equal(np.array(expected_construction_choices), feats.sequence_choices)

    # Check some random DAG states at various stages.
    assert feats.dags_at_construction_stages[0] is None
    expected_dag_indices = np.array([0,0,1,1,2,2,3,3,3,3,4,4])
    np.testing.assert_array_equal(expected_dag_indices, feats.dags_id_at_index)

    # -> Will check the first intermediate DAG
    expected_node_feats = np.array([0])[:, None]
    expected_edges = set()
    expected_graph_ids = np.zeros(1)
    np.testing.assert_array_equal(expected_graph_ids, feats.dags_at_construction_stages[1].node_to_graph_id)
    np.testing.assert_array_equal(expected_node_feats, feats.dags_at_construction_stages[1].node_features)

    actual_edges = set([tuple(e.tolist()) for e in feats.dags_at_construction_stages[1].edge_type_to_adjacency_list_map['reactions'].T])
    assert actual_edges == expected_edges

    # -> Will check the last intermediate DAG
    expected_node_feats = np.array([0,2,1,6])[:, None]
    expected_edges = set([(3,0),(3,1)])
    expected_graph_ids = np.zeros(4)
    np.testing.assert_array_equal(expected_graph_ids, feats.dags_at_construction_stages[4].node_to_graph_id)
    np.testing.assert_array_equal(expected_node_feats, feats.dags_at_construction_stages[4].node_features)

    actual_edges = set([tuple(e.tolist()) for e in feats.dags_at_construction_stages[4].edge_type_to_adjacency_list_map['reactions'].T])
    assert actual_edges == expected_edges

    # Check the edge masks
    edge_masks = np.array([
                           [1, 1, 1, 0, 0, 0, 0, 0, 1],
                           [0, 1, 1, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [1, 1, 1, 0, 0, 0, 1, 0, 1]],
                            dtype=feats.sequence_masks_for_edge_steps.dtype
                          )
    np.testing.assert_array_equal(edge_masks, feats.sequence_masks_for_edge_steps)

    # Check reactant masks
    reactant_masks = np.array([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 0, 1, 1]],
                              dtype=feats.sequence_masks_for_reactant_steps.dtype)
    np.testing.assert_array_equal(reactant_masks, feats.sequence_masks_for_reactant_steps)

    # Check that the dictionary does not get changed
    assert mol_to_graph_id == feats.mol_to_graph_idx
    assert num_initial_reactants == feats.num_that_are_initial_reactants


def test_collate_func():
    """

    Checks that the features are calculated correctly and collating done properly. Covers some of the tests above again
    so run those first -- sorry this is a bit of an ugly huge test checking by hand many things but useful to ensure
    data coming in correctly.

    Non exhaustive -- just checks the features in a few of the places.
    """

    list_of_tuple_trees = [
        ('CC(=O)Nc1ccc(Cl)c(I)c1F',
         [('CC(=O)Nc1ccc(Cl)cc1F',
           [('CC(=O)OC(C)=O', []), ('Nc1ccc(Cl)cc1F', [])]),
           ('FC(F)(F)CI', [])]),
        # 11 steps

         ('O=C(Cl)c1cc([N+](=O)[O-])ccc1F',
          [('O=C(Cl)C(=O)Cl', []), ('O=C(O)c1cc([N+](=O)[O-])ccc1F', [])]),
        # 5 steos

        # This sequence below is the one we shall check has been done correctly:
        ('CC(C)(C)OC(=O)N1CCC(Oc2ccc(C(N)=O)cn2)CC1',
          [('CC(C)(C)OC(=O)N1CCC(Oc2ccc(C#N)cn2)CC1',
            [('N#Cc1ccc(Cl)nc1', []), ('CC(C)(C)OC(=O)N1CCC(O)CC1', [])]),
           ('O=C([O-])[O-]', [])]),
        # 11 steps

         ('CC(n1ncnn1)C1(c2ccc(F)cc2F)CO1',
          [('CC(O)C1(c2ccc(F)cc2F)CO1', []), ('c1nnn[nH]1', [])]),
        # 5 steps

         ('COc1ccc(Oc2ccnc3cc(OC)c(OC)cc23)cn1',
          [('C[O-]', []),
           ('COc1cc2nccc(Oc3ccc(Cl)nc3)c2cc1OC',
            [('Oc1ccc(Cl)nc1', []), ('COc1cc2nccc(Cl)c2cc1OC', [])])])
        # 11 steps

    ]

    reactant_vocab = ['CC(=O)OC(C)=O', 'FC(F)(F)CI', 'Nc1ccc(Cl)cc1F', 'O=C(Cl)C(=O)Cl', 'O=C(O)c1cc([N+](=O)[O-])ccc1F',
                      'N#Cc1ccc(Cl)nc1', 'CC(C)(C)OC(=O)N1CCC(O)CC1', 'O=C([O-])[O-]', 'CC(O)C1(c2ccc(F)cc2F)CO1',
                      'c1nnn[nH]1', 'C[O-]', 'COc1cc2nccc(Cl)c2cc1OC', 'Oc1ccc(Cl)nc1']

    collate_func = synthesis_trees.CollateWithLargestFirstReordering(reactant_vocab)
    collated_batch, new_order = collate_func(list_of_tuple_trees)

    # Check that new order is valid ie it is one of deceresing size.
    sizes = np.array([11, 5, 11, 5, 11])
    assert np.all(np.diff(sizes[new_order]) <= 0 )

    # will check the third one (note there are four possible ways that this one can be built.
    indx_for_original_third = int(np.nonzero(new_order == 2)[0])

    # Check sequence choices
    # four possible
    expected_v1 = np.array([synthesis_trees.ADD_STEP_CHOOSE_REACTANT, 5, synthesis_trees.ADD_STEP_CHOOSE_REACTANT,
                            6, synthesis_trees.ADD_STEP_CHOOSE_REACTANT, 7,
                            synthesis_trees.ADD_STEP_CHOOSE_PRODUCT,
                            5, 6, len(collated_batch.mol_to_graph_idx), synthesis_trees.ADD_STEP_CHOOSE_PRODUCT,
                            len(collated_batch.mol_to_graph_idx)+1])
    expected_v2 = np.array([synthesis_trees.ADD_STEP_CHOOSE_REACTANT, 6, synthesis_trees.ADD_STEP_CHOOSE_REACTANT,
                            5, synthesis_trees.ADD_STEP_CHOOSE_REACTANT, 7,
                            synthesis_trees.ADD_STEP_CHOOSE_PRODUCT,
                            5, 6, len(collated_batch.mol_to_graph_idx), synthesis_trees.ADD_STEP_CHOOSE_PRODUCT,
                            len(collated_batch.mol_to_graph_idx)+1])
    expected_v3 = np.array([synthesis_trees.ADD_STEP_CHOOSE_REACTANT, 5, synthesis_trees.ADD_STEP_CHOOSE_REACTANT,
                            6,
                            synthesis_trees.ADD_STEP_CHOOSE_PRODUCT,
                            5, 6, len(collated_batch.mol_to_graph_idx),
                            synthesis_trees.ADD_STEP_CHOOSE_REACTANT, 7, synthesis_trees.ADD_STEP_CHOOSE_PRODUCT,
                            len(collated_batch.mol_to_graph_idx)+1])
    expected_v4 = np.array([synthesis_trees.ADD_STEP_CHOOSE_REACTANT, 6, synthesis_trees.ADD_STEP_CHOOSE_REACTANT,
                            5, synthesis_trees.ADD_STEP_CHOOSE_PRODUCT,
                            5, 6, len(collated_batch.mol_to_graph_idx),
                            synthesis_trees.ADD_STEP_CHOOSE_REACTANT, 7, synthesis_trees.ADD_STEP_CHOOSE_PRODUCT,
                            len(collated_batch.mol_to_graph_idx)+1])

    actual_choices = rnn.pad_packed_sequence(collated_batch.sequence_choices, batch_first=True)[0][
                     indx_for_original_third, :]
    actual_choices = actual_choices.detach().cpu().numpy()
    out = np.array([np.all(expected_v1 == actual_choices),
           np.all(expected_v2 == actual_choices),
           np.all(expected_v3 == actual_choices),
           np.all(expected_v4 == actual_choices)])
    assert out.sum() == 1
    which_indx = int(np.nonzero(out)[0])

    # Check sequence action kinds
    expected_action_kinds_1 = np.array([synthesis_trees.ADD_STEP_VAL, synthesis_trees.REACTANT_CHOOSE_STEP_VAL,
                                      synthesis_trees.ADD_STEP_VAL, synthesis_trees.REACTANT_CHOOSE_STEP_VAL,
                                      synthesis_trees.ADD_STEP_VAL, synthesis_trees.REACTANT_CHOOSE_STEP_VAL,
                                      synthesis_trees.ADD_STEP_VAL, synthesis_trees.EDGE_ADD_STEP_VAL,
                                      synthesis_trees.EDGE_ADD_STEP_VAL, synthesis_trees.EDGE_ADD_STEP_VAL,
                                      synthesis_trees.ADD_STEP_VAL, synthesis_trees.EDGE_ADD_STEP_VAL])
    expected_action_kinds_2 = np.array([synthesis_trees.ADD_STEP_VAL, synthesis_trees.REACTANT_CHOOSE_STEP_VAL,
                                      synthesis_trees.ADD_STEP_VAL, synthesis_trees.REACTANT_CHOOSE_STEP_VAL,
                                      synthesis_trees.ADD_STEP_VAL,  synthesis_trees.EDGE_ADD_STEP_VAL,
                                      synthesis_trees.EDGE_ADD_STEP_VAL, synthesis_trees.EDGE_ADD_STEP_VAL,
                                      synthesis_trees.ADD_STEP_VAL,synthesis_trees.REACTANT_CHOOSE_STEP_VAL,
                                      synthesis_trees.ADD_STEP_VAL, synthesis_trees.EDGE_ADD_STEP_VAL])
    actual_action_kind = rnn.pad_packed_sequence(collated_batch.sequence_action_kinds, batch_first=True)[0][indx_for_original_third, :]
    actual_action_kind = actual_action_kind.detach().cpu().numpy()

    if which_indx in {0, 1}:
        np.testing.assert_array_equal(expected_action_kinds_1, actual_action_kind)
    else:
        np.testing.assert_array_equal(expected_action_kinds_2, actual_action_kind)


    # Check dag for input
    # --> create the expected graph
    expected_graph = nx.DiGraph()
    expected_graph.add_node(collated_batch.mol_to_graph_idx["CC(C)(C)OC(=O)N1CCC(Oc2ccc(C(N)=O)cn2)CC1"])
    expected_graph.add_node(collated_batch.mol_to_graph_idx["CC(C)(C)OC(=O)N1CCC(Oc2ccc(C#N)cn2)CC1"])
    expected_graph.add_node(collated_batch.mol_to_graph_idx["N#Cc1ccc(Cl)nc1"])
    expected_graph.add_node(collated_batch.mol_to_graph_idx["CC(C)(C)OC(=O)N1CCC(O)CC1"])
    expected_graph.add_node(collated_batch.mol_to_graph_idx["O=C([O-])[O-]"])
    expected_graph.add_edge(collated_batch.mol_to_graph_idx["O=C([O-])[O-]"], collated_batch.mol_to_graph_idx["CC(C)(C)OC(=O)N1CCC(Oc2ccc(C(N)=O)cn2)CC1"])
    expected_graph.add_edge(collated_batch.mol_to_graph_idx["CC(C)(C)OC(=O)N1CCC(Oc2ccc(C#N)cn2)CC1"],
                            collated_batch.mol_to_graph_idx["CC(C)(C)OC(=O)N1CCC(Oc2ccc(C(N)=O)cn2)CC1"])
    expected_graph.add_edge(collated_batch.mol_to_graph_idx["N#Cc1ccc(Cl)nc1"],
                            collated_batch.mol_to_graph_idx["CC(C)(C)OC(=O)N1CCC(Oc2ccc(C#N)cn2)CC1"])
    expected_graph.add_edge(collated_batch.mol_to_graph_idx["CC(C)(C)OC(=O)N1CCC(O)CC1"],
                            collated_batch.mol_to_graph_idx["CC(C)(C)OC(=O)N1CCC(Oc2ccc(C#N)cn2)CC1"])

    # --> Now get the actual
    actual_graph = nx.DiGraph()
    dag_mask = collated_batch.dags_for_inputs.node_to_graph_id == indx_for_original_third
    dag_nodes = collated_batch.dags_for_inputs.node_features[dag_mask].squeeze().detach().cpu().numpy()
    for n in dag_nodes:
        actual_graph.add_node(n)
    dag_indcs = set(torch.arange(collated_batch.dags_for_inputs.node_features.shape[0])[dag_mask].detach().cpu().numpy().tolist())

    for edge in collated_batch.dags_for_inputs.edge_type_to_adjacency_list_map['reactions'].T:
        in_, out_ = edge.detach().cpu().numpy().tolist()
        if in_ in dag_indcs:
            actual_graph.add_edge(
            int(collated_batch.dags_for_inputs.node_features[out_, 0]),
            int(collated_batch.dags_for_inputs.node_features[in_, 0]))

    # check they are the same.
    assert nx.is_isomorphic(expected_graph, actual_graph)


    # Check molecular graphs (very roughly)
    # --> do this by just checking atom counts for  Nc1ccc(Cl)cc1F (the third graph)
    mol_graph_mask = collated_batch.molecular_graphs.node_to_graph_id == 2
    mol_graph_feats = collated_batch.molecular_graphs.node_features[mol_graph_mask]
    molecular_graph_features = mol_graph_feats.sum(dim=0).detach().cpu().numpy()
    default_atom_featurizer = smiles_to_feats.DEFAULT_SMILES_FEATURIZER
    assert molecular_graph_features[default_atom_featurizer.atm_featurizer.atms_to_idx['C']] == 6
    assert molecular_graph_features[default_atom_featurizer.atm_featurizer.atms_to_idx['F']] == 1
    assert molecular_graph_features[default_atom_featurizer.atm_featurizer.atms_to_idx['N']] == 1
    assert molecular_graph_features[default_atom_featurizer.atm_featurizer.atms_to_idx['Cl']] == 1


    # Check edge sequence masks
    edge_masks = rnn.pad_packed_sequence(collated_batch.sequence_masks_for_edge_steps,
                                         batch_first=True)[0][indx_for_original_third]

    edge_masks = edge_masks[rnn.pad_packed_sequence(collated_batch.sequence_action_kinds,
                            batch_first=True)[0][indx_for_original_third, :] == synthesis_trees.EDGE_ADD_STEP_VAL]
    edge_masks = edge_masks.detach().cpu().numpy()
    np.testing.assert_array_equal(edge_masks[2],
                                  np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,1.]))

    expected_end_edge_mask = np.array([0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0, 1.])
    expected_end_edge_mask[collated_batch.mol_to_graph_idx['CC(C)(C)OC(=O)N1CCC(Oc2ccc(C#N)cn2)CC1']] = 1.
    np.testing.assert_array_equal(edge_masks[3], expected_end_edge_mask)

    assert np.all(edge_masks[0]
                                  <= np.array([0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1.]))
    assert np.all(edge_masks[1] <= np.array([0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,1.]))
    # ^ dont know which of the edges it will do first.


    # Check reactant sequence masks
    reactant_masks = rnn.pad_packed_sequence(collated_batch.sequence_masks_for_reactant_steps,
                                         batch_first=True)[0][indx_for_original_third]

    reactant_masks = reactant_masks[rnn.pad_packed_sequence(collated_batch.sequence_action_kinds,
                            batch_first=True)[0][indx_for_original_third, :] == synthesis_trees.REACTANT_CHOOSE_STEP_VAL]
    reactant_masks = reactant_masks.detach().cpu().numpy()

    expected_initial_mask = np.ones((13))
    np.testing.assert_array_equal(expected_initial_mask, reactant_masks[0])

    expected_third_mask = np.ones((13))
    expected_third_mask[[5,6]] = 0
    np.testing.assert_array_equal(expected_third_mask, reactant_masks[2])


    # Check that mol to graph idx is correct
    # should have reactants in it first
    for i, react in enumerate(reactant_vocab):
        assert collated_batch.mol_to_graph_idx[react] == i
    expected_intermediates = ['CC(=O)Nc1ccc(Cl)c(I)c1F', 'CC(=O)Nc1ccc(Cl)cc1F', 'O=C(Cl)c1cc([N+](=O)[O-])ccc1F',
                              'CC(C)(C)OC(=O)N1CCC(Oc2ccc(C#N)cn2)CC1', 'CC(C)(C)OC(=O)N1CCC(Oc2ccc(C(N)=O)cn2)CC1',
                              'CC(n1ncnn1)C1(c2ccc(F)cc2F)CO1', 'COc1ccc(Oc2ccnc3cc(OC)c(OC)cc23)cn1',
                              'COc1cc2nccc(Oc3ccc(Cl)nc3)c2cc1OC'
                              ]
    for expect_inter in expected_intermediates:
        assert collated_batch.mol_to_graph_idx[expect_inter] >= collated_batch.num_that_are_initial_reactants

    # Check num that are initial reactants is correct.
    assert len(reactant_vocab) == collated_batch.num_that_are_initial_reactants

