
import typing
import collections
from dataclasses import dataclass

import multiset
import tqdm
import numpy as np
import torch
from torch.utils import data
from torch.nn.utils import rnn

import networkx as nx

import os, sys 
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append(os.path.join(path_here, '../../submodules/autoencoders'))
sys.path.append(os.path.join(path_here, '../../submodules/GNN'))



from graph_neural_networks.sparse_pattern import graph_as_adj_list as grphs

from . import smiles_to_feats
from ..utils import settings

ADD_STEP_VAL = 0
REACTANT_CHOOSE_STEP_VAL = 1
EDGE_ADD_STEP_VAL = 2

ADD_STEP_CHOOSE_REACTANT = 0
ADD_STEP_CHOOSE_PRODUCT = 1


@dataclass
class PredOutputSingle:
    dag_for_input: grphs.DirectedGraphAsAdjList
    """The DAG which describes the tree. T
    # The features are the indices of the graphs in mol_graphs.."""

    dags_at_construction_stages: typing.List[grphs.DirectedGraphAsAdjList]
    """
    All DAGs that may be needed at construction time -- consists of dag_for_input at various stages of construction. 
    None means an empty graph.
    """

    dags_id_at_index: np.array
    """
    Index into dags_at_construction_stages for which DAG exists at each construction step
     (NB that DAG is not necessarily changed at each construction step).
    """

    sequence_action_kinds: np.array
    """
    Indexes what kind of action sequence_choices belongs to (ie 0 for ADD, 1 for REACTANT and 2 for EDGE)
    """

    sequence_choices: np.array
    """
    The correct choice at each construction steps. Note for different action kinds this means different things.
    ie:
    - on an ADD step 0 means reactant, 1 means product
    - on a REACTANT add step it indexes all reactant molecules 
    - on  add steps indexes all edges added so far (if applicable), the stop-intermediate and stop-final actions.
    """

    sequence_masks_for_edge_steps: typing.Optional[np.array]
    """
    Masks for the EDGE steps, for the available intermediate/reactant molecules and the stop-intermediate,
    stop-final action.
    None if no EDGE STEPS
    """

    sequence_masks_for_reactant_steps: np.array
    """
    Masks for the REACTANT steps, ie you cannot select a reactant that you have already selected
    """

    mol_to_graph_idx: collections.OrderedDict
    """
    Molecules as keys to their index in mol_graphs as values. Initial reactant molecules all must come at the beginning.
    """

    num_that_are_initial_reactants: int
    """
    The number of keys at the beginning of mol_to_graph_idx that are initial reactants. Ones after this are intermediate 
    reactants
    """



class SynthesisTree:
    def __init__(self, tree: nx.DiGraph,
                 root_smi: str,
                 order_for_construction: typing.List = None,
                 rng: np.random.RandomState = None
                 ):
        """
        NOTE All SMILES at this point should be canoncalised.

        :param tree: a tree with the final molecule as root and directed edges from leaves to root (note unlike usual
        tree definition of edge direction as we want to show the direction of reactions and message passing).
        Nodes are one element tuples of the form (SMILES str,)
        :param root_smi: the smi str of the root or final molecule
        by our autoencoder.
        :param order_for_construction: A linearised way that the synthesis DAG can be created. This is just a list of
        the node names that we will visit.
        """
        if rng is None:
            rng = np.random.RandomState(2891732)
        self.rng = rng

        self.tree: nx.DiGraph = tree
        self.root_smi = root_smi

        self.unique_molecules_in_tree = sorted(list(set([e[0] for e in self.tree])))

        self.num_nodes = len(tree)
        self.num_uniq_nodes = len(self.unique_molecules_in_tree)

        if order_for_construction is None:
            order_for_construction = self.tree_to_seq(tree, self.root_smi, rng)
        self.order_for_construction = order_for_construction

    def immutable_repr(self, include_order=True):
        immutable_tuple_tree = self.tuple_tree_to_immutable(self.tuple_tree_repr())
        immutable_root_smi = self.root_smi
        immutable_order = tuple(self.order_for_construction)
        if include_order:
            return (immutable_tuple_tree, immutable_root_smi, immutable_order)
        else:
            return (immutable_tuple_tree, immutable_root_smi)

    @staticmethod
    def tuple_tree_to_immutable(tuple_tree):
        def recursive_func(elem_in):
            if isinstance(elem_in, (list, tuple)):
                if len(elem_in):
                    return tuple([recursive_func(elem) for elem in elem_in])
                else:
                    empty_tuple = ()
                    return empty_tuple
            elif isinstance(elem_in, str):
                return elem_in
            else:
                raise RuntimeError

        return recursive_func(tuple_tree)

    @staticmethod
    def clean_dirty_tuple_tree(possibly_dirty_synthesis_tree, existing_reactants: set):
        """
        Cleans up and returns a new synthesis tree in which nodes below reactants get pruned (as unnecessary)
        """
        def prune_unnecessary_nodes(tuple_tree_in):
            if isinstance(tuple_tree_in, str):
                raise RuntimeError("Tuple tree had a non reactant end node")

            if tuple_tree_in[0] in existing_reactants:
                return (tuple_tree_in[0], [])
            else:
                return (tuple_tree_in[0], [prune_unnecessary_nodes(elem) for elem in tuple_tree_in[1]])
        return prune_unnecessary_nodes(possibly_dirty_synthesis_tree)

    @staticmethod
    def tuple_tree_to_nx(tuple_tree):
        graph = nx.DiGraph()
        final_product_node = (tuple_tree[0],)
        graph.add_node(final_product_node)

        def recursive_builder(child_node, list_to_build_out):
            for tt in list_to_build_out:
                parent_node = (tt[0],)
                graph.add_node(parent_node)
                graph.add_edge(parent_node, child_node)
                recursive_builder(parent_node, tt[1])

        recursive_builder(final_product_node, tuple_tree[1])
        return graph

    @staticmethod
    def starting_reactants_from_tuple_tree(tuple_tree):
        starting_reactants = set()
        def descend(tuple_in):
            if len(tuple_in[1]) == 0:
                # no children therefore starting
                starting_reactants.add(tuple_in[0])
            else:
                [descend(t) for t in tuple_in[1]]
        descend(tuple_tree)
        return starting_reactants

    @property
    def starting_reactants(self):
        starting_reactants = set()
        for n in self.tree.nodes:
            if len(list(self.tree.predecessors(n))) == 0:
                starting_reactants.add(n)
        return starting_reactants

    def max_distance_from_root(self):
        return max(self._longest_path_backwards(self.tree, (self.root_smi,)).values())

    @staticmethod
    def nx_to_tuple_tree(nx_tree, root_smi):
        def recursive_build(child_node):
            return [(elem[0][0], recursive_build(elem[0])) for elem in nx_tree.in_edges(child_node)]
        return (root_smi, recursive_build((root_smi,)))


    @staticmethod
    def make_tuple_tree_invariant(tuple_tree):
        def make_tt_invariant(tt_in):
            if isinstance(tt_in, str):
                return tt_in
            elif isinstance(tt_in, tuple) and len(tt_in) == 2:
                return (make_tt_invariant(tt_in[0]), make_tt_invariant(tt_in[1]))
            elif isinstance(tt_in, list):
                return frozenset([make_tt_invariant(el) for el in tt_in])
            else:
                return RuntimeError
        return make_tt_invariant(tuple_tree)

    def tuple_tree_repr(self):
        return self.nx_to_tuple_tree(self.tree, self.root_smi)

    def get_features_required_for_training_model(self,
                 mol_to_graph_idx: collections.OrderedDict,
                 num_that_are_initial_reactants: typing.Optional[int]) -> PredOutputSingle:
        """
        :param mol_to_graph_idx: if passed this is a index from each molecule that turns up in the synthesis tree to its
        index in the collection of graphs (next argument)
        :param: num_that_are_initial_reactants: the number of molecules in mol_to_graph_idx which are just initial
                reactants.
        :return:
        pred_output (ie the tree in graph form.
        """
        dag_for_input = self.network_x_graph_to_graph_as_adj_list(self.tree, mol_to_graph_idx)

        order_when_picking_edges = dict(zip(self.order_for_construction, range(len(self.order_for_construction))))

        # === Create the starting inputs/variables to store the output at each stage ===
        # The parts which are relevant to the sequence.
        dags_at_construction_stages = [None]
        dags_id_at_index = [0]
        sequence_action_kinds = [ADD_STEP_VAL]

        sequence_choices = []
        sequence_masks_for_edge_steps = []
        sequence_masks_for_reactant_steps = []

        working_tree = nx.DiGraph()

        # === Go through the sequence of nodes we visit in constructing the synthesis tree, for each one
        # we need to add the corresponding actions for prediction ===
        for node in self.order_for_construction:
            smi_for_node = node[0]
            is_final = len(list(self.tree.successors(node))) == 0
            is_base_reactant = len(list(self.tree.predecessors(node))) == 0
            assert not (is_final and is_base_reactant), "only one disconnected element"


            if is_base_reactant:
                # == REACTANT NODE
                # == Reactant nodes consist of choosing a reactant node and then choosing the particular reactant.
                mol_indx = mol_to_graph_idx[smi_for_node]
                assert mol_indx < num_that_are_initial_reactants, "initial not stored correctly in" \
                                                                                        "mol to graph index"

                # Step 1 we add a reactant
                sequence_choices.append(ADD_STEP_CHOOSE_REACTANT)  # choose add reactant action

                # now next step will consist of choosing the actual reactant chosen:
                dags_id_at_index.append(len(dags_at_construction_stages)-1)
                sequence_action_kinds.append(REACTANT_CHOOSE_STEP_VAL)

                # Step 2 choose the actual reactant
                # We first create a mask of the reactants that we can choose (cannot pick same reactant twice)
                reactant_mask = np.ones(num_that_are_initial_reactants)
                for mol_node in working_tree.nodes:
                    mol_node_smi = mol_node[0]
                    mol_id = mol_to_graph_idx[mol_node_smi]
                    as_reactant = mol_id < num_that_are_initial_reactants
                    if as_reactant:
                        reactant_mask[mol_id] = 0
                sequence_masks_for_reactant_steps.append(reactant_mask)

                # We then note what the correct choice for this action is
                sequence_choices.append(mol_indx)

                # Then filling in the input at the next step.
                working_tree.add_node(node)
                dags_at_construction_stages.append(self.network_x_graph_to_graph_as_adj_list(working_tree,
                                                                                             mol_to_graph_idx))
                dags_id_at_index.append(len(dags_at_construction_stages)-1)
                sequence_action_kinds.append(ADD_STEP_VAL)

            else:
                # == PRODUCT NODE
                # == The intermediate nodes consist of choosing an intermediate node, then the previous nodes it connects
                # to before finally choosing the stop node.

                mol_indx = mol_to_graph_idx[smi_for_node]
                assert mol_indx >= num_that_are_initial_reactants, "initial not stored correctly in" \
                                                                  "mol to graph index"

                # We first add the intermediate node action.
                sequence_choices.append(ADD_STEP_CHOOSE_PRODUCT)
                dags_id_at_index.append(len(dags_at_construction_stages)-1)
                sequence_action_kinds.append(EDGE_ADD_STEP_VAL)

                # Then we go through adding the individual edges
                # we first work out what edges we want to add
                edges_to_add = []
                for edge_src, edge_dest in sorted(list(self.tree.in_edges(node)),
                                                      key=lambda x: order_when_picking_edges[x[0]]):
                    # ^ nb we are yet to account for this sorted rule in the masks.

                    # If on final node we only add edges which should be included due
                    # to already existing in the DAG with a successor  all others will get picked up
                    # anyway with the stop-final action:
                    if not is_final or (len(list(working_tree.successors(edge_src))) > 0):
                        edges_to_add.append((edge_src, edge_dest))

                mask = np.zeros(len(mol_to_graph_idx) + 2)  # +2 for stop-intermediate and stop-final

                for n in working_tree.nodes:
                    mask[mol_to_graph_idx[n[0]]] = 1  # can select this molecule as exists in graph
                mask[-1] = 1  # can always stop-final
                sequence_masks_for_edge_steps.append(mask)
                working_tree.add_node(node)


                for edge_src, edge_dest in edges_to_add:
                    assert edge_src in working_tree, "adding an edge to a node not added yet"
                    assert edge_dest == node

                    src_smi = edge_src[0]
                    sequence_choices.append(mol_to_graph_idx[src_smi])

                    # Next time round we are still on an edge selection step and also the DAG is unchanged.
                    dags_id_at_index.append(len(dags_at_construction_stages)-1)
                    sequence_action_kinds.append(EDGE_ADD_STEP_VAL)

                    # For the mask for which edges we can select at next step we can now stop either way
                    # (if not before) but can no longer select what we have previously selected.
                    mask = mask.copy()
                    mask[-2] = 1  # now can stop-intermediate too.

                    mask[mol_to_graph_idx[src_smi]] = 0
                    sequence_masks_for_edge_steps.append(mask)

                    working_tree.add_edge(edge_src, edge_dest)

                # The last step consists of adding the stop edge action, this can be either
                if is_final:
                    # stop-final
                    stop_final_idx = len(mol_to_graph_idx) + 1
                    sequence_choices.append(stop_final_idx)
                else:
                    # stop-intermediate
                    stop_intermediate_idx = len(mol_to_graph_idx)
                    sequence_choices.append(stop_intermediate_idx)
                    dags_at_construction_stages.append(self.network_x_graph_to_graph_as_adj_list(working_tree,
                                                                                                 mol_to_graph_idx))
                    # stop-intermediate means we continue next time around (now back to add step)
                    dags_id_at_index.append(len(dags_at_construction_stages)-1)
                    sequence_action_kinds.append(ADD_STEP_VAL)

        # Having constructed all the data parts we can now put it in the structure and pass this back
        stacked_sequence_masks = np.stack(sequence_masks_for_edge_steps) if len(sequence_masks_for_edge_steps) else None
        stacked_reactant_masks = np.stack(sequence_masks_for_reactant_steps) if len(sequence_masks_for_reactant_steps) else None

        return_ = PredOutputSingle(dag_for_input, dags_at_construction_stages, np.array(dags_id_at_index),
                                   np.array(sequence_action_kinds), np.array(sequence_choices),
                                   stacked_sequence_masks, stacked_reactant_masks, mol_to_graph_idx,
                                   num_that_are_initial_reactants)
        return return_

    def network_x_graph_to_graph_as_adj_list(self, nx_graph: nx.DiGraph, mol_to_graph_idx: collections.OrderedDict)\
            -> grphs.DirectedGraphAsAdjList:

        existing_nodes = set(nx_graph.nodes)
        ordered_smiles = collections.OrderedDict([(x[0], i) for i, x in enumerate(self.order_for_construction) if
                                                  x in existing_nodes])
        # ^ smiles string (taken out of tuple) followed by index in the graph


        node_features = np.array([mol_to_graph_idx[smi] for smi in ordered_smiles.keys()])[:, None]
        edges = np.array([(ordered_smiles[dest[0]], ordered_smiles[src[0]]) for src, dest in nx_graph.edges]).T
        edges = {'reactions': edges}
        node_to_graph_id = np.zeros(len(ordered_smiles), dtype=settings.NP_INT)

        return grphs.DirectedGraphAsAdjList(node_features, edges, node_to_graph_id)

    @staticmethod
    def tree_to_seq(tree, root_smi, rng: np.random.RandomState):
        assert nx.is_directed_acyclic_graph(tree)

        # We are going to form an order to iterate over the tree. For this we want to start with one of the nodes
        # furthest away from the root nodes although we do not care about ordering of nodes on same level -- and
        # actually we are gonna try to be random wrt this.
        lengths = SynthesisTree._longest_path_backwards(tree, (root_smi,))
        key_dict = {k: (-lengths[k], rng.choice(10000), k[0]) for k in tree.nodes}
        # ^ for the lexicographical sort by names we want to first use distance from root and then want to be random,
        # hence first two items of tuple above

        order_ = list(nx.lexicographical_topological_sort(tree, lambda node: key_dict[node]))

        return order_

    def text_for_construction(self, strict_mode=False):
        """
        Turns the construction into a text representation.
        """
        out = []
        for node in self.order_for_construction:
            is_final = len(list(self.tree.successors(node))) == 0
            is_base_reactant = len(list(self.tree.predecessors(node))) == 0
            if strict_mode:
                assert not (is_final and is_base_reactant), "only one disconnected element"

            if is_base_reactant:
                out.extend(["<ADD-REACTANT>", f"<REACTANT_({node[0]})>"])
            else:
                # must be an intermediate node, which we add and then add all the nodes which already exist in it.
                if is_final:
                    reactants_in = [n[0] for n in self.tree.predecessors(node) if len(list(self.tree.successors(n))) >= 2]
                    # ^ stop only has edges to reactants that already exist.
                    reactants_in_for_final = [n[0] for n in self.tree.predecessors(node)]
                    # ^ but the reaction includes everything (ie also stuff that is not yet joined up)
                    final_action = f"<STOP-FINAL_({'.'.join(reactants_in_for_final)}>>{node[0]})"

                else:
                    reactants_in = [n[0] for n in self.tree.predecessors(node)]
                    final_action = f"<STOP-INTER_({'.'.join(reactants_in)}>>{node[0]})"
                out.extend(["<ADD-PROD>"] + [f"<EDGE-From_({el})>" for el in reactants_in] +
                           [final_action])
        return ','.join(out)

    def draw(self):
        nx.draw(self.tree)

    @staticmethod
    def _longest_path_backwards(nx_dag, source):
        distances = {}
        assert isinstance(nx_dag, nx.DiGraph)

        def _add_node_and_check_children(node, dist_so_far):
            if node not in distances or dist_so_far > distances[node]:
                distances[node] = dist_so_far
            for n in nx_dag.predecessors(node):
                _add_node_and_check_children(n, dist_so_far + 1)
        _add_node_and_check_children(source, 0)
        return distances

    def compare_with_other_jacard(self, other_syn_tree):
        """
        This is useful for getting an idea of how close this tree is to other tree -- for instance checking
        reconstruction "accuracy".
        """
        this_reactants = self.starting_reactants
        other_reactants = other_syn_tree.starting_reactants
        jacard_similarity_between_reactants = float(len(this_reactants & other_reactants)) / len(this_reactants | other_reactants)

        this_nodes = set(self.tree.nodes)
        other_nodes = set(other_syn_tree.tree.nodes)
        jacard_similarity_between_nodes = float(len(this_nodes & other_nodes)) / len(this_nodes | other_nodes)
        return jacard_similarity_between_reactants, jacard_similarity_between_nodes

    def compare_with_other_graph_edit(self, other_syn_tree, upper_bound=10):
        # == As a  measure we are going to get graph edit distance
        this_tree = self.tree.copy()
        other_tree = other_syn_tree.tree.copy()

        # We're going to put the node name into the node attribute dict
        for node in this_tree.nodes:
            this_tree.nodes[node]['name'] = node

        for node in other_tree.nodes:
            other_tree.nodes[node]['name'] = node

        def node_comparison_func(node1_dict, node2_dict):
            return node1_dict['name'] == node2_dict['name']

        edit_distance = nx.graph_edit_distance(this_tree, other_tree, node_match=node_comparison_func,
                                               upper_bound=upper_bound)
        if edit_distance is None:
            edit_distance = upper_bound + 1

        return edit_distance

    def compare_with_other_ismorphism(self, other_syn_tree):
        this_tree = self.tree.copy()
        other_tree = other_syn_tree.tree.copy()

        for node in this_tree.nodes:
            this_tree.nodes[node]['name'] = node

        for node in other_tree.nodes:
            other_tree.nodes[node]['name'] = node

        def node_comparison_func(node1_dict, node2_dict):
            return node1_dict['name'] == node2_dict['name']

        is_isomprhic = nx.is_isomorphic(this_tree, other_tree, node_match=node_comparison_func)

        return is_isomprhic


class CollateWithLargestFirstReordering:
    def __init__(self, all_reactants: list):
        self.reactants = all_reactants
        self.reactants_set = frozenset(all_reactants)
        self.base_mol_to_idx_dict = collections.OrderedDict([(x,i) for i, x in enumerate(self.reactants)])

        print("Creating the features for the reactant graphs...")
        graphs = [smiles_to_feats.DEFAULT_SMILES_FEATURIZER.smi_to_feats(smi) for smi in
                  tqdm.tqdm(self.base_mol_to_idx_dict.keys(), desc="Building reactant features")]
        self.reactant_graphs = graphs[0].concatenate(graphs)

    def _get_seq_for_each_data_item(self, list_of_tuple_trees):
        # Go through and convert all the tuple trees to networkx data structure and collect up all extra molecules that
        # appear in this batch (we call these the intermediate molecules).
        all_molecules = set()
        syn_trees = []

        for tt in list_of_tuple_trees:
            nx_graph = SynthesisTree.tuple_tree_to_nx(tt)
            new_molecules = [e[0] for e in nx_graph.nodes]  # unpack the smiles out of tuple with 0 indexing.
            all_molecules.update(new_molecules)
            syn_trees.append(SynthesisTree(nx_graph, tt[0]))

        # Now add the intermediate graphs to mol_to_graph_idx and create the molecular graph features.
        all_molecules = list(all_molecules - self.reactants_set)
        mol_to_graph_idx = self.base_mol_to_idx_dict.copy()
        mol_to_graph_idx.update([(x, i) for i, x in enumerate(all_molecules, start=len(mol_to_graph_idx))])
        num_initial_reactants_in_mol_to_graph_idx = len(self.base_mol_to_idx_dict)

        # Now create the sequence features
        pred_out: typing.List[PredOutputSingle] = [
            st.get_features_required_for_training_model(mol_to_graph_idx, num_initial_reactants_in_mol_to_graph_idx)
            for st in syn_trees]
        return pred_out, mol_to_graph_idx, num_initial_reactants_in_mol_to_graph_idx, syn_trees

    def _get_molecular_graph_features(self, mol_to_graph_idx, num_intial_reactants_in_mol_to_graph_idx):
        extra_graphs = [smiles_to_feats.DEFAULT_SMILES_FEATURIZER.smi_to_feats(smi)
                        for smi in list(mol_to_graph_idx.keys())[num_intial_reactants_in_mol_to_graph_idx:]]
        molecule_graphs = self.reactant_graphs.concatenate([self.reactant_graphs] + extra_graphs)
        return molecule_graphs

    def _reorder_by_length_and_package(self, pred_out, molecule_graphs, mol_to_graph_idx,
                                       num_initial_reactants_in_mol_to_graph_idx, original_syn_trees):

        # Now package everything together!
        # --> compute the order
        PAD_VALUE = settings.PAD_VALUE
        seq_sizes = np.array([p.sequence_choices.size for p in pred_out])
        array_containing_original_indcs = np.argsort(seq_sizes)[::-1]  # we need to put the largest sequence first.
        seq_size_with_padding = seq_sizes.max()
        new_seq_sizes = seq_sizes[array_containing_original_indcs]

        # --> the input DoGs can just be stacked together.
        dags_for_input = [pred_out[i].dag_for_input for i in array_containing_original_indcs]
        dags_for_input = dags_for_input[0].concatenate(dags_for_input)
        dags_for_input.inplace_from_np_to_torch()

        # We also record where the root molecule for each of these lives in the graphs (last poistion of each DAG)
        final_molecule_indcs = np.cumsum(np.bincount(dags_for_input.node_to_graph_id,
                                                                    minlength=dags_for_input.max_num_graphs)) - 1
        final_molecule_indcs = torch.tensor(final_molecule_indcs, dtype=settings.TORCH_INT)

        # --> the other parts we want to put in PackedSequence or have more clearer indication of where they live
        # inside the other parts
        construction_dags: typing.List[grphs.DirectedGraphAsAdjList] = []
        dags_id_at_index = []
        sequence_action_kinds = []
        sequence_choices = []
        number_edge_choices_including_both_stops = len(mol_to_graph_idx) + 2
        edge_masks = np.full((len(array_containing_original_indcs),
                              seq_size_with_padding,
                              number_edge_choices_including_both_stops), PAD_VALUE)
        reactant_masks = np.full((len(array_containing_original_indcs),
                                  seq_size_with_padding,
                                  num_initial_reactants_in_mol_to_graph_idx), PAD_VALUE)

        for new_idx, old_idx in enumerate(array_containing_original_indcs):
            p = pred_out[old_idx]
            p_seq_size = p.sequence_choices.size

            # --> We will deal with the construction DAGS first. The empty DAG is the same for all of them (this is
            # at index 0 and should be None so can be shared)
            num_construction_dags_seen_so_far = len(construction_dags)
            construction_dags.extend(filter(lambda x: x is not None, p.dags_at_construction_stages))

            dags_id_with_correct_shift = p.dags_id_at_index
            dags_id_with_correct_shift[dags_id_with_correct_shift != 0] += num_construction_dags_seen_so_far
            # ^ The DAG ID will get shifted when we concatenate them but not the index for 0 as this is the empty DAG.
            assert dags_id_with_correct_shift.size == p_seq_size
            dags_id_at_index.append(np.pad(dags_id_with_correct_shift,
                                           (0, seq_size_with_padding-p_seq_size),
                                           'constant', constant_values=PAD_VALUE))

            assert p.sequence_action_kinds.size == p_seq_size
            new_seq_action_kinds = np.pad(p.sequence_action_kinds,
                                           (0, seq_size_with_padding-p_seq_size),
                                           'constant', constant_values=PAD_VALUE)
            sequence_action_kinds.append(new_seq_action_kinds)

            assert p.sequence_choices.size == p_seq_size
            sequence_choices.append(np.pad(p.sequence_choices,
                                           (0, seq_size_with_padding-p_seq_size),
                                           'constant', constant_values=PAD_VALUE))

            if p.sequence_masks_for_edge_steps is not None:
                edge_masks[new_idx, new_seq_action_kinds == EDGE_ADD_STEP_VAL, :] = p.sequence_masks_for_edge_steps
            else:
                assert (new_seq_action_kinds == EDGE_ADD_STEP_VAL).sum() == 0

            reactant_masks[new_idx, new_seq_action_kinds == REACTANT_CHOOSE_STEP_VAL, :] = p.sequence_masks_for_reactant_steps

        # --> Put the construction DAGs together
        construction_dags: grphs.DirectedGraphAsAdjList = construction_dags[0].concatenate(construction_dags)
        construction_dags.inplace_from_np_to_torch()

        # --> Pack the padded sequences together
        seq_sizes = torch.tensor(new_seq_sizes)

        dags_id_at_index = torch.tensor(np.stack(dags_id_at_index), dtype=settings.TORCH_INT)
        dags_id_at_index = rnn.pack_padded_sequence(dags_id_at_index, seq_sizes, batch_first=True)

        sequence_action_kinds = torch.tensor(np.stack(sequence_action_kinds), dtype=settings.TORCH_INT)
        sequence_action_kinds = rnn.pack_padded_sequence(sequence_action_kinds, seq_sizes, batch_first=True)

        sequence_choices = torch.tensor(np.stack(sequence_choices), dtype=settings.TORCH_INT)
        sequence_choices = rnn.pack_padded_sequence(sequence_choices, seq_sizes, batch_first=True)

        edge_masks = rnn.pack_padded_sequence(torch.tensor(edge_masks, dtype=torch.bool), seq_sizes, batch_first=True)

        reactant_masks = rnn.pack_padded_sequence(torch.tensor(reactant_masks, dtype=torch.bool), seq_sizes, batch_first=True)

        original_syn_trees = [original_syn_trees[i] for i in array_containing_original_indcs]

        return PredOutBatch(dags_for_inputs=dags_for_input, dags_at_construction_stages=construction_dags,
                            molecular_graphs=molecule_graphs, dags_id_at_index=dags_id_at_index,
                            sequence_action_kinds=sequence_action_kinds, sequence_choices=sequence_choices,
                            sequence_masks_for_edge_steps=edge_masks,
                            sequence_masks_for_reactant_steps=reactant_masks,
                            mol_to_graph_idx=mol_to_graph_idx,
                            num_that_are_initial_reactants=num_initial_reactants_in_mol_to_graph_idx,
                            final_molecule_indcs=final_molecule_indcs, syn_trees=original_syn_trees), \
               array_containing_original_indcs

    def __call__(self, list_of_tuple_trees):
        (pred_out, mol_to_graph_idx,
         num_intial_reactants_in_mol_to_graph_idx, syn_trees) = self._get_seq_for_each_data_item(list_of_tuple_trees)
        molecule_graphs = self._get_molecular_graph_features(mol_to_graph_idx, num_intial_reactants_in_mol_to_graph_idx)
        out = self._reorder_by_length_and_package(pred_out, molecule_graphs, mol_to_graph_idx,
                                                   num_intial_reactants_in_mol_to_graph_idx, syn_trees)
        return out




@dataclass
class PredOutBatch:
    """
    see PredOutputSingle for documentation of properties. Now these fields have been stacked together and moved to
    torch.
    """
    dags_for_inputs: grphs.DirectedGraphAsAdjList
    dags_at_construction_stages: grphs.DirectedGraphAsAdjList
    molecular_graphs: grphs.DirectedGraphAsAdjList
    dags_id_at_index: rnn.PackedSequence
    sequence_action_kinds: rnn.PackedSequence
    sequence_choices: rnn.PackedSequence
    sequence_masks_for_edge_steps: rnn.PackedSequence
    sequence_masks_for_reactant_steps: rnn.PackedSequence
    mol_to_graph_idx: collections.OrderedDict
    num_that_are_initial_reactants: int
    final_molecule_indcs: torch.Tensor
    """
    The indices of the final molecule in dags_for_inputs
    """

    syn_trees: typing.List[SynthesisTree]

    molecular_graph_embeddings: typing.Optional[torch.Tensor] = None
    """
    Molecular graph embeddings when calculated -- filled in later.
    """

    def inplace_to(self, arg):
        self.dags_for_inputs.inplace_torch_to(arg)
        self.dags_at_construction_stages.inplace_torch_to(arg)
        self.molecular_graphs.inplace_torch_to(arg)
        self.dags_id_at_index = self.dags_id_at_index.to(arg)
        self.sequence_action_kinds = self.sequence_action_kinds.to(arg)
        self.sequence_choices = self.sequence_choices.to(arg)
        self.sequence_masks_for_edge_steps = self.sequence_masks_for_edge_steps.to(arg)
        self.sequence_masks_for_reactant_steps = self.sequence_masks_for_reactant_steps.to(arg)
        self.final_molecule_indcs = self.final_molecule_indcs.to(arg)

    @property
    def batch_size(self):
        return len(self.syn_trees)
