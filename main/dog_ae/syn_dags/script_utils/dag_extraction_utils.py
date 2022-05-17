
import logging
import typing
import enum
import functools
import collections

import tqdm
import tabulate
import networkx as nx
import numpy as np
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

from ..data.reaction_datasets import uspto as uspto_ds
from ..chem_ops import rdkit_general_ops
from ..chem_ops import rdkit_reaction_ops

logger = logging.getLogger("DAG-EXTRACTION")


def _canonicalise_and_remove_am(molecules_in):
    return [rdkit_general_ops.canconicalize(smi, remove_am=True) for smi in molecules_in]


class _LoopException(Exception):
    pass


class NodeType(enum.Enum):
    MOLECULE = "MOLECULE"
    REACTION = "REACTION"


class Reaction(typing.NamedTuple):
    reactants: typing.FrozenSet[str]
    products: typing.FrozenSet[str]


def is_node_ancestor_of_node(plausible_ancestor_smi, dag, smis_to_check):
    all_anc = functools.reduce(lambda x,y: x | y, [nx.ancestors(dag, (smi,)) for smi in smis_to_check])
    return (plausible_ancestor_smi,) in all_anc


def extract_reactions(reaction_dataset) -> typing.Tuple[typing.List[Reaction], typing.Set[str], dict]:
    """
    Note:
     1. All molecules coming out of this function will be canconicalised (atom maps removed).
     2. Reagents will be removed from the reaction (reagents classifed by not being involved in the final bond changes)
     3. Currently skipping all the reactions that have multiple products or result in overlap.

    :param reaction_dataset: A dataset that returns reactions as a tuple of:
        (reactants, products, bond_changes)
        eg extracted from the USPTO data format from Jin et al available here: https://github.com/wengong-jin/nips17-rexgen/tree/master/USPTO
    :return: list of reactions extracted; a set of all molecules seen; dict of extraction details.
    """

    reactions = []
    logger.debug("Extracting reactions")

    run_through_stats = dict(
        num_skipped_due_to_multiple_products=0,
        num_multiple_same_reactants=0,
        num_multiple_same_products=0,
        num_overlap_between_reactants_and_products=0,
        num_skipped_as_already_seen=0,
        num_total_molecules=0,
        num_kept=0,
        num_total=len(reaction_dataset))

    all_molecules = set()
    reactant_products_tuples_seen = set()

    for reactants, products, bond_changes in tqdm.tqdm(reaction_dataset, desc="extracting reactions..."):
        action_set = uspto_ds.actionset_from_uspto_line(bond_changes)
        reactants, reagents, products = rdkit_reaction_ops.split_reagents_out_from_reactants_and_products(
            reactants, products, action_set)

        reactants_split = reactants.split('.')
        reactants_split = _canonicalise_and_remove_am(reactants_split)
        reactants_split_set = frozenset(reactants_split)
        if len(reactants_split) != len(reactants_split_set):
            run_through_stats['num_multiple_same_reactants'] += 1

        products_split = products.split('.')
        products_split = _canonicalise_and_remove_am(products_split)
        products_split_set = frozenset(products_split)

        reaction = Reaction(reactants_split_set, products_split_set)

        if len(products_split) != len(products_split_set):
            run_through_stats['num_multiple_same_products'] += 1

        if len(products_split_set & reactants_split_set):
            # We can have the same canonical reactant in the reactants and the products. This does not mean nothing
            # happened -- the atom map numbers (which have been removed now) could be different. However, for our
            # purposes we shall ignore these reactions.
            run_through_stats['num_overlap_between_reactants_and_products'] += 1
            continue

        if len(products_split_set) > 1:
            # A small portion of USPTO is multiple products (even after removing reagents. However, for the moment
            # we shall ignore these reactions and focus on the main product.
            run_through_stats['num_skipped_due_to_multiple_products'] += 1
            continue

        if reaction in reactant_products_tuples_seen:
            # Reaction could already exist
            run_through_stats['num_skipped_as_already_seen'] += 1
            continue

        reactant_products_tuples_seen.add(reaction)
        reactions.append(reaction)
        all_molecules.update(reactants_split_set)
        all_molecules.update(products_split_set)

    run_through_stats['num_kept'] = len(reactions)
    run_through_stats['num_total_molecules'] = len(all_molecules)
    logger.info(f"Extracting reactions done:\n{tabulate.tabulate([[name, value] for name, value in run_through_stats.items()])}")

    logger.debug("Creating tree dict")
    num_reactions_before = len(reactions)
    reactions = sorted(list(set(reactions)))
    num_reactions_after = len(reactions)
    logger.info(f"Removing duplicated reactions {num_reactions_before-num_reactions_after}. leaving: {num_reactions_after}")

    logger.info(f"Number of reactions {len(reactions)}, number of molecules {len(all_molecules)}")

    return reactions, all_molecules, run_through_stats


def create_mega_graph(reactions: typing.List[Reaction], reactants_to_reactant_id: dict) -> nx.DiGraph:
    """
    Create NetworkX Graph
    Reactions represented as tuples holding frozensets of canonical SMILES
    Molecules Represented as one element tuple of canonical SMILES
    :param reactions: list of reactions from the dataset.
    :param reactants_to_reactant_id: dictionary mapping from reactant SMILES string to integer ID
    """

    available_reactions = collections.deque(reactions)

    # Set up initial variables
    reactant_set = set(reactants_to_reactant_id.keys())
    initial_reactant_set = set(reactants_to_reactant_id.keys())
    rxns_later = collections.defaultdict(list)
    mega_graph = nx.DiGraph()

    for initial_compound in reactant_set:
        mega_graph.add_node((initial_compound,))

    # Go through and add reactions
    number_times_gone_through = 0
    num_alternative_routes_to_product = 0
    num_added = 0
    num_skipped_due_to_loop = 0
    while len(available_reactions):
        logger.debug(f"Starting run through {number_times_gone_through}")
        for _ in tqdm.tqdm(range(len(available_reactions))):
            reaction_: Reaction = available_reactions.popleft()

            assert len(reaction_.products) == 1, "currently only using one product reactions"

            okay_to_add = reaction_.reactants.issubset(reactant_set) and not reaction_.products.issubset(initial_reactant_set)

            if okay_to_add:

                # can add the reaction to the graph!
                reaction_node_representation = (reaction_.reactants, reaction_.products)
                mega_graph.add_node(reaction_node_representation)

                # Add each of the product nodes
                reactant_set.update(reaction_.products)
                for prod in reaction_.products:
                    prod_repr = (prod,)
                    if prod_repr not in mega_graph:
                        mega_graph.add_node(prod_repr)
                    else:
                        num_alternative_routes_to_product += 1
                    mega_graph.add_edge(reaction_node_representation, prod_repr)

                    # Add reactions which were unavailable back to the queue and remove them from the dict
                    available_reactions.extend(rxns_later[prod])
                    rxns_later[prod] = []

                for react_ in reaction_.reactants:
                    react_repr = (react_,)
                    mega_graph.add_edge(react_repr, reaction_node_representation)

                num_added += 1
            elif reaction_.reactants.issubset(reactant_set) and reaction_.products.issubset(initial_reactant_set):
                pass
                num_skipped_due_to_loop += 1
                # ^ this is an uninteresting reaction as leads back to
                # a reactant that we already have right at beginning
            else:
                # has at least one reactant that we do not have access to at the moment
                reactants_missing = set(reaction_.reactants - reactant_set)
                one_arbitrary_reactant_missing = reactants_missing.pop()
                rxns_later[one_arbitrary_reactant_missing].append(reaction_)
        number_times_gone_through += 1

    number_discarded = sum([len(el) for el in rxns_later.values()])

    table = [
        ("Total num times gone through", number_times_gone_through),
        ("Number added", num_added),
        ("Number discarded due to not connecting", number_discarded),
        ("Number discarded due to forming loop", num_skipped_due_to_loop),
        ("Number alterative routes to products found", num_alternative_routes_to_product)
    ]
    logger.info(f"Extracting Mega-DAG done:\n{tabulate.tabulate(table)}")
    return mega_graph


def _recursive_sample_from_dag_starting_at_node(rng: np.random.RandomState,
                                               dag,
                                               node_smi,
                                               smiles_seen,
                                               ancestor_smiles,
                                               connect_flag,
                                               max_depth):
    # Check whether it will create a loop. If so then raise an Exception and we'll back up...
    if node_smi in ancestor_smiles:
        raise _LoopException
    else:
        ancestor_smiles = ancestor_smiles | {node_smi}  # note that this creates a new set.

    # Work out whether it has a shared node with somewhere else in the DAG. -- this is just as an interesting statistic
    connect_flag = connect_flag or node_smi in smiles_seen
    smiles_seen.add(node_smi)

    # Record how deep we have gone!
    max_depth += 1

    # If we are at a final node then we are done exploring further.
    in_edges = list(dag.in_edges((node_smi,)))
    if len(in_edges) == 0:
        tuple_tree = (node_smi, [])
    else:
        in_edge_possible_indices = rng.permutation(len(in_edges))

        for idx in in_edge_possible_indices:
            last_possible_idx_flag = idx == in_edge_possible_indices[-1]

            in_edge = in_edges[idx]
            reaction = in_edge[0]
            reactants = reaction[0]
            assert node_smi in reaction[1], "not in products...?"

            try:
                tuple_tree_down, max_depth, connect_flag  = zip(*[
                    _recursive_sample_from_dag_starting_at_node(rng, dag, n, smiles_seen, ancestor_smiles,
                                                                connect_flag, max_depth) for n in reactants])
            except _LoopException as ex:
                if last_possible_idx_flag:
                    raise ex
                else:
                    continue
            else:
                tuple_tree = (node_smi, list(tuple_tree_down))
                max_depth = np.max(max_depth)
                connect_flag = any(connect_flag)
                break  # it worked so do not need to explore other possibilities.

    return tuple_tree, max_depth, connect_flag


def extract_tuple_trees_from_mega_dag(mega_graph: nx.DiGraph,
                                      reactants_to_reactant_id: dict) -> \
        typing.Tuple[
            typing.List[typing.Tuple[int, tuple]],
             dict
                ]:
    """
    :param mega_graph: The DAG which contains the whole reaction network.
    :param reactants_to_reactant_id: dictionary mapping from reactant SMILES string to integer ID
    :return: a list of extracted trees/DAGs leading to one particular product, table of details about them.
    """
    # Load in the main dag
    reactant_set = set(reactants_to_reactant_id.keys())

    # Find all the root nodes.
    interesting_possible_root_nodes = set()
    for node in mega_graph:
        if len(node) == 1 and node[0] not in reactant_set:
            # ie node is a molecule node (not a reaction node) and it is not an initial reactant then it is interesting.
            interesting_possible_root_nodes.add(node[0])
    logger.info(f"Number of possible starting nodes: {len(interesting_possible_root_nodes)}")

    # Sample a tree from each root nodes
    depth_and_tree_tuples = []
    number_dags_of_these = 0
    max_depth = collections.Counter()
    rng = np.random.RandomState(100)
    for r_node in tqdm.tqdm(interesting_possible_root_nodes, desc="going through all possible final nodes."):
        tuple_tree, depth, connect_flag = _recursive_sample_from_dag_starting_at_node(rng, mega_graph,
                                                                                      r_node, set(), set(), False, 0)
        max_depth.update([depth])
        number_dags_of_these += int(connect_flag)
        depth_and_tree_tuples.append((depth, tuple_tree))

    # List some stats.
    stats = [["Number root nodes", len(interesting_possible_root_nodes)],
             ["Number which have repeated nodes", number_dags_of_these]] + \
            [[f"Number that have {k} levels", v] for k,v in max_depth.items()]
    logger.info(f"Extracting final trees done:\n{tabulate.tabulate(stats)}")

    return depth_and_tree_tuples, dict(stats)


