"""
Sketches the synthetic trees in a specified file.
"""
from syn_net.utils.data_utils import *
import argparse
from typing import Tuple
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolToImage
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# define some color maps for plotting
edges_cmap = {
    0 : "tab:brown",  # Add
    1 : "tab:pink",   # Expand
    2 : "tab:gray",   # Merge
    #3 : "tab:olive",  # End  # not currently plotting
}
nodes_cmap = {
    0 : "tab:blue",   # most recent mol
    1 : "tab:orange", # other root mol
    2 : "tab:green",  # product
}


def get_states_and_steps(synthetic_tree : "SyntheticTree") -> Tuple[list, list]:
    """
    Gets the different nodes of the input synthetic tree, and the "action type"
    that was used to get to those nodes.

    Args:
        synthetic_tree (SyntheticTree):

    Returns:
        Tuple[list, list]: Contains lists of the states and steps (actions) from
            the Synthetic Tree.
    """
    states = []
    steps = []

    target = synthetic_tree.root.smiles
    most_recent_mol = None
    other_root_mol = None

    for i, action in enumerate(st.actions):

        # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
        if action != 3:
            r = synthetic_tree.reactions[i]
            mol1 = r.child[0]
            if len(r.child) == 2:
                mol2 = r.child[1]
            else:
                mol2 = None
            state = [mol1, mol2, r.parent]
        else:
            state = [most_recent_mol, other_root_mol, target]

        if action == 2:
            most_recent_mol = r.parent
            other_root_mol = None

        elif action == 1:
            most_recent_mol = r.parent

        elif action == 0:
            other_root_mol = most_recent_mol
            most_recent_mol = r.parent

        states.append(state)
        steps.append(action)

    return states, steps

def draw_tree(states : list, steps : list, tree_name : str) -> None:
    """
    Draws the synthetic tree based on the input list of states (reactant/product
    nodes) and steps (actions).

    Args:
        states (list): Molecular nodes (i.e. reactants and products).
        steps (list): Action types (e.g. "Add" and "Merge").
        tree_name (str): Name of tree to use for file saving purposes.
    """
    G = nx.Graph()
    pos_dict = {}         # sets the position of the nodes, for plotting below
    edge_color_dict = {}  # sets the color of the edges based on the action
    node_color_dict = {}  # sets the color of the box around the node during plotting

    node_idx =0
    prev_target_idx = None
    merge_correction = 0.0
    for state_idx, state in enumerate(states):

        # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
        step = steps[state_idx]
        if step == 3:
            break

        skip_mrm = False
        skip_orm = False
        for smiles_idx, smiles in enumerate(state):

            if smiles is None and smiles_idx == 0:
                skip_mrm = True  # mrm == 'most recent mol'
                continue
            elif smiles is None and smiles_idx == 1:
                skip_orm = True  # orm == 'other root molecule'
                continue
            elif smiles is None and smiles_idx == 2:
                continue
            elif step == 1 and smiles_idx == 0:
                merge_correction -= 0.5
                skip_mrm = True  # mrm == 'most recent mol'
                continue

            # draw the molecules (creates a PIL image)
            img = MolToImage(mol=MolFromSmiles(smiles), fitImage=False)
            G.add_node(str(node_idx), image=img)
            node_color_dict[str(node_idx)] = nodes_cmap[smiles_idx]
            if smiles_idx != 2:
                pos_dict[str(node_idx)] = [state_idx + merge_correction, smiles_idx/2 + 0.01]
            else:
                pos_dict[str(node_idx)] = [state_idx + 0.5 + merge_correction, 0.01]  # 0.01 important to not plot edge under axis label, even if later axis label is turned off (weird behavior)
            if smiles_idx == 2:
                if not skip_mrm:
                    G.add_edge(str(node_idx - 2 + int(skip_orm)), str(node_idx))  # connect most recent mol to target
                    edge_color_dict[(str(node_idx - 2 + int(skip_orm)), str(node_idx))] = edges_cmap[step]
                if not skip_orm:
                    G.add_edge(str(node_idx - 1), str(node_idx))  # connect other root mol to target
                    edge_color_dict[(str(node_idx - 1), str(node_idx))] = edges_cmap[step]
            node_idx += 1

        if prev_target_idx and not step == 1:
            mrm_idx = node_idx - 3 + int(skip_orm)
            G.add_edge(str(prev_target_idx), str(mrm_idx))  # connect the previous target to the current most recent mol
            edge_color_dict[(str(prev_target_idx), str(mrm_idx))] = edges_cmap[step]
        elif prev_target_idx and step == 1:
            new_target_idx = node_idx - 1
            G.add_edge(str(prev_target_idx), str(new_target_idx))  # connect the previous target to the current most recent mol
            edge_color_dict[(str(prev_target_idx), str(new_target_idx))] = edges_cmap[step]
        prev_target_idx = node_idx - 1

    # sketch the tree
    fig, ax = plt.subplots()

    nx.draw_networkx_edges(
        G,
        pos=pos_dict,
        ax=ax,
        arrows=True,
        edgelist=[edge for edge in G.edges],
        edge_color=[edge_color_dict[edge] for edge in G.edges],
        arrowstyle="-",  # suppresses arrowheads
        width=2.0,
        alpha=0.9,
        min_source_margin=15,
        min_target_margin=15,
    )

    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    x = 0
    for positions in pos_dict.values():
        if positions[0] > x:
            x = positions[0]

    _, _ = ax.set_xlim(0, x)
    _, _ = ax.set_ylim(0, 0.6)
    icon_size = 0.2
    icon_center = icon_size / 2.0

    # add a legend for the edge colors
    markers_edges = [plt.Line2D([0,0],[0,0],color=color, linewidth=4, marker='_', linestyle='') for color in edges_cmap.values()]
    markers_nodes = [plt.Line2D([0,0],[0,0],color=color, linewidth=2, marker='s', linestyle='') for color in nodes_cmap.values()]
    markers_labels = ["Add", "Reactant 1", "Expand", "Reactant 2", "Merge", "Product"]
    markers =[markers_edges[0], markers_nodes[0], markers_edges[1], markers_nodes[1], markers_edges[2], markers_nodes[2]]
    plt.legend(markers, markers_labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

    # Add the respective image to each node
    for n in G.nodes:
        xf, yf = tr_figure(pos_dict[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]["image"])
        # add colored boxes around each node:
        plt.gca().add_patch(Rectangle((0,0),295,295, linewidth=2, edgecolor=node_color_dict[n], facecolor="none"))
        a.axis("off")

    ax.axis("off")

    # save the figure
    plt.savefig(f"{tree_name}.png", dpi=100)
    print(f"-- Tree saved in {tree_name}.png", flush=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='/pool001/rociomer/test-data/synth_net/st_hb_test-plot-tests.json.gz',
                        help="Path/filename to the synthetic trees.")
    parser.add_argument("--saveto", type=str, default='/pool001/rociomer/test-data/synth_net/images/',
                        help="Path to save the sketched synthetic trees.")
    parser.add_argument("--nsketches", type=int, default=-1,
                        help="How many trees to sketch. Default -1 means to sketch all.")
    parser.add_argument("--actions", type=int, default=-1,
                        help="How many actions the tree must have in order to sketch it (useful for testing).")
    args = parser.parse_args()

    st_set = SyntheticTreeSet()
    st_set.load(args.file)
    data = st_set.sts

    trees_sketched = 0
    for st_idx, st in enumerate(data):
        if len(st.actions) <= args.actions:
            # don't sketch trees with fewer than n = `args.actions` actions
            continue
        try:
            print("* Getting states and steps...")
            states, steps = get_states_and_steps(synthetic_tree=st)

            print("* Sketching tree...")
            draw_tree(states=states, steps=steps, tree_name=f"{args.saveto}tree{st_idx}")

            trees_sketched += 1

        except Exception as e:
            print(e)
            continue

        if not (args.nsketches == -1) and trees_sketched > args.nsketches:
            break

    print("Done!")
