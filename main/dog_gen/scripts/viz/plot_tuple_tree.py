"""
Script that plots a DAG (in tuple tree structure) in a tree representation for display in a browser.
"""

from string import Template
import json
import os
from os import path
import time

from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx

OP_path = 'data'

template_ = Template(r"""
<!DOCTYPE html>
<html>
    <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
    <meta name="viewport" content="width=device-width">
    <title> Collapsable example </title>
    <link rel="stylesheet" href="shared_parts/Treant.css">
    <link rel="stylesheet" href="shared_parts/collapsable.css">

    <link rel="stylesheet" href="shared_parts/perfect-scrollbar.css">

</head>
<body>
    <div class="chart" id="collapsable-example"></div>
    <script src="shared_parts/raphael.js"></script>
    <script src="shared_parts/Treant.js"></script>

    <script src="shared_parts/jquery.min.js"></script>
    <script src="shared_parts/jquery.easing.js"></script>


    <script>
        chart_config = {
        chart: {
            container: "#collapsable-example",

            animateOnInit: true,

            node: {
                collapsable: true
            },
            animation: {
                nodeAnimation: "easeOutBounce",
                nodeSpeed: 700,
                connectorsAnimation: "bounce",
                connectorsSpeed: 700
            }
        },
        nodeStructure: $node_structure       };
        tree = new Treant( chart_config );
    </script>
</body>
</html>
""")


def tuple_tree_to_nx(tuple_tree):
    tree = nx.DiGraph()
    def recusive_func(tree_remaining, parent):
        this_node = tree_remaining[0]
        tree.add_node(this_node)
        if parent is not None:
            tree.add_edge(parent, this_node)
        for other_nodes in tree_remaining[1]:
            recusive_func(other_nodes, this_node)
    recusive_func(tuple_tree, None)
    return tree


def _get_leaf_nodes(tuple_tree):
    tree = tuple_tree_to_nx(tuple_tree)

    leaf_nodes = set()
    for n in tree:
        is_leaf = not bool(tree.out_degree(n))
        if is_leaf:
            leaf_nodes.add(n)
    return leaf_nodes


def convert_tuple_tree_to_js(tuple_tree):
    def format_level(tuple, smi_list):
        smi, children = tuple
        smi_list.append(smi)
        mol = Chem.MolFromSmiles(smi)
        inchi = Chem.MolToInchiKey(mol)
        out_dict = {
            "image": f"imgs/{inchi}.svg"
        }
        if len(children):
            out_dict["children"] = [format_level(child, smi_list) for child in children]
            out_dict["collapsed"] = False

        return out_dict
    all_smi = []
    out_dict = format_level(tuple_tree, all_smi)
    out_dict.pop("collapsed")  # remove it from the top level
    all_smi = set(all_smi)
    return json.dumps(out_dict), all_smi


def main(tuple_tree):
    smiles_to_draw = set()

    node_structure1, smiles = convert_tuple_tree_to_js(tuple_tree)
    smiles_to_draw.update(smiles)

    with open(path.join(OP_path, f'plot_{time.strftime("%y-%m-%d_%H:%M:%S", time.gmtime())}.html'), 'w') as fo:
        fo.write(template_.substitute(node_structure=node_structure1))

    # Plot images
    os.makedirs(path.join(OP_path, 'imgs'), exist_ok=True)
    for smi in smiles_to_draw:
        mol = Chem.MolFromSmiles(smi)
        inchi_key = Chem.MolToInchiKey(mol)
        op_path = path.join(OP_path, 'imgs', f"{inchi_key}.svg")
        print(f"Saving {smi} to {op_path}")
        Draw.MolToFile(mol, op_path,  size=(200, 200), imageType="svg", useBWAtomPalette=True)

    print("Done!")


if __name__ == '__main__':
    # Modify this line below to change the tree that gets plotted!
    tuple_tree = ('Cc1ccc(F)c(OCC(C)(C)CO)c1', [('CC(C)(CO)CO', []), ('Cc1ccc(F)c(Br)c1', [('Fc1ccc(CBr)cc1Br', [('O=Cc1ccc(F)c(Br)c1', []), ('BrP(Br)Br', [])])])])

    main(tuple_tree)
