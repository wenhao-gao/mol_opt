"""
Visualization tools for molecules
"""

import PIL
import pickle
import matplotlib.pyplot as plt
import numpy as np
from myrdkit import Draw
from mols.molecule import Molecule
from mols.molecule import smile_synpath_to_mols


def visualize_mol(mol: Molecule, path: str):
    """
    Draw a single molecule and save it to `path`
    :param mol: molecule to draw
    :param path: path to save the drawn molecule to
    """
    img = draw_molecule(mol)
    img.save(path)


def draw_molecule(mol: Molecule) -> PIL.Image.Image:
    """
    Draw a single molecule `mol` (make it `PIL.Image.Image`)
    :param mol: molecule to draw
    :return: corresponding image to `mol`
    """
    img = Draw.MolToImage(mol.to_rdkit())
    return img


def draw_synthesis_path(target_smiles: str, synth_path: str, out_path: str) -> None:
    """Draw the synthesis path and save to provided location.
    :param target_smiles: SMILES of the molecule being synthesized
    :param synth_path: dictionary of format SMILES m -> synpath of m
    :param out_path: where to save the resulting pdf
    """
    with open("./mols/best_molecule.pkl", "rb") as f:
        synpath = pickle.load(f)
    synpath = smile_synpath_to_mols(Molecule(smiles=target_smiles), synpath)
    drawer = SynPathDrawer(synpath, "plot")
    drawer.render(out_path)


class SynPathDrawer(object):
    def __init__(self, mol: Molecule, draw_mode: str):
        """
        :param mol: the molecule to draw synthesis path for
        :param draw_mode: "smiles" | "formula" | "plot" way of plotting each single molecule

        Examples::

            >>> drawer = SynPathDrawer(root_mol, "smiles")  # or "formula" or "plot"
            >>> drawer.render("some_output_dir/some_file_name")  # please, no file extension
        """
        assert draw_mode in ["smiles", "formula", "plot"]
        from graphviz import Digraph
        self._mol = mol
        self._dot = Digraph(comment="Synthesis path for {}".format(mol.to_smiles()), format="pdf")
        self._draw_mode = draw_mode
        self._node_counter = 0
        self._sub_dir = None

    def _draw(self, root: Molecule):
        if root.begin_flag:  # base case
            self._draw_node(root)
        else:
            for inp in root.inputs:
                self._draw(inp)
            self._draw_node(root)
            for inp in root.inputs:
                self._draw_edge(tail=inp, head=root)

    def _draw_edge(self, tail: Molecule, head: Molecule):
        self._dot.edge(tail_name=str(id(tail)), head_name=str(id(head)))

    def _draw_node(self, node: Molecule):
        import os
        self._node_counter += 1
        if self._draw_mode == "smiles":
            self._dot.node(name=str(id(node)), label=node.to_smiles())
        elif self._draw_mode == "formula":
            self._dot.node(name=str(id(node)), label=node.to_formula())
        elif self._draw_mode == "plot":
            mol_img_path = os.path.join(self._sub_dir, str(self._node_counter) + ".png")
            visualize_mol(node, path=mol_img_path)
            node_shape = 'rectangle' if node.begin_flag else 'plaintext'
            self._dot.node(name=str(id(node)), label="", image=mol_img_path, shape=node_shape)

    def render(self, out_path: str):
        """
        :param out_path: desired path + filename WITHOUT extension
        """
        import os
        import shutil
        self._sub_dir = os.path.join(os.path.dirname(out_path), ".tmp")
        try:
            os.makedirs(self._sub_dir, exist_ok=False)
            self._draw(self._mol)
            self._dot.render(out_path + ".gv", view=False)
        finally:
            shutil.rmtree(self._sub_dir)


if __name__ == "__main__":
    # smiles = "COc1ccccc1-c1ccc(C(Oc2ccccc2C(Nc2cccc(C)c2C)c2[nH]c3c(-c4noc(-c5ccc(Oc6cc(CC(C)=O)c7c(=O)cc(C)oc7c6)c(Cl)c5)n4)cccc3c2CC(=O)Nc2cccc(Nc3nc4c(-c5ccc(S(C)(=O)=O)cc5)cccn4n3)c2)=C(NC(=O)c2ccccc2)C(=O)O)o1"
    # mol = Molecule(smiles=smiles)
    # img = draw_molecule(mol)
    # img.save('./experiments/visualizations/opt_mol_plogp_5.png')


    # draw_synthesis_path(target_smiles="Cc1ccc(CC(CN=C(SC(=O)Cc2c[nH]c3c(-c4noc(-c5ccc(OC(C)C)c(Cl)c5)n4)cccc23)N(Cc2ccc(NS(=O)(=O)C(F)(F)F)cc2)C(=NCc2ccc(NS(=O)(=O)C(F)(F)F)cc2)SC(=O)Cc2c[nH]c3c(-c4noc(-c5ccc(OC(C)C)c(Cl)c5)n4)cccc23)COC(=O)C(C)(C)C)cc1C",
    #                     synth_path="./experiments/final/chemist_exp_dir_20190519053341/best_molecule.pkl",
    #                     out_path="./experiments/visualizations/synpath_plogp11")

    # draw_synthesis_path(target_smiles="CC(=NCN(CCC1=CCCCC1)C(=O)NCc1ccc(CN2CCCC(Nc3ccc4[nH]ncc4c3)C2)cc1)N(COc1ccccc1CNc1c(C(C)(C)C)ccc(C)c1C)c1ccc(C)cc1",
    #                     synth_path="./experiments/final/chemist_exp_dir_20190520035241/best_molecule.pkl",
    #                     out_path="./experiments/visualizations/synpath_plogp8")

    # draw_synthesis_path(target_smiles="CC(=O)Cc1cc(O)c(C(C)(C)C)c2oc(C)cc(=O)c12",
    #                     synth_path="./experiments/final/chemist_exp_dir_20190518184219/best_molecule.pkl",
    #                     out_path="./experiments/visualizations/synpath_qed92")

    draw_synthesis_path(target_smiles="Cc1c(Br)sc2nc(C(C)NC(C)C)oc(=O)c12",
                        synth_path="./experiments/extra_exps/chemist_exp_dir_20190627233033/best_molecule.pkl",
                        out_path="./experiments/visualizations/synpath_qed93")



