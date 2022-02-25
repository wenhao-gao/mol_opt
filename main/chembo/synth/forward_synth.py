"""
Implements forward synthesis

TODO:
* Template synthesis and sanity checks

Notes:
* Using pretrained models:
* There is code for MoleculeTransformers:
    see https://github.com/pschwllr/MolecularTransformer
    so train it?
* Another option:
    https://github.com/connorcoley/rexgen_direct

"""

import sys
import logging

from mols.molecule import Molecule, Reaction
from rexgen_direct.core_wln_global.directcorefinder import DirectCoreFinder
from rexgen_direct.rank_diff_wln.directcandranker import DirectCandRanker


class ForwardSynthesizer:
    """
    Class for answering forward prediction queries.
    """
    def __init__(self):
        # load trained model
        pass

    def predict_outcome(self, reaction):
        """
        Using a predictor, produce the most likely reaction

        Params:
            reaction {Reaction} - reaction object that holds 
                                  lists of reactants/reagents
        Returns:
            {list[Molecule]} - list of k most likely reaction outcomes
        """
        raise NotImplementedError("Implement in child class.")


class TemplateForwardSynthesizer(ForwardSynthesizer):
    """ Class for rule-based synthesis using rdkit library. """
    pass


class RexgenForwardSynthesizer(ForwardSynthesizer):
    def __init__(self):
        # load trained model
        self.directcorefinder = DirectCoreFinder()
        self.directcorefinder.load_model()
        self.directcandranker = DirectCandRanker()
        self.directcandranker.load_model()

    def predict_outcome(self, reaction, k=1):
        """
        Using a predictor, produce top-k most likely reactions

        Params:
            reaction {Reaction}
            k {int} - how many top predictions to set and return
        Returns:
            {list[Molecule]} - list of products of reaction
        """
        react = reaction.get_input_str()
        try:
            (react, bond_preds, bond_scores, cur_att_score) = self.directcorefinder.predict(react)
            outcomes = self.directcandranker.predict(react, bond_preds, bond_scores)
        except RuntimeError as e:
            logging.error(f"Error occured in DirectCandRanker.predict: {e}")
            raise e

        res = []
        for out in outcomes:
            if out["smiles"]:  # may be empty for some reason?
                smiles = out["smiles"][0]
                mol = Molecule(smiles)
                mol.set_synthesis(reaction.inputs)
                res.append(mol)
            else:
                continue
        # outcomes are sorted by probability in decreasing order
        res = res[:k]

        # setting predicted products, if not already set:
        reaction.set_products(res)
        return res


if __name__=="__main__":
    list_of_mols = ["[CH3:26][c:27]1[cH:28][cH:29][cH:30][cH:31][cH:32]1", 
                    "[Cl:18][C:19](=[O:20])[O:21][C:22]([Cl:23])([Cl:24])[Cl:25]",
                    "[NH2:1][c:2]1[cH:3][cH:4][c:5]([Br:17])[c:6]2[c:10]1[O:9][C:8]"+
                    "([CH3:11])([C:12](=[O:13])[O:14][CH2:15][CH3:16])[CH2:7]2"
                    ]
    list_of_mols = [Molecule(smiles) for smiles in list_of_mols]
    t = RexgenForwardSynthesizer()
    reaction = Reaction(list_of_mols)
    t.predict_outcome(reaction)
