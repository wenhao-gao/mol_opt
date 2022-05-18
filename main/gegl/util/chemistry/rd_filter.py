import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
import os, sys 
path_here = os.path.dirname(os.path.realpath(__file__))

RULES_FILENAME = os.path.join(path_here, "../../resource/rd_filter/rules.json")
ALERT_FILENAME = os.path.join(path_here, "../../resource/rd_filter/alert_collection.csv")


class RDFilter:
    def __init__(self):
        with open(RULES_FILENAME) as json_file:
            self.rule_dict = json.load(json_file)

        rule_list = [
            x.replace("Rule_", "")
            for x in self.rule_dict.keys()
            if x.startswith("Rule") and self.rule_dict[x]
        ]

        rule_df = pd.read_csv(ALERT_FILENAME).dropna()
        rule_df = rule_df[rule_df.rule_set_name.isin(rule_list)]

        self.rule_list = []
        tmp_rule_list = rule_df[["rule_id", "smarts", "max", "description"]].values.tolist()
        for rule_id, smarts, max_val, desc in tmp_rule_list:
            smarts_mol = Chem.MolFromSmarts(smarts)
            self.rule_list.append([smarts_mol, max_val, desc])

    def __call__(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if not (self.rule_dict["MW"][0] <= MolWt(mol) <= self.rule_dict["MW"][1]):
            return False

        if not (self.rule_dict["LogP"][0] <= MolLogP(mol) <= self.rule_dict["LogP"][1]):
            return False

        if not (self.rule_dict["HBD"][0] <= NumHDonors(mol) <= self.rule_dict["HBD"][1]):
            return False

        if not (self.rule_dict["HBA"][0] <= NumHAcceptors(mol) <= self.rule_dict["HBA"][1]):
            return False

        if not (self.rule_dict["TPSA"][0] <= TPSA(mol) <= self.rule_dict["TPSA"][1]):
            return False

        for row in self.rule_list:
            patt, max_val, desc = row
            if len(mol.GetSubstructMatches(patt)) > max_val:
                return False

        return True
