from tqdm import tqdm 
import os
# from chemutils import vocabulary, smiles2word 
from chemutils import is_valid, logp_modifier 
smiles_database = "data/zinc.tab"
clean_smiles_database = "data/zinc_clean.txt"


with open(smiles_database, 'r') as fin:
	lines = fin.readlines()[1:]
smiles_lst = [i.strip().strip('"') for i in lines]

clean_smiles_lst = []
for smiles in tqdm(smiles_lst):
	if is_valid(smiles):
		clean_smiles_lst.append(smiles)
clean_smiles_set = set(clean_smiles_lst)
with open(clean_smiles_database, 'w') as fout:
	for smiles in clean_smiles_set:
		fout.write(smiles + '\n')


