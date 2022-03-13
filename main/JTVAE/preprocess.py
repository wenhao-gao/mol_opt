from fast_jtnn import *
from tqdm import tqdm 

with open("../data/moses/train.txt",'r') as fin:
	lines = fin.readlines() 
smiles_lst = [line.strip() for line in lines][:5000]

new_smiles_lst = []
for smiles in tqdm(smiles_lst):
	try:
		mt = MolTree(smiles)
		new_smiles_lst.append(smiles)
	except:
		pass 

with open("../data/moses/train_valid.txt",'w') as fout:
	for smiles in new_smiles_lst:
		fout.write(smiles + '\n')


