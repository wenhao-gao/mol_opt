import os
from tqdm import tqdm 
vocab_file = 'data/Voc'
selfies_file = 'data/zinc.selfies'
from guacamol.utils.chemistry import canonicalize_list


if not os.path.exists(selfies_file):
	from tdc.generation import MolGen
	data = MolGen(name = 'ZINC')
	smiles_lst = data.smiles_lst.tolist()
	smiles_lst = canonicalize_list(smiles_lst, include_stereocenters=True)
	from tdc.chem_utils import MolConvert
	converter = MolConvert(src = 'SMILES', dst = 'SELFIES')
	selfies_lst = converter(smiles_lst)
	with open(selfies_file, 'w') as fout:
		for selfies in selfies_lst:
			fout.write(selfies + '\n')

with open(selfies_file, 'r') as fin:
	selfies_lst = fin.readlines() 
	selfies_lst = [selfies.strip() for selfies in selfies_lst]

vocab_set = set()
for selfies in tqdm(selfies_lst):
	words = selfies.strip().strip('[]').split('][')
	words = ['['+word+']' for word in words]
	words = set(words)
	vocab_set = vocab_set.union(words)

with open(vocab_file, 'w') as fout:
	for word in vocab_set:
		fout.write(word + '\n')



