from tqdm import tqdm 
import numpy as np 
from tdc.generation import MolGen
data = MolGen(name = 'ZINC')
from tdc import Oracle 
smiles_lst = data.get_data()['smiles'].tolist()

output_file = "./zinc_label.txt"


oracle_names = ['qed', 'JNK3', 'GSK3B', 'DRD2', \
				'celecoxib_rediscovery', 'troglitazone_rediscovery', 'thiothixene_rediscovery', \
				   'aripiprazole_similarity', 'albuterol_similarity', 'mestranol_similarity', \
				   'isomers_c7h8n2o2', 'isomers_c9h10n2o2pf2cl', 'isomers_c11h24', \
				   'osimertinib_mpo', 'fexofenadine_mpo', 'ranolazine_mpo', 'perindopril_mpo', \
				   'amlodipine_mpo', 'sitagliptin_mpo', 'zaleplon_mpo', \
				   'median1', 'median2', \
				   'valsartan_smarts', 'deco_hop', 'scaffold_hop']
print("# of Oracle", len(oracle_names))
oracle_list = [Oracle(name) for name in oracle_names]


# batch_size = 10
# num_of_batch = int(np.ceil(len(smiles_lst) / batch_size))

with open(output_file, 'w') as fout:
	fout.write('\t'.join(['smiles'] + oracle_names) + '\n')
	for smiles in tqdm(smiles_lst):
		scores_lst = []
		fout.write(smiles)
		for oracle in oracle_list:
			fout.write('\t' + str(oracle(smiles)))	
			# fout.write(smiles + '\t' + str(s1) + '\t' + str(s2) + '\t' + str(s3) + '\t' + str(s4) + '\t' + str(s5) + '\n')
		fout.write('\n')



'''

ZINC 250K 

  - QED  6 min 

  - LogP <1.5hours 

  - JNK3 10 hours 
	- 0.15 second/mol

  - GSK 10 hours   
	- 0.15 second/mol

'''

