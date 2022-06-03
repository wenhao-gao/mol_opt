import sys
import yaml, os
method_name = sys.argv[1] 
result_folder = "/project/molecular_data/graphnn/mol_opt/main/" +sys.argv[1]+ "/results/" 
import numpy as np 
from tqdm import tqdm 


def file2result(file):
	result = yaml.safe_load(open(file))
	value_list = [value for smiles, (value, idx) in result.items()]
	value_list.sort(reverse=True)
	top1 = value_list[0]
	top10 = np.mean(value_list[:10])
	top100 = np.mean(value_list[:100])
	return top1, top10, top100

oracle_list = ['albuterol_similarity', 'amlodipine_mpo',
			   'celecoxib_rediscovery', 'deco_hop', 
			   'drd2', 'fexofenadine_mpo', 'gsk3b', 
			   'isomers_c7h8n2o2', 'isomers_c9h10n2o2pf2cl', 
			   'jnk3', 'median1', 'median2', 'mestranol_similarity', 'osimertinib_mpo', 
			   'perindopril_mpo', 'qed', 'ranolazine_mpo', 'scaffold_hop', 
			   'thiothixene_rediscovery', 'troglitazone_rediscovery', 'zaleplon_mpo']

files = os.listdir(result_folder)
# print(files)
# for file in files:
# 	fullfile = os.path.join(result_folder, file)
# 	result = yaml.safe_load(open(fullfile))
# 	## {'CC1CCC(C)CC1': [0.3333333333333333, 2572], ..., }
whole_oracle_result = dict()
for oracle in tqdm(oracle_list):
	oracle_files = list(filter(lambda x:oracle in x, files))
	# if len(oracle_files)==0:
	# 	continue 
	oracle_result = []
	for file in oracle_files:
		fullfile = os.path.join(result_folder, file)
		oracle_result.append(file2result(fullfile))
	oracle_result.sort(key=lambda x:x[-1], reverse=True) 
	oracle_result = oracle_result[:5]
	top1 = np.mean([i[0] for i in oracle_result])
	top10 = np.mean([i[1] for i in oracle_result])
	top100 = np.mean([i[2] for i in oracle_result])
	whole_oracle_result[oracle]= (top1, top10, top100) 


print('------- TOP-1 ------')
for oracle in oracle_list:
	print(whole_oracle_result[oracle][0])
print(sum([whole_oracle_result[oracle][0] for oracle in oracle_list]))	

print('------- TOP-10 ------')
for oracle in oracle_list:
	print(whole_oracle_result[oracle][1])	
print(sum([whole_oracle_result[oracle][1] for oracle in oracle_list]))	


print('------- TOP-100 ------')
for oracle in oracle_list:
	print(whole_oracle_result[oracle][2])	
print(sum([whole_oracle_result[oracle][2] for oracle in oracle_list]))	





# albuterol_similarity amlodipine_mpo celecoxib_rediscovery deco_hop drd2 fexofenadine_mpo gsk3b isomers_c7h8n2o2 isomers_c9h10n2o2pf2cl jnk3 median1 median2 mestranol_similarity osimertinib_mpo perindopril_mpo qed ranolazine_mpo scaffold_hop thiothixene_rediscovery troglitazone_rediscovery zaleplon_mpo









