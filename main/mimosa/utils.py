import torch 



class Molecule_Dataset(torch.utils.data.Dataset):
	def __init__(self, smiles_lst):
		self.smiles_lst = smiles_lst

	def __len__(self):
		return len(self.smiles_lst)

	def __getitem__(self, idx):
		return self.smiles_lst[idx]




