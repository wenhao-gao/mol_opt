import torch 

class Molecule_Dataset(torch.utils.data.Dataset):
	def __init__(self, molecule_X_label_y_list):
		self.smiles_lst = [i[0] for i in molecule_X_label_y_list]
		self.label_lst = [i[1] for i in molecule_X_label_y_list] 

	def __len__(self):
		return len(self.label_lst)

	def __getitem__(self, idx):
		return self.smiles_lst[idx], self.label_lst[idx]




