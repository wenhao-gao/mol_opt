
from os import path
import typing
import itertools

from torch.utils import data

from ...utils import settings
from .. import general


class UsptoDataset(data.Dataset):
    def __init__(self, dataset_partition: general.DatasetPartitions, transforms=None):
        uspto_path = path.join(settings.get_repo_path(),
                               settings.get_config().get('DataDirectories', 'uspto'),
                               f'{dataset_partition.value}.txt')
        with open(uspto_path, 'r') as fo:
            data = fo.readlines()
        self.reaction_lines = data
        self.transforms = transforms

    def __getitem__(self, idx: int):
        smiles = self.reaction_lines[idx]
        rest, bond_changes = smiles.split()
        (reactants, products) = rest.split('>>')

        return_val: typing.Tuple[str] = (reactants, products, bond_changes)
        if self.transforms is not None:
            return_val = self.transforms(*return_val)
        return return_val

    def __len__(self):
        return len(self.reaction_lines)


def actionset_from_uspto_line(change_str):
    """
    """
    change_list = change_str.split(';')
    atoms = set(itertools.chain(*[map(int, c.split('-')) for c in change_list]))
    return atoms
