import functools
import typing

from .rdkit_general_ops import get_atom_map_nums
from .rdkit_general_ops import canconicalize


def split_reagents_out_from_reactants_and_products(reactant_all_str: str,
                                                   product_all_str: str,
                                                   action_set: set,
                                                   ) -> typing.Tuple[str, str, str]:
    """
    :param reactant_all_str: SMILES string of all reactants -- individual reactants seperated by dots.
    :param product_all_str: SMILES string of all products -- individual reactants seperated by dots.
    :param action_set: list of atoms involved in reaction
    :return:
    """
    canon_map = lambda in_list: list(map(canconicalize, in_list))

    reactants_str = reactant_all_str.split('.')
    products_str = product_all_str.split('.')

    products_str_canon_set = set(canon_map(product_all_str.split('.')))

    product_smiles_set = set(products_str)
    products_to_keep = set(products_str)
    product_atom_map_nums = functools.reduce(lambda x, y: x | y, (get_atom_map_nums(prod) for prod in products_str))
    actions_atom_map_nums = action_set

    reactants = []
    reagents = []
    for candidate_reactant in reactants_str:
        atom_map_nums = get_atom_map_nums(candidate_reactant)

        # a) any atoms in products
        in_product = list(product_atom_map_nums & atom_map_nums)
        # b) any atoms in reaction center
        in_center = list(set(actions_atom_map_nums & atom_map_nums))

        # 1. Rule as reagent out due to missing in products and not being in reaction centre
        def reagent_flag1():
            return (len(in_product) == 0) and (len(in_center) == 0)

        # 2. Rule as reagent if unchanged in product
        def reagent_flag2():
            return (canconicalize(candidate_reactant) in products_str_canon_set)


        if reagent_flag1():
            reagents.append(candidate_reactant)
        elif reagent_flag2():
            reagents.append(candidate_reactant)
            products_to_keep -= {candidate_reactant}  # remove it from the products too.
        else:
            reactants.append(candidate_reactant)

    product_all_str = '.'.join(products_to_keep)
    return '.'.join(reactants), '.'.join(reagents), product_all_str