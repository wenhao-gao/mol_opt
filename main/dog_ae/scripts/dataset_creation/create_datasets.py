"""
Script to create Synthesis DAGs dataset.

"""

import typing
from os import path
import subprocess
import datetime
import logging
import collections

from dataclasses import dataclass

import networkx as nx
import numpy as np
import tabulate

from syn_dags.script_utils import dag_extraction_utils
from syn_dags.data.reaction_datasets import uspto as uspto_ds
from syn_dags.data import general
from syn_dags.utils import misc

PATH = 'data'
NAME_FOR_TRAIN = "train"

@dataclass
class _Dataset:
    name: str
    sources: typing.List[typing.Callable]
    split_proportions: typing.Dict[str, float]
    reactant_to_reactant_id_json_path: str


params = {
    'datasets': [
        _Dataset("uspto",
                 [lambda: uspto_ds.UsptoDataset(general.DatasetPartitions.TRAIN),
                  lambda: uspto_ds.UsptoDataset(general.DatasetPartitions.VALID),
                  lambda: uspto_ds.UsptoDataset(general.DatasetPartitions.TEST)
                  ],
                 {NAME_FOR_TRAIN: 0.9, "valid": 0.05, "test": 0.05},
                 'reactants_to_reactant_id.json'
                 )

    ]
}


def _split_on_proportions_and_save(name, proportion_dict, logger, depth_and_tree_tuples):

    length_all_data = len(depth_and_tree_tuples)
    assert sum(proportion_dict.values()) == 1., "proportions should sum to one"

    used_so_far = 0
    out_dict = {}
    out_trees_dict = {}

    for subset_name, proportion in proportion_dict.items():
        number_to_use = int(np.ceil(proportion * length_all_data))
        end_indx = min(used_so_far + number_to_use, length_all_data)
        indices = list(range(used_so_far, end_indx))

        depth_and_trees_for_subset = [depth_and_tree_tuples[i] for i in indices]
        out_trees_dict[subset_name] = depth_and_trees_for_subset
        misc.to_pickle(depth_and_trees_for_subset, path.join(PATH, f"{name}-{subset_name}-depth_and_tree_tuples.pick"))

        depths = collections.Counter([el[0] for el in depth_and_trees_for_subset])
        out_table = tabulate.tabulate([("Number of levels", "Freq")] + sorted(list(depths.items())) + [("Total", len(depth_and_trees_for_subset))])
        logger.info(f"For name: {name}, subset: {subset_name}, the tree levels are:\n{out_table}")

        out_dict[subset_name] = indices
        used_so_far = end_indx
    return out_dict, out_trees_dict


def _create_equiv_train_val_sets_and_save(name, out_trees_dict, reactants_list, name_of_train):

    def _get_top_smi(list_of_tt):
        return [elem[1][0] for elem in list_of_tt]

    out_smiles_lists = {}
    out_smiles_lists[name_of_train] = _get_top_smi(out_trees_dict[name_of_train])
    out_smiles_lists[name_of_train].extend(list(reactants_list))

    other_names = set(out_trees_dict.keys()) - {name_of_train}
    out_smiles_lists.update({k:_get_top_smi(out_trees_dict[k]) for k in other_names})

    for subset_name, subset_smiles in out_smiles_lists.items():
        with open(path.join(PATH, f"{name}-{subset_name}-equiv_smiles.txt"), 'w') as fo:
            fo.writelines('\n'.join(subset_smiles))

    return out_smiles_lists


def main(params):
    rng = np.random.RandomState(89424798)

    log_hndlr_stream = logging.StreamHandler()
    log_hndlr_stream.setLevel(logging.DEBUG)
    log_handlr_file = logging.FileHandler(path.join(PATH, f"create_datasets_{datetime.datetime.now().isoformat()}.log"))
    log_handlr_file.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_hndlr_stream.setFormatter(formatter)
    log_handlr_file.setFormatter(formatter)

    dag_extraction_utils.logger.addHandler(log_hndlr_stream)
    dag_extraction_utils.logger.addHandler(log_handlr_file)
    dag_extraction_utils.logger.setLevel(logging.DEBUG)


    for ds in params['datasets']:
        dag_extraction_utils.logger.info(f"\n\n # Working on {ds.name}")
        reactant_to_reactant_id_map = dict(misc.load_json(ds.reactant_to_reactant_id_json_path))

        all_reactions = set()
        for ds_creator in ds.sources:
            dataset = ds_creator()
            reactions, *_ = dag_extraction_utils.extract_reactions(dataset)
            all_reactions.update(set(reactions))
        all_reactions = list(all_reactions)
        rng.shuffle(all_reactions)
        dag_extraction_utils.logger.info(f"Finished merging all reaction sets, left with {len(all_reactions)} total reactions.")

        mega_graph = dag_extraction_utils.create_mega_graph(all_reactions, reactant_to_reactant_id_map)
        depth_and_tree_tuples, _ = dag_extraction_utils.extract_tuple_trees_from_mega_dag(mega_graph,
                                                                                          reactant_to_reactant_id_map)
        rng.shuffle(depth_and_tree_tuples)

        nx.write_gpickle(mega_graph, path.join(PATH, f"{ds.name}-mgraph.gpickle"))
        misc.to_pickle(all_reactions, path.join(PATH, f"{ds.name}-reactions.pick"))
        indics, out_trees_dict = _split_on_proportions_and_save(ds.name, ds.split_proportions,
                                                                dag_extraction_utils.logger, depth_and_tree_tuples)
        misc.to_pickle(
            {'all_depth_and_tree_tuples': depth_and_tree_tuples,
             'subset_indices': indics},
            path.join(PATH, f"{ds.name}-all_depth_and_tree_tuples.pick"))

        _create_equiv_train_val_sets_and_save(ds.name, out_trees_dict, list(reactant_to_reactant_id_map.keys()),
                                              NAME_FOR_TRAIN)

    subprocess.run((f"cd {PATH}; shasum -a 256 * >"
                    f" {datetime.datetime.now().isoformat()}_data_checklist.sha256"),
                   shell=True)


if __name__ == '__main__':
    main(params)
