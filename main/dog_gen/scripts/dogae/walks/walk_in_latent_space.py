"""Walk in latent space
This walks randomly in latent space

Usage:
  walk_in_latent_space.py <weight_path>

"""


from time import strftime, gmtime
from os import path
import os

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from docopt import docopt

from syn_dags.script_utils import train_utils
from syn_dags.script_utils import dogae_utils
from syn_dags.utils import misc
from syn_dags.data import synthesis_trees

OUT_DIR = "out_walks"

class Params:
    def __init__(self, weight_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tuple_tree_path = "../../dataset_creation/data/uspto-train-depth_and_tree_tuples.pick"
        self.weight_path = weight_path

        self.num_starting_locations = 50
        self.num_steps_before_filtering = 30
        self.epsilon = 0.01
        self.num_unique_after = 6
        self.include_same_trees_different_order = False

        self.walk_strategy = "random"
        print(self.walk_strategy)

        time_run = strftime("%d-%b-%H-%M", gmtime())
        f_name_weights = path.splitext(path.basename(self.weight_path))[0]
        self.run_name = f"walks_for{f_name_weights}_done_{time_run}_strategy{self.walk_strategy}"
        print(f"Run name is {self.run_name}")
        print(f"Checkpoint name is {self.weight_path}")
        print(f"Tuple tree path (where we pick starting points from)  is {self.tuple_tree_path}")


def get_walk_direction_function(name):
    # Although we only consider walking randomly we leave it open to consider alternative ways to explore the latent
    # space in the future

    if name == 'random':
        # Set up function that will give direction to walk in
        def get_direction(model, current_z):
            return F.normalize(torch.randn(*current_z.shape, device=current_z.device), dim=1)
    else:
        raise NotImplementedError
    return get_direction


def main(params: Params):
    # Seeds
    rng = np.random.RandomState(564165416)
    torch.manual_seed(6514564)

    # Model!
    log_path = path.join("logs", f"reactions-{params.run_name}.log")
    model, collate_func, *_ = dogae_utils.load_dogae_model(params.device, log_path,
                                                           weight_path=params.weight_path)

    # Some starting locations
    tuple_trees = train_utils.load_tuple_trees(params.tuple_tree_path, rng)
    indices_chosen = rng.choice(len(tuple_trees), params.num_starting_locations, replace=False)
    tuple_trees = [tuple_trees[i] for i in indices_chosen]

    # Get the first embeddings
    pred_batch_largest_first, new_orders = collate_func(tuple_trees)
    pred_batch_largest_first.inplace_to(params.device)

    embedded_graphs = model.mol_embdr(pred_batch_largest_first.molecular_graphs)
    pred_batch_largest_first.molecular_graph_embeddings = embedded_graphs
    new_node_feats_for_dag = pred_batch_largest_first.molecular_graph_embeddings[pred_batch_largest_first.dags_for_inputs.node_features.squeeze(), :]
    pred_batch_largest_first.dags_for_inputs.node_features = new_node_feats_for_dag
    initial_embeddings = model._run_through_to_z(pred_batch_largest_first, sample_z=False)

    # Set up storage
    samples_out = [[] for _ in range(initial_embeddings.shape[0])]
    samples_out_set = [set() for _ in range(initial_embeddings.shape[0])]

    # walk dir
    get_direction = get_walk_direction_function(params.walk_strategy)

    # Now loop through
    last_z = initial_embeddings.detach().clone()
    while True:

        # Run for params.num_steps_before_filtering steps
        temp_samples_out_before_filtering = [[] for _ in range(initial_embeddings.shape[0])]
        for i in tqdm(range(params.num_steps_before_filtering), desc="Inner Loop..."):

            if i > 0:  # first time round just decode.
                direction = get_direction(model, last_z)
            else:
                direction = 0.

            with torch.no_grad():
                new_z = last_z + params.epsilon * direction
                new_out, _ = model.decode_from_z_no_grad(new_z, sample_x=False)
            for current_list, new_item in zip(temp_samples_out_before_filtering, new_out):
                current_list.append(new_item)
            last_z = new_z.detach().clone()

        # Work out how many unique we actually got on this time.
        for current_samples, current_samples_set, new_possible_samples in zip(samples_out, samples_out_set, temp_samples_out_before_filtering):
            for new_sample in new_possible_samples:
                new_sample: synthesis_trees.SynthesisTree
                immutable_repr = new_sample.immutable_repr(include_order=params.include_same_trees_different_order)
                if immutable_repr in current_samples_set:
                    pass
                else:
                    current_samples.append(new_sample)
                    current_samples_set.add(immutable_repr)
                    if len(current_samples) >= params.num_unique_after:
                        break

        # If we have got the number we need we can break!
        num_that_are_unique = sum([len(el) >= params.num_unique_after for el in samples_out])
        tqdm.write(f"## {num_that_are_unique} / {len(samples_out)} have found at least {params.num_unique_after} unique samples")
        if num_that_are_unique == len(samples_out):
            tqdm.write("completed while loop as collected enough samples for each.")
            break
        else:
            tqdm.write("Starting new loop!.")

    # Create the results
    order_change = np.argsort(new_orders)
    samples_out_reordered = [samples_out[i] for i in order_change ]
    syn_trees_initial_reordered = [pred_batch_largest_first.syn_trees[i] for i in order_change]

    results = dict(path_to_starting=params.tuple_tree_path,
                   indices_chosen=indices_chosen,
                   tuple_trees_in=tuple_trees,
                   syn_trees_initial=syn_trees_initial_reordered,
                   syn_trees_decoded=samples_out_reordered,
                   walk_strategy=params.walk_strategy)

    # Save to disk
    os.makedirs(path.join(OUT_DIR, params.run_name))
    fname = path.join(OUT_DIR, f"{params.run_name}.pick")
    tqdm.write(f"Saving results in {fname}")
    misc.to_pickle(results, fname)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    weight_path = arguments['<weight_path>']
    main(Params(weight_path))
    print("Done!")
