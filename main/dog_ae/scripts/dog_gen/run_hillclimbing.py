"""Run Hillclimbing (or finetuning)
This runs the optimization experiments (i.e. hill climbing) for optimizing molecules (and DAGs) for particular properties.

Usage:
  run_hillclimbing.py <weight_path> <task_name>

"""

from os import path
from time import strftime, gmtime
import uuid
import pickle
import csv

import numpy as np
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from docopt import docopt

from syn_dags.script_utils import train_utils
from syn_dags.model import doggen
from syn_dags.script_utils import doggen_utils
from syn_dags.script_utils import opt_utils
from syn_dags.utils import settings

TB_LOGS_FILE = 'tb_logs'
HC_RESULTS_FOLDER = 'hc_results'

class Params:
    def __init__(self, task_name, weight_path: str):
        self.device = settings.torch_device()

        self.train_tree_path = "../dataset_creation/data/uspto-train-depth_and_tree_tuples.pick"
        self.valid_tree_path = "../dataset_creation/data/uspto-valid-depth_and_tree_tuples.pick"

        self.weight_path = weight_path
        self.num_dataloader_workers = 4

        self.opt_name = task_name
        time_run = strftime("%y-%m-%d_%H:%M:%S", gmtime())
        self.exp_uuid = uuid.uuid4()
        self.run_name = f"doggen_hillclimbing_{time_run}_{self.exp_uuid}_{self.opt_name}"
        print(f"Run name is {self.run_name}\n\n")
        self.property_predictor = opt_utils.get_task(task_name)

        self.log_for_reaction_predictor_path = path.join("logs", f"reactions-{self.run_name}.log")


def main(params: Params):
    # # Random Seeds
    rng = np.random.RandomState(5156)
    torch.manual_seed(5115)

    # # Data
    train_trees = train_utils.load_tuple_trees(params.train_tree_path, rng)
    val_trees = train_utils.load_tuple_trees(params.valid_tree_path, rng)
    train_trees = train_trees + val_trees
    # ^ nb we add the train and valid datasets from ordinary training together now for the optimizing as is done
    # for the baselines.

    # # Model (from chkpt)
    model, collate_func, other_parts = doggen_utils.load_doggen_model(params.device, params.log_for_reaction_predictor_path,
                                                                      weight_path=params.weight_path)

    # # TensorBoard
    tb_summary_writer = SummaryWriter(log_dir=TB_LOGS_FILE)

    # # Setup functions needed for hillclimber
    def loss_fn(model: doggen.DogGen, x, new_order):
        # Outside the model shall compute the embeddings of the graph -- these are needed in both the encoder
        # and decoder so saves compute to just compute them once.
        embedded_graphs = model.mol_embdr(x.molecular_graphs)
        x.molecular_graph_embeddings = embedded_graphs
        new_node_feats_for_dag = x.molecular_graph_embeddings[x.dags_for_inputs.node_features.squeeze(), :]
        x.dags_for_inputs.node_features = new_node_feats_for_dag

        loss = model(x).mean()
        return loss

    def prepare_batch(batch, device):
        x, new_order = batch
        x.inplace_to(device)
        return x, new_order

    def create_dataloader(tuple_trees, batch_size):
        return data.DataLoader(tuple_trees, batch_size=batch_size,
                               num_workers=params.num_dataloader_workers, collate_fn=collate_func,
                               shuffle=True)

    # # Now put this together for hillclimber constructor arguments
    hparams = doggen_utils.DogGenHillclimbingParams(30, 7000, 1500)
    parts = doggen_utils.DogGenHillclimberParts(model, params.property_predictor,
                                                set(other_parts['mol_to_graph_idx_for_reactants'].keys()), rng,
                                                create_dataloader, prepare_batch, loss_fn, params.device)

    # # Now create hillclimber
    hillclimber = doggen_utils.DogGenHillclimber(parts, hparams)

    # # Run!
    print("Starting hill climber")
    sorted_tts = hillclimber.run_hillclimbing(train_trees, tb_summary_writer)

    # # Save the molecules that were queried
    print(f"Best molecule found {params.property_predictor.best_seen}")
    # Store pickle of the datastructures
    out_data = {'seen_molecules': params.property_predictor.seen_molecules,
                    'sorted_tts': sorted_tts,
                    'opt_name': params.opt_name
        }

    with open(path.join(HC_RESULTS_FOLDER, f'results_{params.run_name}.pick'), 'wb') as fo:
        pickle.dump(out_data, fo)

    # Also get score and best molecule for top 100 and add into a tsv file
    best_molecules = sorted(out_data['seen_molecules'].items(), key=lambda x: x[1], reverse=True)
    smiles_score = [(elem[0], elem[1][0]) for elem in best_molecules]
    with open(path.join(HC_RESULTS_FOLDER, f'results_{params.run_name}.tsv'), 'w') as fo:
        w = csv.writer(fo, dialect=csv.excel_tab)
        w.writerows(smiles_score[:100])

    print(f"Done run of hillclimbing for {params.opt_name}")
    return out_data


if __name__ == '__main__':
    arguments = docopt(__doc__)
    weight_path = arguments['<weight_path>']
    task_name = arguments['<task_name>']

    main(Params(task_name, weight_path))
    print("Done!")
