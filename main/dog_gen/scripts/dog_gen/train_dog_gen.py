
from os import path
from time import strftime, gmtime
import uuid
import os

import numpy as np
import torch
from torch.utils import data
from torch import optim
from tqdm import tqdm
from ignite.engine import Events

from syn_dags.model import doggen
from syn_dags.utils import ignite_utils
from syn_dags.utils import misc
from syn_dags.utils.settings import TOTAL_LOSS_TB_STRING, torch_device
from syn_dags.script_utils import tensorboard_helper as tb_
from syn_dags.script_utils import train_utils
from syn_dags.script_utils import doggen_utils

TB_LOGS_FILE = 'tb_logs'
CHKPT_FOLDER = 'chkpts'


class Params:
    def __init__(self):
        self.device = torch_device()

        self.train_tree_path = "../dataset_creation/data/uspto-train-depth_and_tree_tuples.pick"
        self.val_tree_path = "../dataset_creation/data/uspto-valid-depth_and_tree_tuples.pick"
        self.reactant_vocab_path = "../dataset_creation/reactants_to_reactant_id.json"

        self.num_dataloader_workers = 4
        self.num_epochs = 30
        self.batch_size = 64
        self.val_batch_size = 200
        self.learning_rate = 0.001
        self.gamma = 0.1
        self.milestones = [100, 200]
        self.expensive_ops_freq = 30

        time_run = strftime("%y-%m-%d_%H-%M-%S", gmtime())
        self.exp_uuid = uuid.uuid4()
        self.run_name = f"dog_gen_train_{time_run}_{self.exp_uuid}"
        print(f"Run name is {self.run_name}\n\n")

        self.log_for_reaction_predictor_path = path.join("logs", f"reactions-{self.run_name}.log")


def validation(model, mean_loss_fn, dataloader, prepare_batch, device, tb_logger):
    model.eval()
    summed_loss = 0
    num_items = 0
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = prepare_batch(batch, device)
            loss = mean_loss_fn(model, batch)
        nitems_batch = len(batch)
        summed_loss += nitems_batch * loss
        num_items += nitems_batch
    tb_logger.add_scalar(TOTAL_LOSS_TB_STRING, summed_loss/num_items)
    return summed_loss/num_items


def main(params: Params):
    # # Random Seeds
    rng = np.random.RandomState(4545)
    torch.manual_seed(2562)

    # # Create a folder to store the checkpoints.
    os.makedirs(path.join(CHKPT_FOLDER, params.run_name))

    # # Data
    train_trees = train_utils.load_tuple_trees(params.train_tree_path, rng)
    val_trees = train_utils.load_tuple_trees(params.val_tree_path, rng)
    print(f"Number of trees in valid set: {len(val_trees)}")
    starting_reactants = train_utils.load_reactant_vocab(params.reactant_vocab_path)

    # # Model Params
    dg_params = {
        "latent_dim": 50,
        "decoder_params": dict(gru_insize=160, gru_hsize=512, num_layers=3, gru_dropout=0.1, max_steps=100),
        "mol_graph_embedder_params": dict(hidden_layer_size=80, edge_names=["single", "double", "triple", "aromatic"],
                                          embedding_dim=160, num_layers=5),
    }

    # # Model
    model, collate_func, model_other_parts = doggen_utils.load_doggen_model(params.device, params.log_for_reaction_predictor_path,
                                            doggen_train_details=doggen_utils.DoggenTrainDetails(starting_reactants, dg_params))

    # # Dataloaders
    train_dataloader = data.DataLoader(train_trees, batch_size=params.batch_size,
                                       num_workers=params.num_dataloader_workers, collate_fn=collate_func,
                                       shuffle=True)
    val_dataloader = data.DataLoader(val_trees, batch_size=params.val_batch_size,
                                     num_workers=params.num_dataloader_workers,
                                     collate_fn=collate_func, shuffle=False)

    # # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)

    # # Tensorboard loggers
    tb_writer_train = tb_.get_tb_writer(f"{TB_LOGS_FILE}/{params.run_name}_train")
    tb_writer_train.global_step = 0
    tb_writer_train.add_hparams({**misc.unpack_class_into_params_dict(model_other_parts['hparams'], prepender="model:"),
                                 **misc.unpack_class_into_params_dict(params, prepender="train:")}, {})
    tb_writer_val = tb_.get_tb_writer(f"{TB_LOGS_FILE}/{params.run_name}_val")

    # # Create Ignite trainer
    def loss_fn(model: doggen.DogGen, x):
        # Outside the model shall compute the embeddings of the graph -- these are needed in both the encoder
        # and decoder so saves compute to just compute them once.
        x, new_order = x
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

    def setup_for_val():
        tb_writer_val.global_step = tb_writer_train.global_step  # match the steps
        model.eval()

    trainer, timers = ignite_utils.create_unsupervised_trainer_timers(
        model, optimizer, loss_fn, device=params.device, prepare_batch=prepare_batch
    )

    # # Now create the Ignite callbacks for dealing with the progressbar and performing validation etc...
    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=len(train_dataloader), desc=desc.format(0))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        tb_writer_train.global_step += 1
        tb_writer_train.add_scalar(TOTAL_LOSS_TB_STRING, engine.state.output)
        pbar.update()

    @trainer.on(Events.EPOCH_STARTED)
    def setup_trainer(engine):
        timers.reset()
        tb_writer_train.add_scalar("epoch_num", engine.state.epoch)
        tqdm.write(f"\n\n# Epoch {engine.state.epoch} starting!")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        tqdm.write(f"\n\n# Epoch {engine.state.epoch} finished")
        tqdm.write(f"## Timings:\n{str(timers)}")
        tqdm.write(f"## Validation")

        # Switch the logger for validation:
        setup_for_val()

        # Validate via teach forced loss and reconstruction stats.
        val_loss = validation(model, loss_fn, val_dataloader, prepare_batch, params.device, tb_writer_val)
        print(f"validation loss is {val_loss}")

        # Save a checkpoint
        time_chkpt = strftime("%y-%m-%d_%H:%M:%S", gmtime())
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'mol_to_graph_idx_for_reactants': model_other_parts['mol_to_graph_idx_for_reactants'],
            'run_name': params.run_name,
            'iter': engine.state.iteration,
            'epoch': engine.state.epoch,
            'model_params': dg_params,
            },
            path.join(CHKPT_FOLDER, params.run_name, f'epoch-{engine.state.epoch}_time-{time_chkpt}.pth.pick'))

        # Reset the progress bar and run the LR scheduler.
        pbar.n = pbar.last_print_n = 0
        pbar.reset()
        lr_scheduler.step()

    @trainer.on(Events.STARTED)
    def initial_validation(engine):
        tqdm.write(f"# Initial Validation")
        # Switch the logger for validation:
        setup_for_val()
        val_loss = validation(model, loss_fn, val_dataloader, prepare_batch, params.device, tb_writer_val)
        print(f"validation loss is {val_loss}")

    # # Now we can train!
    print("Beginning Training!")
    trainer.run(train_dataloader, max_epochs=params.num_epochs)
    pbar.close()


if __name__ == "__main__":
    main(Params())
