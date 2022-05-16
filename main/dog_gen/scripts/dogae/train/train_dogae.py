
from os import path
from time import strftime, gmtime
import uuid
import os

import requests
import numpy as np
import torch
from torch.utils import data
from torch import optim
from tqdm import tqdm

from ignite.engine import Events

from autoencoders import logging_tools

from syn_dags.script_utils import train_utils
from syn_dags.utils import ignite_utils
from syn_dags.utils import misc
from syn_dags.script_utils import tensorboard_helper as tb_
from syn_dags.script_utils import dogae_utils
from syn_dags.utils.settings import TOTAL_LOSS_TB_STRING

TB_LOGS_FILE = 'tb_logs'
CHKPT_FOLDER = 'chkpts'

class Params:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_tree_path = "../../dataset_creation/data/uspto-train-depth_and_tree_tuples.pick"
        self.val_tree_path = "../../dataset_creation/data/uspto-valid-depth_and_tree_tuples.pick"
        self.reactant_vocab_path = "../../dataset_creation/reactants_to_reactant_id.json"

        self.num_dataloader_workers = 4
        self.batch_size = 64
        self.val_batch_size = 400
        self.learning_rate = 0.001
        self.log_interval_histograms = 100
        self.num_epochs = 400
        self.gamma = 0.1
        self.milestones = [300, 350]
        self.lambda_value = 10.  # see WAE paper, section 4
        self.expensive_ops_freq = 25

        time_run = strftime("%y-%m-%d_%H-%M-%S", gmtime())
        self.exp_uuid = uuid.uuid4()
        self.run_name = f"train_dogae_run_{time_run}_{self.exp_uuid}"
        print(f"Run name is {self.run_name}\n\n")

        self.log_for_reaction_predictor_path = path.join("logs", f"reactions-{self.run_name}.log")


def main(params: Params):
    # # Random seeds
    rng = np.random.RandomState(4545)
    torch.manual_seed(2562)

    # # Data
    train_trees = train_utils.load_tuple_trees(params.train_tree_path, rng)
    val_trees = train_utils.load_tuple_trees(params.val_tree_path, rng)
    print(f"Number of trees in valid set: {len(val_trees)}")
    starting_reactants = train_utils.load_reactant_vocab(params.reactant_vocab_path)

    # # Model Params
    _dogae_params = {'latent_dim': 25,
                     'mol_graph_embedder_params': {'hidden_layer_size': 80,
                                                   'edge_names': ['single', 'double', 'triple', 'aromatic'],
                                                   'embedding_dim': 50,
                                                   'num_layers': 4},
                     'dag_graph_embedder_gnn_params': {'hlayer_size': 50,
                                                       'edge_names': ['reactions'],
                                                       'num_layers': 7},
                     'dag_embedder_aggr_type_s': 'FINAL_NODE',
                     'decoder_params': {'gru_insize': 50,
                                        'gru_hsize': 200,
                                        'num_layers': 3,
                                        'gru_dropout': 0.1,
                                        'max_steps': 100},
                     }

    # # Model
    model, collate_func, model_other_parts = dogae_utils.load_dogae_model(params.device, params.log_for_reaction_predictor_path,
                                 dogae_train_details=dogae_utils.DogaeTrainDetails(starting_reactants, _dogae_params))

    # # Dataloaders
    train_dataloader = data.DataLoader(train_trees, batch_size=params.batch_size,
                                       num_workers=params.num_dataloader_workers, collate_fn=collate_func,
                                       shuffle=True)
    val_dataloader = data.DataLoader(val_trees, batch_size=params.val_batch_size, num_workers=params.num_dataloader_workers,
                                     collate_fn=collate_func, shuffle=False)

    # # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)

    # # Create a folder to store the checkpoints.
    os.makedirs(path.join(CHKPT_FOLDER, params.run_name))

    # # Tensorboard loggers
    tb_writer_train = tb_.get_tb_writer(f"{TB_LOGS_FILE}/{params.run_name}_train")
    tb_writer_train.global_step = 0
    tb_writer_train.add_hparams({**misc.unpack_class_into_params_dict(model_other_parts['hparams'], prepender="model:"),
                                **misc.unpack_class_into_params_dict(params, prepender="train:")}, {})
    tb_writer_val = tb_.get_tb_writer(f"{TB_LOGS_FILE}/{params.run_name}_val")
    def add_details_to_train(dict_to_add):
        for name, value in dict_to_add.items():
            tb_writer_train.add_scalar(name, value)
    train_log_helper = logging_tools.LogHelper([add_details_to_train])

    # # Create Ignite trainer
    def loss_fn(model, x):
        # Note that outside the model shall compute the embeddings of the graph
        # -- these are needed in both the encoder
        # and decoder so saves compute to just compute them once.
        x, new_order = x
        embedded_graphs = model.mol_embdr(x.molecular_graphs)
        x.molecular_graph_embeddings = embedded_graphs
        new_node_feats_for_dag = x.molecular_graph_embeddings[x.dags_for_inputs.node_features.squeeze(),:]
        x.dags_for_inputs.node_features = new_node_feats_for_dag

        # Then we can run the model forward.
        loss = -model(x, lambda_=params.lambda_value).mean()

        return loss

    def prepare_batch(batch, device):
        x, new_order = batch
        x.inplace_to(device)
        return x, new_order

    def setup_for_val():
        tb_writer_val.global_step = tb_writer_train.global_step  # match the steps between the train and val tensorboards
        model._logger_manager = None  # turn off the more precise logging for when we go through validation set/sample.

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
        if engine.state.iteration % params.log_interval_histograms == 0:
            # Every 100 steps we store the histograms of our sampled z's to ensure not getting posterior collapse
            model.encoder.shallow_dist._tb_logger = tb_writer_train  # turn it on for this step
        else:
            model.encoder.shallow_dist._tb_logger = None # otherwise we do not save this due to speed.

        pbar.update()

    @trainer.on(Events.EPOCH_STARTED)
    def setup_trainer(engine):
        timers.reset()
        model._logger_manager = train_log_helper
        tb_writer_train.add_scalar("epoch_num", engine.state.epoch)
        tqdm.write(f"\n\n# Epoch {engine.state.epoch} starting!")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        """
        This callback does validation at the end of a training epoch.
        
        Note we have two kinds of validation: simple and expensive. The simple runs the same loss calculation we use in
        training (i.e. with teacher forcing), so that it can evaluate the whole sequence at once. This means it runs
         quickly. On the other hand the expensive evaluation runs slower operations. It does greedy reconstruction
        of the sequence, always feeding in the previous chosen action as input to the next time steps. New reactions
        have to be predicted by calling the reaction predictor oracle. We also sample from the model at new places in
        latent space, which also requires the evaluation of one step at a time and calls to the Transformer.
        Given the expense of doing this form of validation it is done less frequently.
        """
        tqdm.write(f"\n\n# Epoch {engine.state.epoch} finished")
        tqdm.write(f"## Timings:\n{str(timers)}")
        tqdm.write(f"## Validation")

        # Setup for validation
        setup_for_val()
        run_expensive_ops_flag = (engine.state.epoch % params.expensive_ops_freq) == 0
        # ^ we will only do the ops that involve sampling infrequently to save constantly bombarding the server.

        # ## Main validation code

        def val_func():
            # First look at performance on validation dataset
            dogae_utils.validation(val_dataloader, model, tb_writer_val, params.device, {'lambda_': params.lambda_value},
                                   run_expensive_ops_flag)

            # Then create some samples!
            if run_expensive_ops_flag:
                out_tuple_trees = dogae_utils.sample_n_from_prior(model, 10, rng)
                tuple_trees_as_text = ' ;\n'.join(map(str, out_tuple_trees))
                tb_writer_train.add_text("tuple_trees_sampled", f"```{tuple_trees_as_text}```")

        misc.try_but_pass(val_func, requests.exceptions.Timeout, False)
        # ^ can have problems with reaction server so continue
        # training for now and ignore validation (can do it from the checkpoints later).

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
            'lambda': params.lambda_value,
            'dogae_params': _dogae_params,
            },
            path.join(CHKPT_FOLDER, params.run_name, f'time-{time_chkpt}_epoch-{engine.state.epoch}.pth.pick'))

        # Reset the progress bar and run the LR scheduler.
        pbar.n = pbar.last_print_n = 0
        pbar.reset()
        lr_scheduler.step()

    @trainer.on(Events.STARTED)
    def initial_validation(engine):
        tqdm.write(f"# Initial Validation")

        # Switch the logger for validation:
        setup_for_val()

        dogae_utils.validation(val_dataloader, model, tb_writer_val, params.device,
                               {'lambda_': params.lambda_value})  # run before start training.

    # # Now we can train!
    print("Beginning Training!")
    trainer.run(train_dataloader, max_epochs=params.num_epochs)
    pbar.close()


if __name__ == '__main__':
    main(Params())
    print("Done!")
