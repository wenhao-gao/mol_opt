
import typing
import copy
import logging
from dataclasses import dataclass, fields

import numpy as np
import tabulate
import torch
from torch.utils import tensorboard
from tqdm import tqdm


from ..model import dog_decoder
from ..model import dogae
from ..model import reaction_predictors
from ..data import smiles_to_feats
from ..data import synthesis_trees
from ..utils import ignite_utils
from ..utils import settings
from ..utils import misc

from autoencoders import base_ae


@dataclass
class DogaeTrainDetails:
    starting_reactants: list
    params: dict


def load_dogae_model(device, log_path_for_react_predict, *, weight_path=None, dogae_train_details: DogaeTrainDetails=None):
    """
    This utility method loads the DoG-AE model, setting up the reaction predictor, loggers etc.
    It can be called either at initial training time or after training with a weight path in which case it will obtain
    the required parameters from the checkpoint.


    :param device: eg cpu or cuda
    :param log_path_for_react_predict: where to write out the reaction predictors log
    :param weight_path: if already have a trained version of the model give path here...
    :param dogae_train_details: ... or if not provide the reactants and parameter details to create a _new_ model.
    """
    assert dogae_train_details is None or weight_path is None, "Should either create a new model or load an existing. Not both!"

    if weight_path is not None:
        # if using an existing model then load the checkpoint first
        chkpt = torch.load(weight_path, device)
        print(f"Loading an existing model from {weight_path}.")
        starting_reactants = list(chkpt['mol_to_graph_idx_for_reactants'].keys())
        _dogae_params = chkpt['dogae_params']
    else:
        # otherwise unpack them from the passed parameters
        chkpt = None
        starting_reactants = dogae_train_details.starting_reactants
        _dogae_params = dogae_train_details.params

    # Collate function
    collate_func = synthesis_trees.CollateWithLargestFirstReordering(starting_reactants)

    # Set up individual model components that are needed first
    mol_to_graph_idx_for_reactants = collate_func.base_mol_to_idx_dict.copy()
    reactant_graphs = copy.copy(collate_func.reactant_graphs)
    reactant_graphs.inplace_torch_to(device)
    reactant_vocab = dog_decoder.DOGGenerator.ReactantVocab(reactant_graphs, mol_to_graph_idx_for_reactants)

    smi2graph_func = lambda smi: smiles_to_feats.DEFAULT_SMILES_FEATURIZER.smi_to_feats(smi)
    reaction_predictor = reaction_predictors.OpenNMTServerPredictor()

    # Add a logger to the reaction predictor so can find out the reactions it predicts later.
    log_hndlr = logging.FileHandler(log_path_for_react_predict)
    log_hndlr.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_hndlr.setFormatter(formatter)
    reaction_predictor.logger.addHandler(log_hndlr)
    reaction_predictor.logger.setLevel(logging.DEBUG)
    reaction_predictor.logger.propagate = False

    # Model
    model, hparams = dogae.get_model(reaction_predictor, smi2graph_func, reactant_vocab, params=_dogae_params)
    model = model.to(device)

    # If we're reloading an existing model then load weights into model now
    if chkpt is not None:
        print("loading weights into model...")
        model.load_state_dict(chkpt['model'])

    # Collect other parts into a dictionary to return.
    other_parts = dict(
        log_hndlr=log_hndlr, hparams=hparams, chkpt=chkpt, mol_to_graph_idx_for_reactants=mol_to_graph_idx_for_reactants
    )

    return model, collate_func, other_parts


@torch.no_grad()
def sample_n_from_prior(ae, n, rng: np.random.RandomState, return_extras=False):
    ae.eval()
    latent_dim = ae.latent_prior.mean_log_var[0].shape[1]
    z = torch.tensor(rng.randn(n, latent_dim), dtype=settings.TORCH_FLT).to(next(iter(ae.parameters())).device)
    x, log_probs = ae.decode_from_z_no_grad(z)
    out = [tree.nx_to_tuple_tree(tree.tree, tree.root_smi) for tree in x]
    if return_extras:
        return out, z, log_probs
    else:
        return out


@torch.no_grad()
def sample_ntimes_using_z_and_sort(ae: base_ae.SingleLatentWithPriorAE, n: int,
                                   z: torch.Tensor, return_top_only=False):
    """
    Samples n times from autoencoder, then revaluate the likelihood of each of the samples and sorts based on this.

    :param z: [b, ...]
    List[List], [batch_size, num_samples]
    """
    # Get the samples!
    sample_log_prob_tuples = [ae.decode_from_z_no_grad(z, sample_x=True) for _ in tqdm(range(n), desc="sampling...")]

    # Then separate them into the trees and the log probs
    syn_tree_samples, log_probs = zip(*sample_log_prob_tuples)

    # Rearrange so that the samples of the same z location are together
    log_probs = torch.stack(list(log_probs), dim=0)  # [num_samples, seq_size, batch_size]
    samples_grouped = list(zip(*syn_tree_samples))

    # we sum the log probs over the sequence and sort over the different samples
    log_probs = torch.sum(log_probs, dim=1).transpose(0,1)  # [batch_size, num_samples]
    sorted_log_probs, indices = torch.sort(log_probs, dim=1, descending=True)

    # we then sort the synthesis trees the same way
    synthesis_trees_sorted = []
    for trees_in_old_order, new_indices in zip(samples_grouped, indices):
        synthesis_trees_sorted.append([trees_in_old_order[i] for i in new_indices])

    # if return_top_only flag is set then we will only return the top synthesis tree from each element.
    if return_top_only:
        out = [elem[0] for elem in synthesis_trees_sorted]
        return out

    return synthesis_trees_sorted, sorted_log_probs


@dataclass
class ReconstructionAccResults:
    root_node_match: float = 0
    tree_match: float = 0
    graph_edit_distance: float = 0.
    jacard_similarity_nodes: float = 0.
    jacard_similarity_reactants: float = 0.

    def set_to_average(self, total_num):
        for field in fields(self):
            field_name = field.name
            setattr(self, field_name, getattr(self, field_name) / total_num)


def evaluate_reconstruction_accuracy(x_in: synthesis_trees.PredOutBatch,
                                     syn_trees_out: typing.List[synthesis_trees.SynthesisTree]) -> ReconstructionAccResults:
    """
    Computes a series of reaconstruction accuracy metrics given the initial and reconstruceted DAGs,

    :param x_in: the batch fed in.
    :param syn_trees_out: the reconstructed batch
    """
    out = ReconstructionAccResults()

    for i, (tree_in, tree_out) in enumerate(zip(x_in.syn_trees, syn_trees_out)):
        out.root_node_match += int(tree_in.root_smi == tree_out.root_smi)
        out.tree_match += int(tree_in.compare_with_other_ismorphism(tree_out))
        out.graph_edit_distance += tree_in.compare_with_other_graph_edit(tree_out)
        j_reactants, j_nodes = tree_in.compare_with_other_jacard(tree_out)
        out.jacard_similarity_nodes += j_nodes
        out.jacard_similarity_reactants += j_reactants

    total_num = float(len(syn_trees_out))
    out.set_to_average(total_num)
    return out


def _print_first_n_tree_reconstructions_to_tensorboard(x_in: synthesis_trees.PredOutBatch,
                                                       syn_trees_out: typing.List[synthesis_trees.SynthesisTree],
                                                       tb_writer, n=30):

    for i, (tree_in, tree_out) in enumerate(zip(x_in.syn_trees, syn_trees_out)):

        misc.try_but_pass(lambda : tb_writer.add_text(f"reconstructed-tuple-trees-{i}",
                                  f"in: `{str(tree_in.tuple_tree_repr())}`   ; out: `{str(tree_out.tuple_tree_repr())}`"),
                          Exception, True)

        misc.try_but_pass(lambda : tb_writer.add_text(f"reconstructed-seqs-{i}",
                                      f"in: `{tree_in.text_for_construction(strict_mode=True)}`   ; "
                                      f"out: `{tree_out.text_for_construction()}`"),
                          Exception, True)

        if i >= n:
            break


@torch.no_grad()
def validation(val_dataloader, ae, tb_writer_val, device, kw_args_for_ae,
               run_expensive_ops: bool=True):
    """
    Runs validation on validation dataset with printing to Tensorboard and to stdout.

    :param val_dataloader: Dataloader for validation dataset
    :param ae: autoencoder model
    :param tb_writer_val: TensorBoard logger
    :param device: device to use (eg cuda or cpu)
    :param kw_args_for_ae: any other arguments to feed into autoencoder
    :param run_expensive_ops: flag for whether to run the more expensive (greedy sampled reconstruction)
    """
    # # Print out performing loop
    print(f"Performing a validation loop. "
          f"\n Kw args: {kw_args_for_ae}"
          f"\n Device: {device}"
          f"\n Running expensive ops: {run_expensive_ops}")

    # # Set model into validation mode
    ae.eval()

    # # We are going to record a series of measurements that we will average after we have gone through the whole data:
    meters = {settings.TOTAL_LOSS_TB_STRING: ignite_utils.AverageMeter()}
    if run_expensive_ops:
        meters.update({k.name:ignite_utils.AverageMeter() for k in fields(ReconstructionAccResults)})

    # # Set up some lists to store the results from the validation run.
    in_trees_out_trees = []

    # # Now we will iterate through the dataloader.
    with tqdm(val_dataloader, total=len(val_dataloader)) as t:
        for i, (x, _) in enumerate(t):
            # Set up the data
            x: synthesis_trees.PredOutBatch
            x.inplace_to(device)
            batch_size = x.batch_size

            # Get graph embeddings and set these as features -- do this outside so can be reused
            embedded_graphs = ae.mol_embdr(x.molecular_graphs)
            x.molecular_graph_embeddings = embedded_graphs
            new_node_feats_for_dag = x.molecular_graph_embeddings[x.dags_for_inputs.node_features.squeeze(), :]
            x.dags_for_inputs.node_features = new_node_feats_for_dag

            # Evaluate reconstruction accuracy -- if the flag is set
            meter_name_update_values = []
            if run_expensive_ops:
                # a. Perform the reconstruction:
                reconstruction_trees, _ = ae.reconstruct_no_grad(x)

                # b. Compute the results metrics:
                recon_acc_results = evaluate_reconstruction_accuracy(x, reconstruction_trees)
                if i == 0: _print_first_n_tree_reconstructions_to_tensorboard(x, reconstruction_trees, tb_writer_val)

                # c. Add to output.
                in_trees_out_trees.extend(list(zip(x.syn_trees, reconstruction_trees)))
                meter_name_update_values.extend([(fie.name, getattr(recon_acc_results, fie.name))
                                                 for fie in fields(recon_acc_results)])

            # Compute the loss using the teacher forcing method.
            loss = -ae(x, **kw_args_for_ae).mean()

            # Update the meters that record the various statistics.
            meter_name_update_values = [(settings.TOTAL_LOSS_TB_STRING, loss)] + meter_name_update_values
            for meter_name, value in meter_name_update_values:
                meters[meter_name].update(value, n=batch_size)

            # Update the stats in the progress bar
            t.set_postfix(**{k: f"{v.avg:.4E}" for k,v in meters.items()})

    # # Print out the final averages and add them to TB:
    _print_out_val_meters_and_add_to_tb(meters, tb_writer_val)

    # # Format the output variables.
    return meters, in_trees_out_trees


def _print_out_val_meters_and_add_to_tb(meters, tb_writer_val):

    out = {k:v.avg for k,v in meters.items()}
    if tb_writer_val is not None:
        for k, v in out.items():
            tb_writer_val.add_scalar(k, v)

    print(f"\n## Validation finished over a total of {meters[settings.TOTAL_LOSS_TB_STRING].count} items.")
    print(tabulate.tabulate(list(out.items()), tablefmt="simple", floatfmt=".4f"))
    print("===============================================")
