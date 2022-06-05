
import typing
import copy
import logging
import os
from os import path

import torch
from torch import optim
from torch import nn
from dataclasses import dataclass, field
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data import synthesis_trees
from ..data import smiles_to_feats
from ..model import doggen
from ..model import dog_decoder
from ..model import reaction_predictors
from . import opt_utils


@dataclass(order=True, frozen=True)
class ScoredTupleTree:
    tuple_tree: tuple = field(compare=False)
    score_to_maximize: float

    @property
    def root_smi(self):
        return self.tuple_tree[0]


@dataclass
class DogGenHillclimbingParams:
    n_samples_per_round: int = 7000
    n_samples_to_keep_per_round: int = 1500
    n_epochs_for_finetuning: int = 2
    batch_size: int = 64
    break_down_tuple_tree_into_parts: bool = False # this is if you want to break down the tuple tree into
    sample_batch_size: int = 200
    learning_rate: float = 1e-3
    clip_gradients: bool = True


@dataclass
class DogGenHillclimberParts:
    model: doggen.DogGen
    scorer: opt_utils.PropertyEvaluator
    reactant_vocab_set: typing.Set[str]
    rng: np.random.RandomState
    dataloader_factory: typing.Callable  # Creates a dataloader given a list of tuple trees and batch size.
    prepare_batch: typing.Callable  # prepares the batch for model (i.e. puts on GPU if necessary)
    loss_fn: typing.Callable  # model, x, new_order -> mean_loss
    device: typing.Union[str, torch.device]  # Device for Torch to use


class DogGenHillclimber:
    def __init__(self, parts: DogGenHillclimberParts, params: DogGenHillclimbingParams):
        self.parts = parts
        self.hparams = params

        self.optimizer = None
        self._num_total_train_steps_for_hc = None

    def run_hillclimbing(self, initial_tuple_trees, tb_logger: SummaryWriter):
        """
        See Alg.2 of our paper.
        :param initial_tuple_trees:  (NB all SMILES should be in canoncical form already.)
        :param tb_logger: Tensorboard logger
        """
        seen_tts: typing.List[tuple] = self.filter_out_uninteresting_trees_and_clean(initial_tuple_trees, set())
        seen_tts = seen_tts[:2000] #### limit the initial training size, the budget is 10K. 
        sorted_tts: typing.List[ScoredTupleTree] = self.score_new_trees_and_sort(seen_tts, [])
        self._report_best(sorted_tts, tb_logger, 0)

        self.optimizer = optim.Adam(self.parts.model.parameters(), lr=self.hparams.learning_rate)
        self._num_total_train_steps_for_hc = 0

        print('## Sampling before tuning...')
        sampled_dirty_tts = self.sample_from_model()
        sampled_clean_tts = self.filter_out_uninteresting_trees_and_clean(sampled_dirty_tts, sorted_tts)
        sorted_tts: typing.List[ScoredTupleTree] = self.score_new_trees_and_sort(sampled_clean_tts, sorted_tts)
        self._report_best(sorted_tts, tb_logger, 0)

        round = 0
        patience = 0
        
        while True:
            round += 1
            print(f"# Starting round {round}")

            if len(self.parts.scorer) > 100:
                self.parts.scorer.sort_buffer()
                old_scores = [item[1][0] for item in list(self.parts.scorer.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            print('## Setting up new batch for training...')
            new_batch_for_fine_tuning = [e.tuple_tree for e in sorted_tts[:self.hparams.n_samples_to_keep_per_round]]

            # print('## Starting dog_gen on new batch...')
            self.train_one_round(new_batch_for_fine_tuning, tb_logger)

            # print('## Sampling...')
            sampled_dirty_tts = self.sample_from_model()
            sampled_clean_tts = self.filter_out_uninteresting_trees_and_clean(sampled_dirty_tts, sorted_tts)
            sorted_tts: typing.List[ScoredTupleTree] = self.score_new_trees_and_sort(sampled_clean_tts, sorted_tts)
            if self.parts.scorer.finish: 
                print("max oracle calls hit, exit")
                break 
            self._report_best(sorted_tts, tb_logger, round)

            # early stopping
            if len(self.parts.scorer) > 100:
                self.parts.scorer.sort_buffer()
                new_scores = [item[1][0] for item in list(self.parts.scorer.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= 5:
                        self.parts.scorer.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

        return sorted_tts

    def train_one_round(self, tuple_trees_to_train_on: typing.List[tuple], tb_logger: SummaryWriter):
        self.parts.model.train()
        train_dataloader = self.parts.dataloader_factory(tuple_trees=tuple_trees_to_train_on,
                                                         batch_size=self.hparams.batch_size)
        for epoch in range(self.hparams.n_epochs_for_finetuning):
            # print(f"### Training epoch {epoch}")
            loss = 0.
            for data in train_dataloader:
                self.optimizer.zero_grad()
                batch = self.parts.prepare_batch(data, self.parts.device)
                loss = self.parts.loss_fn(self.parts.model, *batch)
                loss.backward()
                tb_logger.add_scalar("train_one_round_loss", loss.item(), self._num_total_train_steps_for_hc)
                if self.hparams.clip_gradients:
                    nn.utils.clip_grad_norm_(self.parts.model.parameters(), 1.0)
                self.optimizer.step()
                self._num_total_train_steps_for_hc += 1
            # print(f"loss, last batch: {loss.item()}")

    @staticmethod
    def _report_best(sorted_list: typing.List[ScoredTupleTree], tb_logger: SummaryWriter, step_num):
        print(f"Step {step_num}, Best 3 TTs so far are {sorted_list[:3]}")
        tb_logger.add_scalar("Best score So Far", sorted_list[0].score_to_maximize,
                             global_step=step_num)
        tb_logger.add_scalar("Second best score So Far", sorted_list[1].score_to_maximize,
                             global_step=step_num)
        tb_logger.add_scalar("Third best score So Far", sorted_list[2].score_to_maximize,
                             global_step=step_num)
        tb_logger.add_scalar("Mean of top 50 scores", np.mean([sorted_list[i].score_to_maximize for
                                                              i in range(50)]), global_step=step_num)

    def sample_from_model(self) -> typing.List[tuple]:
        self.parts.model.eval()
        out_list: typing.List[synthesis_trees.SynthesisTree] = []
        for _ in tqdm(range(int(np.ceil(self.hparams.n_samples_per_round / self.hparams.sample_batch_size))),
                      desc="sampling from model"):
            syn_trees, _ = self.parts.model.sample(batch_size=self.hparams.sample_batch_size)
            out_list.extend(syn_trees)
        out_tts = [e.tuple_tree_repr() for e in out_list]
        return out_tts

    def score_new_trees_and_sort(self, new_tts: typing.List[tuple],
                                 existing_tree_scores: typing.List[ScoredTupleTree]) -> typing.List[ScoredTupleTree]:
        existing_tree_scores = copy.copy(existing_tree_scores)
        # scores = self.parts.scorer.evaluate_molecules([e[0] for e in new_tts])
        try:
            # scores = self.parts.scorer([e[0] for e in new_tts]) #### smiles_list, output is np.array([xx, xxx, xxxx, ...]) 
            scores = self.parts.scorer.evaluate_molecules([e[0] for e in new_tts]) #### smiles_list, output is np.array([xx, xxx, xxxx, ...]) 
        except:
            scores = np.array(self.parts.scorer([e[0] for e in new_tts]))  

        new_scored_tts = [ScoredTupleTree(tt, score) for tt, score in zip(new_tts, scores)]
        existing_tree_scores.extend(new_scored_tts)
        return sorted(existing_tree_scores, reverse=True)

    def filter_out_uninteresting_trees_and_clean(self, list_of_tuple_trees: typing.List[tuple],
                                                 seen_tt_scores: typing.Iterable[ScoredTupleTree]) -> typing.List[tuple]:
        """
        Filters out tuple trees that have either been seen or lead to an item in the reactant set.
        Cleans tuple trees such that any unecessary parts (ie any parts that lead to a reactant and so are
        superfluous) are removed.
        """
        out = []
        invariant_seen_tts = set([synthesis_trees.SynthesisTree.make_tuple_tree_invariant(elem.tuple_tree)
                                  for elem in seen_tt_scores])
        for tt in tqdm(list_of_tuple_trees, desc="cleaning trees"):

            # Filter out reactants at top, as these we already know how to make -- just buy them!
            if tt[0] in self.parts.reactant_vocab_set:
                continue

            # If already exists then remove as already seen it
            invariant_rep = synthesis_trees.SynthesisTree.make_tuple_tree_invariant(tt)
            if invariant_rep in invariant_seen_tts:
                continue
            else:
                invariant_seen_tts.add(invariant_rep)

            # Clean out any tree parts that lead to an already reactant
            clean_tt = synthesis_trees.SynthesisTree.clean_dirty_tuple_tree(tt, self.parts.reactant_vocab_set)
            out.append(clean_tt)
        return out


@dataclass
class DoggenTrainDetails:
    starting_reactants: list
    params: dict


def load_doggen_model(device, log_path_for_react_predict, *,
                      weight_path=None, doggen_train_details: DoggenTrainDetails=None):
    """
    This utility function loads the DoG-Gen model, setting up the reaction predictor, loggers etc.
    It can be called either at initial training time or after training with a weight path in which case it will obtain
    the required parameters from the checkpoint.
    See the load_doggae_model function which works similarly for the DoG-AE model.

    :param device: eg cpu or cuda
    :param log_path_for_react_predict: where to write out the reaction predictors log
    :param weight_path: if already have a trained version of the model give path here...
    :param doggen_train_details: ... or if not provide the parameter details/starting reactants to create a _new_ model.
    """
    assert doggen_train_details is None or weight_path is None, "Should either create a new model or load an existing. Not both!"
    if weight_path is not None:
        # if using an existing model then load the checkpoint first
        chkpt = torch.load(weight_path, device)
        print(f"Loading an existing model from {weight_path}.")
        _doggen_params = chkpt['model_params']
        starting_reactants = list(chkpt['mol_to_graph_idx_for_reactants'].keys())
    else:
        print("Creating a new dog gen model")
        # otherwise unpack them from the passed parameters
        chkpt = None
        _doggen_params = doggen_train_details.params
        print(_doggen_params)
        starting_reactants = doggen_train_details.starting_reactants

    # Collate function
    collate_func = synthesis_trees.CollateWithLargestFirstReordering(starting_reactants)

    # Model components -- set up individual components
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
    model, hparams = doggen.get_dog_gen(reaction_predictor, smi2graph_func,
                                        reactant_vocab, _doggen_params)
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
