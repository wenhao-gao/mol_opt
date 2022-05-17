
import enum
import typing
import collections

from dataclasses import dataclass
import multiset
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from torch.distributions import gumbel
import numpy as np
import networkx as nx


from autoencoders.dist_parameterisers import base_parameterised_distribution
from autoencoders import similarity_funcs
from graph_neural_networks.sparse_pattern import graph_as_adj_list as grphs

from . import dag_embedder
from . import molecular_graph_embedder as mol_graph_embedder
from . import reaction_predictors
from ..utils import settings
from ..utils import torch_utils
from ..data import synthesis_trees


@dataclass
class DecoderParams:
    gru_insize: int
    gru_hsize: int
    num_layers: int
    gru_dropout: float

    max_steps: int  # max number of steps for creating a sequence out.


class DecoderPreDefinedNetworks(nn.Module):
    def __init__(self,
                 mol_embdr: mol_graph_embedder.GraphEmbedder,  # embeds the molecular graphs
                 f_z_to_h0: nn.Module,  # function from the input to the initial settings of the hidden layers.

                 f_ht_to_e_add: nn.Module,
                 f_ht_to_e_reactant: nn.Module,
                 f_ht_to_e_edge: nn.Module):
        super().__init__()
        self.mol_embdr = mol_embdr
        self.f_z_to_h0 = f_z_to_h0
        self.f_ht_to_e_add = f_ht_to_e_add
        self.f_ht_to_e_reactant = f_ht_to_e_reactant
        self.f_ht_to_e_edge = f_ht_to_e_edge


class ExtraSymbols(nn.Module):
    class _InitTypes(enum.Enum):
        UNIFORM = "uniform_sample"
        ZERO = "zero"

    def __init__(self,
                 action_embedding_size):
        super().__init__()

        self.action_embedding_size = action_embedding_size
        self.add_step_action_embeddings = self.create_and_return(2, action_embedding_size)
        # ^ first row reactant, second product

        self.edge_action_stop = self.create_and_return(2, action_embedding_size)
        # first row stop intermediate, second row stop final

    @staticmethod
    def create_and_return(*sizes, init_type=_InitTypes.UNIFORM):
        param = nn.Parameter(torch.Tensor(*sizes).to(settings.TORCH_FLT))
        if init_type.UNIFORM:
            bound = 1 / np.sqrt(sizes[-1])
            nn.init.uniform_(param, -bound, bound)
        elif init_type.ZERO:
            nn.init.constant_(param, 0.)
        else:
            raise NotImplementedError
        return param


class DOGGenerator(base_parameterised_distribution.BaseParameterisedDistribution):
    @dataclass
    class ReactantVocab:
        molecular_graphs: grphs.DirectedGraphAsAdjList
        mol_to_graph_idx: collections.OrderedDict

    def __init__(self, params: DecoderParams,
                 other_nets: DecoderPreDefinedNetworks,
                 react_pred: reaction_predictors.AbstractReactionPredictor,
                 smi2graph: typing.Callable[[str], grphs.DirectedGraphAsAdjList],
                 reactant_vocab: ReactantVocab, rng: np.random.RandomState=None):
        super().__init__()

        self.rng = rng if rng is not None else np.random.RandomState(6215)

        self.params = params
        self.gru = nn.GRU(input_size=params.gru_insize, hidden_size=params.gru_hsize,
                          num_layers=params.num_layers, dropout=params.gru_dropout)
        self.other_nets = other_nets
        self.react_pred = react_pred

        action_embedding_dim = self.other_nets.mol_embdr.embedding_dim
        # ^ get the action embeddings to match the molecular graph embedding sizes
        self.learnt_symbols = ExtraSymbols(action_embedding_dim)

        self._z_sample = None
        self._initial_hidden_after_update = None
        self.reactant_vocab = reactant_vocab
        self.smi2graph = smi2graph

    def _compute_initial_h(self, z_sample):
        h_samples = self.other_nets.f_z_to_h0(z_sample)
        initial_hidden = h_samples.unsqueeze(0).repeat(self.params.num_layers, 1, 1)
        return initial_hidden  # [num_layers, b, h]

    def update(self, input_):
        self._z_sample = input_
        self._initial_hidden_after_update = self._compute_initial_h(self._z_sample)
        return self

    def mode(self):
        def sample_func(logits):
            mode = torch.argmax(logits, dim=-1)
            log_prob = -F.cross_entropy(logits,mode, reduction='none')

            return mode, log_prob

        return self._run_forward_via_sampling(sample_func)

    def sample_no_grad(self, num_samples: int = 1):
        samples = []

        def sample_func(logits):
            gumbel_dist = gumbel.Gumbel(0, 1)
            logits_plus_gumbel_noise = logits + \
                                       gumbel_dist.sample(sample_shape=logits.shape).to(str(logits.device))
            mode = torch.argmax(logits_plus_gumbel_noise, dim=-1)
            log_prob = -F.cross_entropy(logits, mode, reduction='none')
            return mode, log_prob

        with torch.no_grad():
            for _ in range(num_samples):
                samples.append(self._run_forward_via_sampling(sample_func))
        return samples

    def get_reactant_embeddings(self) -> torch.Tensor:
        return self.other_nets.mol_embdr(self.reactant_vocab.molecular_graphs)

    def nlog_like_of_obs(self, obs: synthesis_trees.PredOutBatch):
        """
        Here we calculate the negative log likelihood of the sequence. For each  we feed in the previous observation
        ie if you use this function during training then doing teacher forcing.
        """
        molecular_graph_embeddings_incl_stops = torch.cat([obs.molecular_graph_embeddings, self.learnt_symbols.edge_action_stop])

        # --> Form the inputs for the RNN by using previous action.
        # --> Working out the correct action embeddings
        prev_action_choice_ = torch_utils.remove_last_from_packed_seq(obs.sequence_choices)
        prev_action_kind_ = torch_utils.remove_last_from_packed_seq(obs.sequence_action_kinds)

        out_data_minus_first = torch.zeros(prev_action_choice_.data.shape[0], self.learnt_symbols.action_embedding_size,
                               dtype=settings.TORCH_FLT, device=str(molecular_graph_embeddings_incl_stops.device))
        # ----> Add steps
        add_step_locations = prev_action_kind_.data == synthesis_trees.ADD_STEP_VAL
        add_step_choice = prev_action_choice_.data[add_step_locations]
        add_step_embeddings = self.learnt_symbols.add_step_action_embeddings[add_step_choice, :]
        out_data_minus_first[add_step_locations, :] = add_step_embeddings
        # ----> reactant add steps and edge add steps both index graphs:
        grph_locations = prev_action_kind_.data != synthesis_trees.ADD_STEP_VAL
        grph_step_choice = prev_action_choice_.data[grph_locations]
        grph_step_embeddings = molecular_graph_embeddings_incl_stops[grph_step_choice, :]
        out_data_minus_first[grph_locations, :] = grph_step_embeddings

        input_data = rnn.PackedSequence(out_data_minus_first, prev_action_choice_.batch_sizes)

        # --> Feed through the RNN
        initial_hidden = self._initial_hidden_after_update
        outputs, _ = self.gru(input_data, initial_hidden)

        # --> Take out and feed through the in individual networks.
        out_packed_losses_data = torch.zeros(outputs.data.shape[0],
                               dtype=settings.TORCH_FLT, device=obs.molecular_graph_embeddings.device)

        # --> Get out the actions we need to predict and the relevant possible edge masks. Note we are not interested
        # in the first item as it is ALWAYS add a reactant node.
        section_action_kinds_to_predict = torch_utils.remove_first_from_packed_seq(obs.sequence_action_kinds)
        section_actions_to_predict = torch_utils.remove_first_from_packed_seq(obs.sequence_choices)
        edge_masks = torch_utils.remove_first_from_packed_seq(obs.sequence_masks_for_edge_steps)
        reactant_masks = torch_utils.remove_first_from_packed_seq(obs.sequence_masks_for_reactant_steps)

        # ----> Add steps
        out_add_step_locs = section_action_kinds_to_predict.data == synthesis_trees.ADD_STEP_VAL
        add_steps_embeddings = self.other_nets.f_ht_to_e_add(outputs.data[out_add_step_locs, :])
        add_step_logits = dot_product_similarity(add_steps_embeddings, self.learnt_symbols.add_step_action_embeddings)
        add_step_losses = F.cross_entropy(add_step_logits, section_actions_to_predict.data[out_add_step_locs], reduction='none')
        out_packed_losses_data[out_add_step_locs] = add_step_losses

        # ----> Building block choose step
        out_reactant_choose_step_locs = section_action_kinds_to_predict.data == synthesis_trees.REACTANT_CHOOSE_STEP_VAL
        reactant_step_embeddings = self.other_nets.f_ht_to_e_reactant(outputs.data[out_reactant_choose_step_locs, :])
        reactant_step_logits = dot_product_similarity(reactant_step_embeddings,
                                                      obs.molecular_graph_embeddings[:obs.num_that_are_initial_reactants, :])
        reactant_step_losses = torch_utils.masked_softmax_nll(reactant_step_logits,
                                                              section_actions_to_predict.data[out_reactant_choose_step_locs],
                                                              reactant_masks.data[out_reactant_choose_step_locs])
        out_packed_losses_data[out_reactant_choose_step_locs] = reactant_step_losses

        # ----> Edge add step values
        out_edge_add_step_locs = section_action_kinds_to_predict.data == synthesis_trees.EDGE_ADD_STEP_VAL
        edge_step_embeddings = self.other_nets.f_ht_to_e_edge(outputs.data[out_edge_add_step_locs, :])
        edge_step_logits = dot_product_similarity(edge_step_embeddings, molecular_graph_embeddings_incl_stops)
        edge_step_losses = torch_utils.masked_softmax_nll(edge_step_logits,
                                                          section_actions_to_predict.data[out_edge_add_step_locs],
                                                          edge_masks.data[out_edge_add_step_locs])
        out_packed_losses_data[out_edge_add_step_locs] = edge_step_losses

        # --> Pack then pad and take the sum...
        packed_losses = rnn.PackedSequence(out_packed_losses_data, section_action_kinds_to_predict.batch_sizes)
        padded_losses, _ = rnn.pad_packed_sequence(packed_losses, batch_first=True, padding_value=0.)  # [B, S_max]
        losses = torch.sum(padded_losses, dim=1)
        return losses

    def convolve_with_function(self, obs,
                               function: similarity_funcs.BaseSimilarityFunctions) -> torch.Tensor:
        # With the WAE you are minimising an optimal transport cost, usually using the squared Euclidean distance
        # to measure distance between points. Using this metric on real output means that the first term of the WAE loss
        # matches the usual log likelihood term you would get in a VAE and it is only the second KL diveregence term that
        # is  different. Our task is more like classification where we are using a post softmax vector to represent the
        # probability of picking different actions. So we shall still keep the first term of the WAE the same as it
        # would be in a VAE (even if this does not correspond to an optimal transport loss in our case) as we think it
        # is a sensible way to penalise reconstruction.
        return self.nlog_like_of_obs(obs)

    @torch.no_grad()
    def _run_forward_via_sampling(self, sampling_func):
        """
        This runs the decoder forward conditioning on the samples produced so far by the sampling function.
        It returns a list of synthesis trees.

        NOTE THAT THIS IS NOT ORDERED BY SEQUENCE SIZE
        """
        # --> Create the initial variables that we need
        hidden = self._initial_hidden_after_update
        batch_size = hidden.shape[1]
        mol_features_reactants = self.get_reactant_embeddings()
        inter_node_mngr = _IntermediateManager(self.reactant_vocab.mol_to_graph_idx, mol_features_reactants,
                                               self.learnt_symbols.edge_action_stop, batch_size, self.smi2graph,
                                               self.other_nets.mol_embdr)


        # --> Create the tensors etc to store the actions, which will later be converted into each of the synthesis trees.
        action_choices = torch.full((self.params.max_steps, batch_size), settings.PAD_VALUE,
                                                                    dtype=settings.TORCH_INT,
                                                                    device=hidden.device)
        action_kinds = torch.full((self.params.max_steps+1, batch_size), settings.PAD_VALUE,
                                    dtype=settings.TORCH_INT,
                                    device=hidden.device)
        log_probs = torch.zeros_like(action_choices, dtype=settings.TORCH_FLT)

        # Note we always start by adding a building block so fill in this step below:
        action_kinds[0, :] = synthesis_trees.ADD_STEP_VAL
        action_choices[0, :] = synthesis_trees.ADD_STEP_CHOOSE_REACTANT
        action_kinds[1, :] = synthesis_trees.REACTANT_CHOOSE_STEP_VAL
        lengths = torch.ones(batch_size, dtype=settings.TORCH_INT, device=hidden.device)

        # --> Set up initial x and the indices we are interested in.
        x = self.learnt_symbols.add_step_action_embeddings[synthesis_trees.ADD_STEP_CHOOSE_REACTANT, :].unsqueeze(0).repeat(1, batch_size, 1)
        # ^ [1, b, h]
        cont_indcs_into_original = torch.arange(batch_size, dtype=settings.TORCH_INT, device=hidden.device)

        # --> Go through and iterate through time steps, sampling a sequence.
        # Start at step one as we know we pick add reactant  node on the first step.
        for i in range(1, self.params.max_steps):
            # ----> All the current ones will increase by size 1
            lengths[cont_indcs_into_original] += 1

            # ----> Run through one step and sample the values for each batch member:
            op, hidden = self.gru(x, hidden)  # op [1, b', h]
            op = op.squeeze(0)   # op [ b', h]
            x_next_time = torch.full_like(x, settings.PAD_VALUE).squeeze(0)  # [b, h]
            x_type = action_kinds[i, cont_indcs_into_original]
            continue_mask = torch.ones_like(cont_indcs_into_original, dtype=torch.bool)

            # ----> Map through the required MLPs
            for computed_op in (
                    # add new node:
                    self._sample_add_step_points(x_type, op, sampling_func, cont_indcs_into_original),
                    # reactant choice:
                    self._sample_reactant_choose_step_points(x_type, op, sampling_func, cont_indcs_into_original
                        , mol_features_reactants, inter_node_mngr),
                    # edge choice:
                    self._sample_edge_add_step_points(x_type, op, sampling_func, cont_indcs_into_original,
                                                      inter_node_mngr)):
                mask_locations, actions_chosen, log_probs_this_action, actions_kind_next_time, new_x_parts, continuing_mask = computed_op
                if actions_chosen is not None:
                    action_choices[i, cont_indcs_into_original[mask_locations]] = actions_chosen
                    action_kinds[i + 1, cont_indcs_into_original[mask_locations]] = actions_kind_next_time
                    x_next_time[mask_locations] = new_x_parts
                    continue_mask[mask_locations] = continuing_mask
                    log_probs[i, cont_indcs_into_original[mask_locations]] = log_probs_this_action

            cont_indcs_into_original = cont_indcs_into_original[continue_mask]

            # if all the sequences in the batch have finished then we can break early!
            if not cont_indcs_into_original.shape[0] > 0:
                break

            x = x_next_time[continue_mask]
            x = x.unsqueeze(0)  # need to add a time sequence dimension. so [1, b', h]
            hidden = hidden[:, continue_mask, :]  # [1, b', h]

        syn_trees = inter_node_mngr.close_off_unconnected_and_return_final_trees(self._run_forward_reaction_and_get_new_products,
                                                                                 rng=self.rng)

        return syn_trees, log_probs

    def _run_forward_reaction_and_get_new_products(self, reactant_sets):
        try:
            products_sets = self.react_pred(reactant_sets)
        except Exception as ex:
            print("reaction predictor failed.")
            print(ex)
            print(reactant_sets)
            raise ex
        assert all([len(e) == 1 for e in products_sets])
        products = [next(iter(e)) for e in products_sets]
        return products

    def _filter_and_sample_points(self, x_type, op_from_rnn, sampling_func, step_kind_val, embeddings_for_options,
                                  cont_indcs_into_original, indcs2logitmasks=None):
        net_to_use = {
            synthesis_trees.ADD_STEP_VAL: self.other_nets.f_ht_to_e_add,
            synthesis_trees.REACTANT_CHOOSE_STEP_VAL: self.other_nets.f_ht_to_e_reactant,
            synthesis_trees.EDGE_ADD_STEP_VAL: self.other_nets.f_ht_to_e_edge
        }[step_kind_val]

        mask = x_type == step_kind_val
        if mask.sum():
            # ^ ie in at least one location.
            embeddings = net_to_use(op_from_rnn[mask, :])
            logits = dot_product_similarity(embeddings, embeddings_for_options)
            if indcs2logitmasks is not None:
                logit_masks = indcs2logitmasks(cont_indcs_into_original[mask])
                logits[~logit_masks] = -np.inf  # switch off the locations which do not have the mask on.
            sampled_actions, log_probs = sampling_func(logits)
        else:
            # If there are none of those steps this iteration return None and the mask (which should be all False)
            sampled_actions = None
            log_probs = None
        return mask, sampled_actions, log_probs

    def _sample_add_step_points(self, x_type, op, sampling_func, cont_indcs_into_original):

        mask, actions, log_probs = self._filter_and_sample_points(x_type, op, sampling_func, synthesis_trees.ADD_STEP_VAL,
                                                               self.learnt_symbols.add_step_action_embeddings,
                                                       cont_indcs_into_original)
        if actions is None:
            new_action_kind, new_x_parts, continue_mask = None, None, None
        else:
            # on a add  step we can either add a reactant or an edge symbol for the next action
            # (if we add a stop then wont continue anyway)
            new_action_kind = torch.zeros_like(actions, dtype=settings.TORCH_INT)
            new_action_kind[
                actions == synthesis_trees.ADD_STEP_CHOOSE_REACTANT] = synthesis_trees.REACTANT_CHOOSE_STEP_VAL
            new_action_kind[
                actions == synthesis_trees.ADD_STEP_CHOOSE_PRODUCT] = synthesis_trees.EDGE_ADD_STEP_VAL

            new_x_parts = self.learnt_symbols.add_step_action_embeddings[actions]

            continue_mask = torch.ones_like(actions, dtype=torch.bool)
        return mask, actions, log_probs, new_action_kind, new_x_parts, continue_mask

    def _sample_reactant_choose_step_points(self, x_type, op, sampling_func, cont_indcs_into_original, reactant_feats,
                                            inter_node_mngr):
        def get_reactant_mask_func(indcs_to_get_mask_for):
            reactant_masks = inter_node_mngr.get_current_reactant_masks(indcs_to_get_mask_for)
            return reactant_masks
        # Note technically one could have no viable options for reactants if masked them all out (as already selected)
        # technically then at this point the add reactant action should not be selected the step before (ie it should
        # be masked out). However, we shall assume that the reactant vocabularies we are dealing with are so large
        # that no learned model should ever select that large a number of them and so get into this situation anyway.

        mask, actions, log_probs = self._filter_and_sample_points(x_type, op, sampling_func, synthesis_trees.REACTANT_CHOOSE_STEP_VAL,
                                                       reactant_feats, cont_indcs_into_original,
                                                       indcs2logitmasks=get_reactant_mask_func)
        if actions is None:
            new_action_kind, new_x_parts, continue_mask = None, None, None
        else:
            new_action_kind = synthesis_trees.ADD_STEP_VAL
            # ^ on a reactant choose step we go back to an add step after

            new_x_parts = reactant_feats[actions, :]

            indices_of_original = cont_indcs_into_original[mask]
            inter_node_mngr.register_new_reactant_node(indices_of_original, actions)
            # ^ tell the mask manager that new reactant nodes exist in certain DAGs

            continue_mask = torch.ones_like(actions, dtype=torch.bool)

        return mask, actions, log_probs, new_action_kind, new_x_parts, continue_mask


    def _sample_edge_add_step_points(self, x_type, op, sampling_func, cont_indcs_into_original, inter_node_mngr):

        mol_features_incl_edge_stop = inter_node_mngr.molecular_features_including_edge_stops
        def get_edge_masks_func(indcs_to_get_mask_for):
            edge_masks = inter_node_mngr.get_current_edge_masks(indcs_to_get_mask_for)
            return edge_masks
        mask, actions, log_probs = self._filter_and_sample_points(x_type, op, sampling_func,
                                                       synthesis_trees.EDGE_ADD_STEP_VAL,
                                                       mol_features_incl_edge_stop,
                                                       cont_indcs_into_original,
                                                       indcs2logitmasks=get_edge_masks_func)
        if actions is None:
            new_action_kind, new_x_parts, continue_mask = None, None, None
        else:
            new_x_parts = mol_features_incl_edge_stop[actions, :]

            # For the new action kind we are either going to continue adding edges or we will stop if we selected one of
            # top actions.
            new_action_kind = torch.full_like(actions, fill_value=synthesis_trees.EDGE_ADD_STEP_VAL, dtype=settings.TORCH_INT)

            mask_where_stopping_intermediate = actions == inter_node_mngr.current_stop_intermediate_indx
            new_action_kind[mask_where_stopping_intermediate] = synthesis_trees.ADD_STEP_VAL

            mask_where_stopping_final = actions == inter_node_mngr.current_stop_final_indx
            new_action_kind[mask_where_stopping_final] = settings.PAD_VALUE
            # ^ no more actions after so can leave with pad value

            mask_for_continuing_adding_edges = torch.logical_not(mask_where_stopping_final | mask_where_stopping_intermediate)

            indcs_into_original_of_stopping_intermediate = cont_indcs_into_original[mask][mask_where_stopping_intermediate]
            indcs_into_original_of_continuing = cont_indcs_into_original[mask][mask_for_continuing_adding_edges]

            if indcs_into_original_of_continuing.shape[0]:
                inter_node_mngr.register_nonstopping_edge_node(indcs_into_original_of_continuing,
                                                               actions[mask_for_continuing_adding_edges])
                # ^ update mask so do not repeatedly add the same edges to the same intermediate node.

            if indcs_into_original_of_stopping_intermediate.shape[0]:
                # For intermediate products that are finished we need to get the new molecule formed by the reaction
                inter_node_mngr.register_stopping_edge_intermediate_node(indcs_into_original_of_stopping_intermediate,
                                                            self._run_forward_reaction_and_get_new_products)

            # Note we do not register the batch item as stopping now with inter_node_mngr as instead we do all members
            # of the batch at the end so that we only have to make one call to the reaction predictor.

            continue_mask = torch.ones_like(actions, dtype=torch.bool)
            continue_mask[mask_where_stopping_final] = False

        return mask, actions, log_probs, new_action_kind, new_x_parts, continue_mask


class _IntermediateManager(nn.Module):
    """
    Deals with the logic for dealing with an expanding number of new molecules created by reactions which represent
    intermediate and molecules in the tree. Used when sampling etc.

    * Molecule Features
    As molecules are created we need to get their features.

    * Masks
    Masks contain  block the following things happening on edge add steps:

    1. Connecting up to a node that does not yet exist in the DAG in question.
    2. Connecting up to the stop adding edges node if we have not added any edges yet. (we can't add parentless
    intermediate nodes)

    * Disconnected Nodes
    At the end we need to know all the disconnected nodes that we shall connect up into one final product node.

    * Building up the DAG.
    """
    def __init__(self,
                 initial_mol_to_idx_dict: collections.OrderedDict,
                 mol_features_reactants: torch.Tensor,
                 edge_stop_learnt_symbol: torch.Tensor,
                 batch_size: int,
                 smi2grph: typing.Callable[[str], grphs.DirectedGraphAsAdjList],
                 mol_embddr: mol_graph_embedder.GraphEmbedder
                 ):
        """
        :param initial_mol_to_idx_dict: ordered dict with the reactant vocabulary as keys and the values as the indices
        these exist in molecule features.
        :param mol_features_reactants:
        :param edge_stop_learnt_symbol:
        :param batch_size:
        :param smi2grph:
        :param mol_embddr:
        """
        super().__init__()
        self.num_initial_reactants: int = len(initial_mol_to_idx_dict)
        self.initial_reactants_set = set(initial_mol_to_idx_dict.keys())

        self.mol_to_idx_dict = initial_mol_to_idx_dict.copy()
        assert isinstance(self.mol_to_idx_dict, collections.OrderedDict)

        self.edges_added_into_undefined_intermediate_node = collections.defaultdict(list)
        # ^ this acts a temporary store of the edges going into a product node that is being built up.

        self.molecule_features = [mol_features_reactants]
        self.edge_stop_learnt_symbols = edge_stop_learnt_symbol

        self.smi2graph = smi2grph
        self.mol_embddr = mol_embddr

        self.output_trees = [nx.DiGraph() for _ in range(batch_size)]  # will store DAGs
        self.molecules_visited_in_order = collections.defaultdict(list)
        # ^ stores the order in which we visit the nodes when creating the trees.

    @property
    def molecular_features_including_edge_stops(self):
        return torch.cat(self.molecule_features + [self.edge_stop_learnt_symbols], dim=0)

    @property
    def current_stop_intermediate_indx(self):
        """
        Edge stop-intermediate is always put at the end of all available molecules -- its index therefore depends on how many
        molecules have currently been added.
        """
        return sum([e.shape[0] for e in self.molecule_features])

    @property
    def current_stop_final_indx(self):
        """
        Edge stop-final comes after edge stop-intermediate
        """
        return self.current_stop_intermediate_indx + 1

    @property
    def index_to_mol_smiles(self):
        return list(self.mol_to_idx_dict.keys())

    @property
    def _torch_device(self):
        return self.molecule_features[0].device

    def register_new_reactant_node(self, batch_elements_involved: torch.Tensor,
                                   new_element_ids_that_exist_in_dag: torch.Tensor):
        """
        This registers that a new reactant node has been added to the tree.
        """
        idx_to_mol = self.index_to_mol_smiles
        for idx, new_mol_id in zip(batch_elements_involved.cpu().numpy(),
                                   new_element_ids_that_exist_in_dag.cpu().numpy()):
            self._add_new_reactant_or_intermediate_node(idx, idx_to_mol[new_mol_id])

    def register_nonstopping_edge_node(self, batch_elements_involved: torch.Tensor, edges_used: torch.Tensor):
        """
        This registers that a new edge (to an existing node) has been added to the tree (the product that
        forms as a result of this has not been calculated yet).
        """
        batch_elements_involved = batch_elements_involved.cpu().numpy()
        edges_used = edges_used.cpu().numpy()
        for idx, e_add in zip(batch_elements_involved, edges_used):
            self.edges_added_into_undefined_intermediate_node[idx].append(e_add)

    def register_stopping_edge_intermediate_node(self, batch_elements_involved: torch.Tensor, react_predictor_func):
        """
        This registers that the stop adding edges and form intermediate product symbol has been chosen.
        This means that the intermediate node can be calculated from the reaction of those edges.
        :param react_predictor_func: takes in a list of frozen multisets of reactants and returns a list of strings of
        final products
        """
        batch_elements_involved_np = batch_elements_involved.cpu().numpy()
        idx_to_mol = self.index_to_mol_smiles

        # Calculate the reactant sets coming into this node. (These are the edges registered)
        reactant_sets = []
        for b_elem in batch_elements_involved_np:
            reactant_sets.append(multiset.FrozenMultiset([idx_to_mol[idx] for idx in
                                                          self.edges_added_into_undefined_intermediate_node[b_elem]]))

        # Calculate the intermediate product:
        new_products = react_predictor_func(reactant_sets)

        # Now add these products to the tree and connect up the edges which were until now into an undefined
        # intermediate product
        for b_elem, new_prod in zip(batch_elements_involved_np, new_products):
            new_node = (new_prod,)
            self._add_new_reactant_or_intermediate_node(b_elem, new_prod)
            for ed in self.edges_added_into_undefined_intermediate_node[b_elem]:
                incoming_node = (idx_to_mol[ed],)
                self.output_trees[b_elem].add_edge(incoming_node, new_node)
            self.edges_added_into_undefined_intermediate_node[b_elem] = []

        self._add_new_mols_to_dict_and_features(new_products)

    def get_current_edge_masks(self, batch_elements_involved: torch.Tensor):
        """
        this is based on current edges added into the working intermediate product node.
        """
        num_elements = batch_elements_involved.shape[0]
        num_possible_edge_actions = len(self.mol_to_idx_dict) + 2  #  +2 for two different stop actions.
        base_mask = torch.zeros(num_elements, num_possible_edge_actions, dtype=torch.bool, device=self._torch_device)
        base_mask[:, -1] = True  # can always stop final

        for i, batch_element in enumerate(batch_elements_involved.cpu().numpy()):
            # First turn on all molecules in the graph
            molecules_existing_smi = [node[0] for node in self.output_trees[batch_element].nodes]
            molecules_existing_indcs = [self.mol_to_idx_dict[smi] for smi in molecules_existing_smi]
            base_mask[i, molecules_existing_indcs] = True

            # Then turn back off all molecules that have already been connected to this intermediate node.
            if len(self.edges_added_into_undefined_intermediate_node[batch_element]):
                for edges_already_added in self.edges_added_into_undefined_intermediate_node[batch_element]:
                    base_mask[i, edges_already_added] = False
                base_mask[i, -2] = True  # as has at least one edge can also now stop intermediate
        return base_mask

    def get_current_reactant_masks(self,  batch_elements_involved: torch.Tensor):
        """
        Cannot add the same reactant more than once in DAG!
        """
        num_elements = batch_elements_involved.shape[0]
        base_mask = torch.ones(num_elements, self.num_initial_reactants, dtype=torch.bool, device=self._torch_device)
        for i, batch_element in enumerate(batch_elements_involved.cpu().numpy()):
            reactants_existing_smi = [node[0] for node in self.output_trees[batch_element].nodes if node[0] in self.initial_reactants_set]
            reactants_existing_indcs = [self.mol_to_idx_dict[smi] for smi in reactants_existing_smi]
            assert np.all(np.array(reactants_existing_indcs) < self.num_initial_reactants)
            base_mask[i, reactants_existing_indcs] = False
        return base_mask

    def close_off_unconnected_and_return_final_trees(self, react_predictor_func, rng):
        """
         :param react_predictor_func: takes in a list of frozen multisets of reactants and returns a list of strings of
        final products
        :param react_predictor_func:
        :return:
        """
        self._clean_construction_upto_final()

        # Work out what are the nodes with no children and connect these up to the stop node. Then work out what the
        # stop node will therefore be.
        multiset_disconnected = self._get_multisets_of_molecules_coming_into_final()
        final_products = react_predictor_func(multiset_disconnected)

        output_synthesis_trees = []
        mols_visited_in_order = [self.molecules_visited_in_order[i] for i in range(len(self.molecules_visited_in_order))]

        # Add the stop node to the trees and construction sequencing
        for tree, final_smiles, smiles_into_final, mol_order in zip(self.output_trees,
                                                                    final_products, multiset_disconnected,
                                                                     mols_visited_in_order):
            final_node = (final_smiles,)
            tree.add_node(final_node)
            for reactant_in in smiles_into_final:
                tree.add_edge((reactant_in,), final_node)
            mol_order.append(final_node)

            tree, mol_order = self._clean_tree_and_construction_order(tree, mol_order, final_node=final_node)
            syn_tree = synthesis_trees.SynthesisTree(tree, final_smiles, mol_order, rng)
            output_synthesis_trees.append(syn_tree)
        return output_synthesis_trees

    def _clean_construction_upto_final(self):
        for i in range(len(self.output_trees)):
            self.output_trees[i], self.molecules_visited_in_order[i] = \
                self._clean_tree_and_construction_order(self.output_trees[i], self.molecules_visited_in_order[i])

    @classmethod
    def _clean_tree_and_construction_order(cls, tree, order_for_construction, final_node=None):
        """
        This function exists to turn the tree/order_for_construction into the correct format (ie a DAG).
        Generally this should be the case already -- but it could be while sampling that loops are added
        to the directed graph, or that reactants/intermediate molecules
        are added several times. This function removes these (includes the first sampled route to each molecule).
        """
        tree = tree.copy()

        # First filter the order_for_construction so only contains each molecule once and nodes after the final node are
        # removed.
        order_for_construction = order_for_construction
        seen_so_far = set()
        new_order_for_construction = []
        for node in order_for_construction:
            if node not in seen_so_far:
                new_order_for_construction.append(node)
                seen_so_far.add(node)

            # If we add items after final node then remove them:
            if final_node is not None and node == final_node:
                break
        nodes_to_remove = set(tree.nodes) - seen_so_far
        if len(nodes_to_remove):
            tree.remove_nodes_from(nodes_to_remove)

        # Remove loops in the tree:
        node_to_order = dict(zip(new_order_for_construction, range(len(new_order_for_construction))))
        edge_to_remove = []
        for source, dest in tree.edges:
            if node_to_order[source] >= node_to_order[dest]:
                edge_to_remove.append((source, dest))
        tree.remove_edges_from(edge_to_remove)

        # Remove parts that do not go up to final node:
        if final_node is not None:
            nodes_to_keep = nx.ancestors(tree, final_node)
            nodes_to_keep.add(final_node)
            nodes_to_remove = set(tree.nodes) - nodes_to_keep
            tree.remove_nodes_from(nodes_to_remove)

        return tree, new_order_for_construction

    def _add_new_reactant_or_intermediate_node(self, indx, smi):
        if (smi,) not in self.output_trees[indx]:
            self.output_trees[indx].add_node((smi,))
        self.molecules_visited_in_order[indx].append((smi,))

    def _get_multisets_of_molecules_coming_into_final(self):
        out = []
        idx_to_smi = self.index_to_mol_smiles
        for i, tree in enumerate(self.output_trees):
            # The reactants in the final reaction consist of a) those that have no successor at the moment:
            non_follow_this_tree = [n for n in tree.nodes if len(list(tree.successors(n))) == 0]
            smiles_coming_in = [e[0] for e in non_follow_this_tree]  # <- remove from tuple incasing.

            # And b) those that are currently connected up to the product node:
            smiles_coming_in.extend([idx_to_smi[idx] for idx in self.edges_added_into_undefined_intermediate_node[i]])

            out.append(multiset.FrozenMultiset(smiles_coming_in))
        return out

    def _add_new_mols_to_dict_and_features(self, potential_new_molecules: typing.List[str]):
        old_length = len(self.mol_to_idx_dict)

        new_molecules_to_obtain_features_for = []
        # Note that the molecule could already exist in which case we do not add it again to the dict:
        for new_mol in potential_new_molecules:
            if new_mol not in self.mol_to_idx_dict:
                self.mol_to_idx_dict[new_mol] = len(self.mol_to_idx_dict)
                new_molecules_to_obtain_features_for.append(new_mol)

        new_length = len(self.mol_to_idx_dict)

        # We are going to increase the number of molecules in our vocabulary and also update the mask for which members
        # of the batch have elements belonging to this new vocabulary members.
        if old_length < new_length:
            assert len(new_molecules_to_obtain_features_for)
            self._add_new_feats_to_mol_feats(new_molecules_to_obtain_features_for)

    def _add_new_feats_to_mol_feats(self, list_of_smiles):
        mol_graphs: typing.List[grphs.DirectedGraphAsAdjList] = []
        for smi in list_of_smiles:
            try:
                mol_graphs.append(self.smi2graph(smi))
            except Exception as ex:
                print(f"Failed to obtain features for {smi}")
                raise RuntimeError(f"Smiles2Graph conversion failed on SMILES: '{smi}', SMILES obtained from reaction"
                                   f"predictor should be valid and able to be converted to a graph") from ex

        mol_graphs: grphs.DirectedGraphAsAdjList = mol_graphs[0].concatenate(mol_graphs)
        mol_graphs.inplace_torch_to(device=self._torch_device)
        molecule_feats = self.mol_embddr(mol_graphs)
        self.molecule_features.append(molecule_feats)


def dot_product_similarity(proposed_embeddings, all_embeddings):
    """
    :param proposed_embeddings: [b, h]
    :param all_embeddings: eg [V, h]
    """
    return proposed_embeddings @ all_embeddings.transpose(0, 1)
