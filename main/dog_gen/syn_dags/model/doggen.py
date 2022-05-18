
import torch
from torch import nn
from ..data import synthesis_trees
from ..utils import settings
from . import reaction_predictors
from . import molecular_graph_embedder
from . import dog_decoder
from graph_neural_networks.sparse_pattern import ggnn_sparse
from graph_neural_networks.ggnn_general import ggnn_base


class DogGen(nn.Module):
    """
    Wrapper around the Dog Generator to make it more obviously into just a autoregressive model
    rather than part of an autoencoder
    """
    def __init__(self, dog_gen: dog_decoder.DOGGenerator, initial_z_size):
        super().__init__()
        self.dog_gen = dog_gen
        self.initial_z_size = initial_z_size
        self.mol_embdr = dog_gen.other_nets.mol_embdr

    def forward(self, obs: synthesis_trees.PredOutBatch):
        """Gets the negative log likelihood of observation for training (nb note this is via teacher forcing)"""
        self._update_gen(obs.batch_size)
        loss = self.dog_gen.nlog_like_of_obs(obs)
        return loss

    @torch.no_grad()
    def sample(self, batch_size):
        self._update_gen(batch_size)
        sample = self.dog_gen.sample_no_grad(1)[0]
        return sample

    def _update_gen(self, batch_size):
        device = next(self.dog_gen.parameters()).device
        # ^ get the Torch device from the model by looking at where its parameters live.
        initial_hidden = torch.zeros((batch_size, self.initial_z_size), dtype=settings.TORCH_FLT, device=device)
        # ^ for DoG-Gen the initial hidden layer is always set at the same value
        self.dog_gen.update(initial_hidden)


default_params = {
        "latent_dim": 50,
        "decoder_params": dict(gru_insize=160, gru_hsize=512, num_layers=3, gru_dropout=0.1, max_steps=100),
        "mol_graph_embedder_params": dict(hidden_layer_size=80, edge_names=["single", "double", "triple", "aromatic"],
                                          embedding_dim=160, num_layers=5),
    }


def get_dog_gen(react_pred: reaction_predictors.AbstractReactionPredictor, smi2graph_func, reactant_vocab, params=None):
    if params is None:
        params = default_params

    # Molecule Embedder
    mol_embedder = molecular_graph_embedder.GraphEmbedder(**params['mol_graph_embedder_params'])

    # DoG Generator
    decoder_rnn_hidden_size = params['decoder_params']['gru_hsize']
    decoder_embdg_dim = mol_embedder.embedding_dim
    decoder_nets = dog_decoder.DecoderPreDefinedNetworks(
        mol_embedder,
        f_z_to_h0=nn.Linear(params['latent_dim'], decoder_rnn_hidden_size),
        f_ht_to_e_add=nn.Sequential(nn.Linear(decoder_rnn_hidden_size, 28), nn.ReLU(),
                                    nn.Linear(28, decoder_embdg_dim)),
        f_ht_to_e_reactant=nn.Sequential(nn.Linear(decoder_rnn_hidden_size, 28), nn.ReLU(),
                                         nn.Linear(28, decoder_embdg_dim)),
        f_ht_to_e_edge=nn.Sequential(nn.Linear(decoder_rnn_hidden_size, 28), nn.ReLU(),
                                     nn.Linear(28, decoder_embdg_dim))
    )
    decoder_params = dog_decoder.DecoderParams(**params['decoder_params'])
    decoder = dog_decoder.DOGGenerator(decoder_params, other_nets=decoder_nets, react_pred=react_pred,
                                       smi2graph=smi2graph_func, reactant_vocab=reactant_vocab
                                       )

    # Model
    model = DogGen(decoder, params['latent_dim'])

    return model, params



