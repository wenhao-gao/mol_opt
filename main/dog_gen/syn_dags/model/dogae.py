
import torch
from torch import nn

from autoencoders.dist_parameterisers import shallow_distributions
from autoencoders.dist_parameterisers import nn_paramterised_dists
from autoencoders import wasserstein
from autoencoders import variational
from autoencoders import similarity_funcs
from graph_neural_networks.sparse_pattern import ggnn_sparse
from graph_neural_networks.ggnn_general import ggnn_base

from ..utils import settings
from . import dag_embedder
from . import dog_decoder
from . import reaction_predictors
from . import molecular_graph_embedder


default_params = {
    "latent_dim": 25,
    "mol_graph_embedder_params": dict(hidden_layer_size=80, edge_names=["single", "double", "triple", "aromatic"],
                                      embedding_dim=28, num_layers=4),
    "dag_graph_embedder_gnn_params": dict(hlayer_size=28, edge_names=["reactions"], num_layers=7),
    "dag_embedder_aggr_type_s": 'FINAL_NODE',
    "decoder_params": dict(gru_insize=28, gru_hsize=28, num_layers=2, gru_dropout=0., max_steps=100),
    }


def get_model(react_pred: reaction_predictors.AbstractReactionPredictor, smi2graph_func, reactant_vocab, params=None):
    params = params if params is not None else default_params

    # Molecule and DAG embedders which are shared by both the encoder and decoder
    mol_embedder = molecular_graph_embedder.GraphEmbedder(**params['mol_graph_embedder_params'])
    dag_gnn = ggnn_sparse.GGNNSparse(ggnn_base.GGNNParams(**params['dag_graph_embedder_gnn_params']))
    dag_embdr = dag_embedder.DAGEmbedder(dag_gnn, dag_embedder.AggrType[params['dag_embedder_aggr_type_s']],
                                         default_params['latent_dim']*2)

    # Encoder
    encoder = nn_paramterised_dists.NNParamterisedDistribution(dag_embdr,
                        final_parameterised_dist=shallow_distributions.IndependentGaussianDistribution())

    # Latent prior
    latent_prior = shallow_distributions.IndependentGaussianDistribution(
        nn.Parameter(torch.zeros(1, params['latent_dim'] * 2, dtype=settings.TORCH_FLT),
                     requires_grad=False))

    # Create the kernel
    c = 2 * params['latent_dim'] * (1 ** 2)
    # ^ see section 4 of Wasserstein Auto-Encoders by Tolstikhin et al. for motivation behind this value.
    kernel = similarity_funcs.InverseMultiquadraticsKernel(c=c)

    # Decoder
    decoder_rnn_hidden_size = params['decoder_params']['gru_hsize']
    decoder_embdg_dim = mol_embedder.embedding_dim
    decoder_nets = dog_decoder.DecoderPreDefinedNetworks(
        mol_embedder,
        f_z_to_h0=nn.Linear(params['latent_dim'], decoder_rnn_hidden_size),
        f_ht_to_e_add=nn.Sequential(nn.Linear(decoder_rnn_hidden_size, 28), nn.ReLU(), nn.Linear(28, decoder_embdg_dim)),
        f_ht_to_e_reactant=nn.Sequential(nn.Linear(decoder_rnn_hidden_size, 28), nn.ReLU(), nn.Linear(28, decoder_embdg_dim)),
        f_ht_to_e_edge=nn.Sequential(nn.Linear(decoder_rnn_hidden_size, 28), nn.ReLU(), nn.Linear(28, decoder_embdg_dim))
    )
    decoder_params = dog_decoder.DecoderParams(**params['decoder_params'])
    decoder = dog_decoder.DOGGenerator(decoder_params, other_nets=decoder_nets, react_pred=react_pred,
                                       smi2graph=smi2graph_func, reactant_vocab=reactant_vocab
                                       )

    wae = wasserstein.WAEnMMD(encoder=encoder, decoder=decoder, latent_prior=latent_prior,
                              kernel=kernel)
    wae.mol_embdr = mol_embedder
    # ^ add this network on as a property of model. During training it should be used before
    # feeding data into the model as its embeddings are needed by both the encoder and decoder.

    return wae, params
