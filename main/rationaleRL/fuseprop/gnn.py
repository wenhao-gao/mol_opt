import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from fuseprop.mol_graph import MolGraph
from fuseprop.encoder import GraphEncoder
from fuseprop.decoder import GraphDecoder
from fuseprop.nnutils import *

def make_cuda(graph_tensors):
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return graph_tensors


class AtomVGNN(nn.Module):

    def __init__(self, args):
        super(AtomVGNN, self).__init__()
        self.latent_size = args.latent_size
        self.encoder = GraphEncoder(args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depth)
        self.decoder = GraphDecoder(args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.latent_size, args.diter)

        self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.G_var = nn.Linear(args.hidden_size, args.latent_size)

    def encode(self, graph_tensors):
        graph_vecs = self.encoder(graph_tensors)
        graph_vecs = [graph_vecs[st : st + le].sum(dim=0) for st,le in graph_tensors[-1]]
        return torch.stack(graph_vecs, dim=0)

    def decode(self, init_smiles):
        batch_size = len(init_smiles)
        z_graph_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode(z_graph_vecs, init_smiles)

    def rsample(self, z_vecs, W_mean, W_var, mean_only=False):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)

        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        if mean_only: 
            return z_mean, kl_loss
        else:
            epsilon = torch.randn_like(z_mean).cuda()
            z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
            return z_vecs, kl_loss

    def forward(self, graphs, tensors, init_atoms, orders, beta):
        tensors = make_cuda(tensors)
        graph_vecs = self.encode(tensors)
        z_graph_vecs, kl_div = self.rsample(graph_vecs, self.G_mean, self.G_var)
        loss, wacc, tacc, sacc = self.decoder(z_graph_vecs, graphs, tensors, init_atoms, orders)
        return loss + beta * kl_div, kl_div.item(), wacc, tacc, sacc

    def test_reconstruct(self, graphs, tensors, init_atoms, orders, init_smiles):
        tensors = make_cuda(tensors)
        graph_vecs = self.encode(tensors)
        z_graph_vecs, kl_div = self.rsample(graph_vecs, self.G_mean, self.G_var, mean_only=True)
        loss, wacc, tacc, sacc = self.decoder(z_graph_vecs, graphs, tensors, init_atoms, orders)
        return self.decoder.decode(z_graph_vecs, init_smiles)

    def likelihood(self, graphs, tensors, init_atoms, orders):
        tensors = make_cuda(tensors)
        graph_vecs = self.encode(tensors)
        z_graph_vecs, kl_div = self.rsample(graph_vecs, self.G_mean, self.G_var, mean_only=True)
        loss, wacc, tacc, sacc = self.decoder(z_graph_vecs, graphs, tensors, init_atoms, orders)
        return -loss - kl_div # Important: loss is negative log likelihood

