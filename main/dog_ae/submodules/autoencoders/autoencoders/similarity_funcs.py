
import warnings

import torch
from torch import nn


def square_distance_matrix_between_tensors(x1, x2):
    """
    :param x1: [b_1, ...]
    :param x2: [b_2, ...]
    :return: [b_1, b_2] of the squared
    """
    x1_flattened = torch.flatten(x1, start_dim=1)
    x2_flattened = torch.flatten(x2, start_dim=1)
    x1_sq = torch.sum(x1_flattened**2, dim=1)[:, None]
    x2_sq = torch.sum(x2_flattened**2, dim=1)[None, :]
    x1x2 = x1_flattened @ x2_flattened.transpose(0, 1)
    sq_dist_mat = -2 * x1x2 + x2_sq + x1_sq
    return sq_dist_mat


def estimate_mmd(similarity_func, x1, x2):
    assert x1.shape[0] == x2.shape[0]
    if x1.shape[0] == 1:
        warnings.warn("Computing MMD with only 1 example! (are you sure you meant to do this?)")
        return -2 * similarity_func.similarity_matrix(x1, x2).mean()
    else:
        x1_term = off_diagional_similarity_matrix_mean(similarity_func, x1)
        x2_term = off_diagional_similarity_matrix_mean(similarity_func, x2)
        x1_x2_terms = similarity_func.similarity_matrix(x1, x2).mean()
        return x1_term + x2_term - 2 * x1_x2_terms


def off_diagional_similarity_matrix_mean(similarity_func, x):
    num_off_diagonal_terms = x.shape[0] * (x.shape[0] - 1)
    off_diagonal_sum = 2 * similarity_func.similarity_matrix(x, x).triu(diagonal=1).sum()
    off_diagonal_mean = off_diagonal_sum / num_off_diagonal_terms
    return off_diagonal_mean


class BaseSimilarityFunctions(nn.Module):
    def forward(self, x, x2=None):
        if x2 is None:
            x2 = x
        return self.similarity_matrix(x, x2)

    def self_similarities(self, x):
        return self.pairwise_similarities(x, x)

    def similarity_matrix(self, x, x2):
        raise NotImplementedError

    def pairwise_similarities(self, x1, x2):
        return torch.diag(self.similarity_matrix(x1, x2))


class InverseMultiquadraticsKernel(BaseSimilarityFunctions):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def self_similarities(self, x):
        return torch.ones(x.shape[0], device=str(x.device))

    def similarity_matrix(self, x, x2):
        sq_dist = square_distance_matrix_between_tensors(x, x2)
        res = self.c / (self.c + sq_dist)
        return res


class SquaredEuclideanDistSimilarity(BaseSimilarityFunctions):
    def self_similarities(self, x):
        return torch.zeros(x.shape[0], device=str(x.device))

    def pairwise_similarities(self, x1, x2):
        return torch.sum((x1-x2)**2, dim=tuple(range(1, len(x1.shape))))

