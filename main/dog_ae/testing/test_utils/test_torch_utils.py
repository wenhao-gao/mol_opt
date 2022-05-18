
import torch
import numpy as np
from torch.nn.utils import rnn
from torch import nn

from syn_dags.utils import torch_utils


def test_remove_last_from_packed_seq():
    padded_seq = torch.tensor([[1,2,3],[4,3,0], [12,18, 0]])
    orig_lengths = torch.tensor([3,2,2])

    packed_seq = rnn.pack_padded_sequence(padded_seq, orig_lengths, batch_first=True)

    computed = torch_utils.remove_last_from_packed_seq(packed_seq)
    computed_padded_seq, lengths = rnn.pad_packed_sequence(computed, batch_first=True)

    expected_computed_padded_seq = np.array([[1,2],[4,0], [12, 0]])
    expected_lengths = orig_lengths - 1

    np.testing.assert_array_equal(expected_computed_padded_seq, computed_padded_seq.numpy())
    np.testing.assert_array_equal(lengths, expected_lengths.numpy())



def test_prepend_tensor_to_start_of_packed_seq():
    padded_seq = torch.tensor([
        [[1,2], [3,10], [4,7]],
        [[8, 9], [10, 6], [11, 18]],
        [[5,12], [17, 15], [0, 0]]
    ])
    orig_lengths = torch.tensor([3,3,2])

    packed_seq = rnn.pack_padded_sequence(padded_seq, orig_lengths, batch_first=True)

    computed = torch_utils.prepend_tensor_to_start_of_packed_seq(packed_seq, 3)
    computed_padded_seq, lengths = rnn.pad_packed_sequence(computed, batch_first=True)


    expected_computed_padded_seq = np.array(
        [
            [[3, 3], [1, 2], [3, 10], [4, 7]],
            [[3, 3], [8, 9], [10, 6], [11, 18]],
            [[3, 3], [5, 12], [17, 15], [0, 0]]
        ])

    expected_lengths = np.array([4,4,3])

    np.testing.assert_array_equal(expected_computed_padded_seq, computed_padded_seq.numpy())
    np.testing.assert_array_equal(expected_lengths, lengths.numpy())


def test_masked_softmax_nll():

    rng = np.random.RandomState(9823174)
    n_data = 22
    data_dim = 45
    logits_np = 4 * rng.randn(n_data, data_dim)
    mask_np = rng.randn(n_data, data_dim) > 0.
    while not np.all(np.sum(mask_np == 1, axis=1)):
        mask_np = rng.randn(n_data, data_dim) > 0.

    labels_np = (logits_np + 2*rng.randn(n_data, data_dim))
    labels_np[~mask_np] = -np.inf
    labels_np = np.argmax(labels_np, axis=1)

    logits = torch.tensor(logits_np)
    mask = torch.tensor(mask_np)
    labels = torch.tensor(labels_np, dtype=torch.int64)

    # Via function we are testing
    nll_from_func = torch_utils.masked_softmax_nll(logits, labels, mask).numpy()

    # Via Alternative method
    softmax = nn.Softmax(dim=1)
    probs = softmax(logits)

    prob_adjustment_factor = torch.log(torch.sum(probs * mask.float(), dim=1))

    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_pers = criterion(logits, labels)
    loss_pers = loss_pers + prob_adjustment_factor
    nll_from_alternative = loss_pers.numpy()

    np.testing.assert_array_almost_equal(nll_from_func, nll_from_alternative)
