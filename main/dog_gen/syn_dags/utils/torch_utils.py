
import numpy as np

import torch
from torch.nn.utils import rnn


def remove_last_from_packed_seq(symbol_in: rnn.PackedSequence) -> rnn.PackedSequence:
    padded, lengths = rnn.pad_packed_sequence(symbol_in)
    symbol_out = rnn.pack_padded_sequence(padded, lengths - 1)
    return symbol_out


def remove_first_from_packed_seq(symbol_in: rnn.PackedSequence) -> rnn.PackedSequence:
    padded, lengths = rnn.pad_packed_sequence(symbol_in, batch_first=False)
    symbol_out = rnn.pack_padded_sequence(padded[1:, ...], lengths - 1)
    return symbol_out


def remove_first_and_last_from_packed_seq(symbol_in: rnn.PackedSequence, take_off_from_beginning=1,
                                          take_off_from_end=1) -> rnn.PackedSequence:
    padded, lengths = rnn.pad_packed_sequence(symbol_in, batch_first=False)
    symbol_out = rnn.pack_padded_sequence(padded[take_off_from_beginning:, ...], lengths - take_off_from_beginning - take_off_from_end)
    return symbol_out


def prepend_tensor_to_start_of_packed_seq(packed_seq: rnn.PackedSequence, value_to_add):
    """
    This function shifts the whole sequence down and adds value_to_add to the start.
    """

    data, batch_sizes, *others = packed_seq

    # We're gonna be a bit cheeky and construct a Packed Sequence manually at the bottom of this function -- which the
    # docs tell us not to do but have seen others do it, eg
    # https://github.com/pytorch/pytorch/issues/8921#issuecomment-400552029
    # Originally we coded this in PyTorch 1.0 and PackedSequence was a thinner wrapper on a NamedTuple
    # so continue to check that we are still using enforce_sorted=True Packed Sequences
    if len(others):
        assert others[0] is None
        assert others[1] is None

    num_in_first_batch = batch_sizes[0]
    front = torch.zeros_like(data[:num_in_first_batch])
    front[...] = value_to_add
    new_packed_seq_data = torch.cat([front, data], dim=0)
    new_length_at_beginning = batch_sizes[:1].clone()
    new_packed_seq = rnn.PackedSequence(new_packed_seq_data, torch.cat([new_length_at_beginning, packed_seq.batch_sizes]))
    return new_packed_seq


def concat_packed_seq_features(packed_seqs):
    data = []
    batch_sizes = None

    for p_seq in packed_seqs:
        d, b_sizes, *others = p_seq

        # We're gonna be a bit cheeky and construct a Packed Sequence manually at the bottom of this function -- which the
        # docs tell us not to do but have seen others do it, eg
        # https://github.com/pytorch/pytorch/issues/8921#issuecomment-400552029
        # Originally we coded this in PyTorch 1.0 and PackedSequence was a thinner wrapper on a NamedTuple
        # so continue to check that we are still using enforce_sorted=True Packed Sequences
        if len(others):
            assert others[0] is None
            assert others[1] is None

        if batch_sizes is None:
            batch_sizes = b_sizes
        else:
            if not torch.all(batch_sizes == b_sizes):
                raise RuntimeError("Trying to concat incompatible packed sequences")

        data.append(d)

    data = torch.cat(data, dim=-1)
    return rnn.PackedSequence(data, batch_sizes)


def masked_softmax_nll(logits, labels, mask):
    """
    cross entroopy loss function where the mask indicates (with True values) which of the logits should actually
    be considered.
    """
    log_probs = masked_softmax_log_probs(logits, mask)
    loss = -torch.gather(log_probs, dim=1, index=labels.view(-1, 1)).squeeze()
    return loss


def masked_softmax_log_probs(logits, mask):
    assert mask.dtype == torch.bool
    logits = logits.clone()
    logits[~mask] = -np.inf
    max_logit = torch.max(logits, dim=1, keepdim=True)[0]
    logits = logits - max_logit
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return log_probs








