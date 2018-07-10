import torch
from typing import List

from torch.nn.utils.rnn import PackedSequence, pack_sequence

__all__ = [
    'pack_unsorted_sequences',
]


def pack_unsorted_sequences(*sequences: List[torch.LongTensor], key=None, reverse: bool = True) -> List[PackedSequence]:
    """ return the PackedSequence by sorting the sequences with key

    Args:
        *sequences: sort of sequences
        key: the default key function is the length of each item in the first sequence
        reverse: reverse the sequences

    Returns:
        sort of PackedSequence
    """

    sequences = zip(*sequences)
    if key is None:
        key = lambda seq: seq[0].size(0)
    sequences = sorted(sequences, key=key, reverse=reverse)
    return [pack_sequence(sequence) for sequence in zip(*sequences)]
