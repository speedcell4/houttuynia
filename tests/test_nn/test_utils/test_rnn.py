import unittest

import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pad_packed_sequence

import houttuynia as ho
import testing
from nn.utils.rnn import pack_unsorted_sequences


class TestPackUnsortedSequences(unittest.TestCase):
    @given(
        vocab_size=testing.VOCAB_SIZE,
        batch=testing.BATCH,
        times=testing.TIMES,
    )
    def test_one_sequence(self, vocab_size: int, batch: int, times: int):
        sequences = testing.batched_sequences(st.integers(0, vocab_size), batch, 1, times)
        packed, = pack_unsorted_sequences([ho.long_tensor(seq) for seq in sequences])
        a = sorted([len(item) for item in sequences], reverse=True)
        _, b = pad_packed_sequence(packed, batch_first=True, padding_value=-1)

        self.assertEqual(a, b.tolist())

    @given(
        vocab_size=testing.VOCAB_SIZE,
        batch=testing.BATCH,
        times=testing.TIMES,
    )
    def test_two_sequences(self, vocab_size: int, batch: int, times: int):
        lens = testing.batched_lens(batch, 1, times)
        a = testing.sequences_by_batched_lens(st.integers(0, vocab_size), lens)
        b = testing.sequences_by_batched_lens(st.integers(vocab_size, vocab_size * 2), lens)
        a = [ho.long_tensor(item) for item in a]
        b = [ho.long_tensor(item) for item in b]
        for x, y in zip(a, b):
            self.assertEqual(x.shape[0], y.shape[0])
        (_, a), (_, b) = pack_unsorted_sequences(a, b)
        self.assertTrue(torch.equal(a, b))
