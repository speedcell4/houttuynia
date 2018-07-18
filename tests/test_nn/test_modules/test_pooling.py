import unittest

import torch
from hypothesis import given, strategies as st

from houttuynia import testing
from houttuynia.nn.modules.pooling import PiecewiseMaxPool1d


class TestPiecewiseMaxPool1d(unittest.TestCase):
    @given(
        batch=testing.BATCH,
        channel=testing.CHANNEL,
        in_features=testing.NORMAL_FEATURE,
        out_features=testing.NORMAL_FEATURE,
        num_pieces=st.integers(1, 10),
    )
    def test_shape(self, batch, channel, in_features, out_features, num_pieces):
        pool = PiecewiseMaxPool1d(out_features, num_pieces)
        inputs = torch.rand(batch, channel, in_features)
        mask = (torch.rand(batch, channel) * (num_pieces + 1)).long()

        self.assertEqual((batch, channel, num_pieces, out_features), pool(inputs, mask).size())
