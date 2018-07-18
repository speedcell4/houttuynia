import unittest

import torch
from hypothesis import given, strategies as st

from houttuynia import testing
from houttuynia.nn import GramConv1d


class TestGramConv1d(unittest.TestCase):
    @given(
        batch=testing.SMALL_BATCH,
        times=testing.TIMES,
        in_features=testing.NORMAL_FEATURE,
        out_features=testing.NORMAL_FEATURE,
        hidden_features=testing.NORMAL_FEATURE,
        bias=st.booleans(),
        negative_slope=st.floats(0.0, 1.0),
        num_grams=st.sampled_from([1, 3, 5, 7, 9]),
    )
    def test_shape(self, batch, times, in_features, out_features, hidden_features, num_grams, bias, negative_slope):
        conv = GramConv1d(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                          bias=bias, negative_slope=negative_slope, num_grams=num_grams)
        inputs = torch.rand(batch, in_features, times)
        outputs = conv(inputs)

        self.assertEqual(outputs.size(), (batch, out_features, times))
