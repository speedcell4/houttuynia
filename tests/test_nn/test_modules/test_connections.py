import unittest

import torch
from hypothesis import given
from torch import nn

from houttuynia import testing
from houttuynia.nn import DPN, DenseNet, MixNet, ResNet


class TestResNet(unittest.TestCase):
    @given(
        batches=testing.BATCHES,
        in_features=testing.SMALL_FEATURE,
    )
    def test_shape(self, batches, in_features):
        connection = ResNet()
        inner = nn.Linear(in_features, in_features)

        inputs = torch.rand(*batches, in_features)
        outputs = connection(inputs, inner=inner(inputs))
        self.assertEqual(outputs.size(), (*batches, in_features))


class TestDenseNet(unittest.TestCase):
    @given(
        batches=testing.BATCHES,
        in_features=testing.SMALL_FEATURE,
        outer_features=testing.TINY_FEATURE,
    )
    def test_shape(self, batches, in_features, outer_features):
        connection = DenseNet()
        outer = nn.Linear(in_features, outer_features)

        inputs = torch.rand(*batches, in_features)
        outputs = connection(inputs, outer=outer(inputs))
        self.assertEqual(outputs.size(), (*batches, in_features + outer_features))


class TestDPN(unittest.TestCase):
    @given(
        batches=testing.BATCHES,
        in_features=testing.SMALL_FEATURE,
        inner_features=testing.TINY_FEATURE,
        outer_features=testing.TINY_FEATURE,
    )
    def test_shape(self, batches, in_features, inner_features, outer_features):
        connection = DPN()
        inner = nn.Linear(in_features, inner_features)
        outer = nn.Linear(in_features, outer_features)

        inputs = torch.rand(*batches, in_features)
        outputs = connection(inputs, inner=inner(inputs), outer=outer(inputs))
        self.assertEqual(outputs.size(), (*batches, in_features + outer_features))


class TestMixNet(unittest.TestCase):
    @given(
        batches=testing.BATCHES,
        in_features=testing.SMALL_FEATURE,
        inner_features=testing.TINY_FEATURE,
        outer_features=testing.TINY_FEATURE,
    )
    def test_shape(self, batches, in_features, inner_features, outer_features):
        connection = MixNet()
        inner = nn.Linear(in_features, inner_features)
        outer = nn.Linear(in_features, outer_features)

        inputs = torch.rand(*batches, in_features)
        outputs = connection(inputs, inner=inner(inputs), outer=outer(inputs))
        self.assertEqual(outputs.size(), (*batches, in_features + outer_features))
