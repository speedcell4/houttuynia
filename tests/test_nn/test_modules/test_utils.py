import unittest
from unittest import TestCase

import torch
from hypothesis import given

from houttuynia import testing
from houttuynia import byte_tensor, long_tensor
from houttuynia.nn.modules.utils import lens_to_mask, cartesian_view
from houttuynia.nn import expanded_masked_fill


class TestConvertLengthsToMask(TestCase):
    def test_convert_lengths_to_mask(self):
        lengths = long_tensor([5, 4, 2])
        expected_mask = byte_tensor([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 0],
                                     [1, 1, 0],
                                     [1, 0, 0]])
        self.assertTrue(torch.equal(lens_to_mask(lengths), expected_mask))


class TestCartesianView(unittest.TestCase):
    def test_cartesian_view_0(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 0)
        self.assertEqual(c.size(), (2, 3, 4, 7))
        self.assertEqual(d.size(), (2, 3, 5, 9))

    def test_cartesian_view_1(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 1)
        self.assertEqual(c.size(), (2, 3, 4, 7))
        self.assertEqual(d.size(), (2, 3, 5, 9))

    def test_cartesian_view_2(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 2)
        self.assertEqual(c.size(), (2, 3, 4, 7))
        self.assertEqual(d.size(), (2, 3, 5, 9))

    def test_cartesian_view_3(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 3)
        self.assertEqual(c.size(), (2, 3, 4, 5, 7))
        self.assertEqual(d.size(), (2, 3, 4, 5, 9))

    def test_cartesian_view_4(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 4)
        self.assertEqual(c.size(), (2, 3, 4, 5, 7, 9))
        self.assertEqual(d.size(), (2, 3, 4, 5, 7, 9))

    def test_cartesian_view_5(self):
        a = torch.rand(2, 3, 4, 7)
        b = torch.rand(2, 3, 5, 9)
        c, d = cartesian_view(a, b, 0, 5)
        self.assertEqual(c.size(), (2, 3, 4, 5, 7, 9))
        self.assertEqual(d.size(), (2, 3, 4, 5, 7, 9))


class TestExpandedMaskedFill(unittest.TestCase):
    @given(
        batches=testing.SMALL_BATCHES,
        nom1=testing.SMALL_BATCHES,
        nom2=testing.NORMAL_FEATURE,
    )
    def test_shape(self, batches, nom1, nom2):
        tensor = torch.rand(*batches, *nom1, nom2).float()
        mask = (torch.rand(*batches, nom2) * 2).byte()

        masked_tensor = expanded_masked_fill(tensor, ~mask)
        self.assertEqual(masked_tensor.size(), tensor.size())
