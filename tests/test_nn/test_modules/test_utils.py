from unittest import TestCase

import torch

from houttuynia import byte_tensor, long_tensor
from houttuynia.nn.modules.utils import convert_lengths_to_mask


class TestConvertLengthsToMask(TestCase):
    def test_convert_lengths_to_mask(self):
        lengths = long_tensor([5, 4, 2])
        expected_mask = byte_tensor([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 0],
                                     [1, 1, 0],
                                     [1, 0, 0]])
        self.assertTrue(torch.equal(convert_lengths_to_mask(lengths), expected_mask))
