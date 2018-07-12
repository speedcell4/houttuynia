import unittest

import torch

import houttuynia as ho
from nn.modules.position import position_indexes


class TestPositionIndexes(unittest.TestCase):
    def test_token_0(self):
        tensor1 = [1, 2, 3, 4]
        tensor2 = ho.long_tensor([0, 1, 2, 3])
        tensor1 = position_indexes(tensor1)
        self.assertTrue(torch.equal(tensor1, tensor2), msg=f'{tensor1} != {tensor2}')

    def test_token_1(self):
        tensor1 = [1, 2, 3, 4]
        tensor2 = ho.long_tensor([-1, 0, 1, 2])
        tensor1 = position_indexes(tensor1, token=2)
        self.assertTrue(torch.equal(tensor1, tensor2), msg=f'{tensor1} != {tensor2}')

    def test_token_2(self):
        tensor1 = [1, 2, 3, 4]
        tensor2 = ho.long_tensor([3, 4, 5, 6])
        tensor1 = position_indexes(tensor1, token=2, offset=4)
        self.assertTrue(torch.equal(tensor1, tensor2), msg=f'{tensor1} != {tensor2}')

    def test_token_3(self):
        tensor1 = [1, 2, 3, 4]
        tensor2 = ho.long_tensor([3, 4, 5, 5])
        tensor1 = position_indexes(tensor1, token=2, offset=4, min_value=-1, max_value=+1)
        self.assertTrue(torch.equal(tensor1, tensor2), msg=f'{tensor1} != {tensor2}')
