import unittest

from data_loader import iris_data_loader


class TestIrisDataLoader(unittest.TestCase):
    def test_shape(self):
        train_data, test_data = iris_data_loader(1)
        self.assertEqual(train_data.__len__(), 100)
        self.assertEqual(test_data.__len__(), 50)