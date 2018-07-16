import unittest

from hypothesis import given, strategies as st

from houttuynia.data_loader import iris_data_loader


class TestIrisLoader(unittest.TestCase):
    @given(
        batch_size=st.integers(1, 10),
        shuffle=st.booleans(),
        num_workers=st.integers(0, 3),
    )
    def test_iris(self, batch_size: int, shuffle: bool, num_workers: int):
        train, test = iris_data_loader(batch_size, shuffle, num_workers, drop_last=True)
        for batch in train:
            self.assertEqual(batch_size, train.dataset.get_batch_size(batch))
        for batch in test:
            self.assertEqual(batch_size, test.dataset.get_batch_size(batch))
