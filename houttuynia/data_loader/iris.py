from typing import Tuple

from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from houttuynia import float_tensor, long_tensor

__all__ = [
    'iris_data_loader',
]


def iris_data_loader(batch_size: int, shuffle: int = True) -> Tuple[DataLoader, DataLoader]:
    data, targets = load_iris(return_X_y=True)
    dataset = TensorDataset(float_tensor(data), long_tensor(targets))

    train_dataset, test_dataset = random_split(dataset, [100, 50])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
