from typing import Tuple

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from houttuynia import float_tensor, long_tensor

__all__ = [
    'prepare_iris_dataset',
]


def prepare_iris_dataset(batch_size: int, shuffle: int = True,
                         train_size: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    data, targets = load_iris(return_X_y=True)
    data_train, data_test, targets_train, targets_test = \
        train_test_split(data, targets, train_size=train_size, shuffle=True)

    train_dataset = TensorDataset(float_tensor(data_train), long_tensor(targets_train))
    test_dataset = TensorDataset(float_tensor(data_test), long_tensor(targets_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader


if __name__ == '__main__':
    train_data, test_data = prepare_iris_dataset(4)
    for instance in train_data:
        print(instance)
