import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import datasets

from houttuynia import float_tensor, long_tensor

__all__ = [
    'prepare_iris_dataset',
]


def prepare_iris_dataset(batch_size: int, shuffle: int = True, ratio: float = 0.8):
    data, target = datasets.load_iris(return_X_y=True)
    indexes = torch.torch.randperm(target.shape[0])
    pivot = int(indexes.shape[0] * ratio)
    train_idx, test_idx = indexes[:pivot], indexes[pivot:]

    data = float_tensor(data)
    target = long_tensor(target)

    train_dataset = TensorDataset(data[train_idx], target[train_idx])
    test_dataset = TensorDataset(data[test_idx], target[test_idx])
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), \
           DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
