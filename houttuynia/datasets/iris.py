import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import datasets

__all__ = [
    'prepare_iris_dataset',
]


def prepare_iris_dataset(batch_size: int, shuffle: int = True, ratio: float = 0.8):
    data, target = datasets.load_iris(return_X_y=True)
    indexes = torch.torch.randperm(target.shape[0])
    pivot = int(indexes.shape[0] * ratio)
    train_indexes, test_indexes = indexes[:pivot], indexes[pivot:]

    data = torch.tensor(data, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.long)

    _train_dataset = TensorDataset(data[train_indexes], target[train_indexes])
    _test_dataset = TensorDataset(data[test_indexes], target[test_indexes])
    return DataLoader(_train_dataset, batch_size=batch_size, shuffle=shuffle), \
           DataLoader(_test_dataset, batch_size=_test_dataset.__len__(), shuffle=False)
