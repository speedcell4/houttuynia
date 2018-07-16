from typing import List, NamedTuple, Tuple

from sklearn.datasets import load_iris
import torch
from torch.utils.data import DataLoader

from houttuynia import float_tensor, long_tensor, config
from houttuynia.data_loader import Dataset

__all__ = [
    'iris_data_loader',
    'IrisDataset',
]


class IrisDataset(Dataset):
    Batch = NamedTuple('Batch', data=torch.FloatTensor, targets=torch.LongTensor)
    Instance = NamedTuple('Instance', data=torch.FloatTensor, targets=torch.LongTensor)

    def __init__(self, data: torch.FloatTensor, targets: torch.LongTensor, indexes: torch.LongTensor) -> None:
        super(IrisDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.indexes = indexes

    def __len__(self) -> int:
        return self.indexes.size(0)

    def __getitem__(self, index: int) -> Instance:
        index = self.indexes[index]
        return self.Instance(self.data[index], self.targets[index])

    def collate_fn(self, instances: List[Instance]) -> Batch:
        data, targets = zip(*instances)
        data = torch.stack(data, dim=0).to(config['device'])
        targets = torch.stack(targets, dim=0).to(config['device'])
        return self.Batch(data, targets)

    @staticmethod
    def get_batch_size(batch: Batch) -> int:
        data, targets = batch
        return data.size(0)


def iris_data_loader(batch_size: int, shuffle: int = True,
                     num_workers: int = 1, drop_last: bool = False) -> Tuple[DataLoader, DataLoader]:
    data, targets = load_iris(return_X_y=True)
    data = float_tensor(data)
    targets = long_tensor(targets)
    indexes = torch.randperm(data.size(0))

    train_dataset = IrisDataset(data, targets, indexes[:100])
    test_dataset = IrisDataset(data, targets, indexes[100:])
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        drop_last=drop_last, num_workers=num_workers)
    return train_loader, test_loader
