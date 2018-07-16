from typing import List, NamedTuple

from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    Batch = NamedTuple
    Instance = NamedTuple

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Instance:
        raise NotImplementedError

    def collate_fn(self, instances: List[Instance]) -> Batch:
        raise NotImplementedError

    def get_batch_size(self, batch: Batch) -> int:
        raise NotImplementedError


from houttuynia.data_loader.iris import iris_data_loader, IrisDataset
