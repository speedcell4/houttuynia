import random

from torch import Tensor
import torch
from torch import nn
import numpy as np

from . import log_system


class Configuration(dict):
    def __getattr__(self, item):
        return super(Configuration, self).__getitem__(item)

    def __setattr__(self, key, value):
        return super(Configuration, self).__setitem__(key, value)

    def __delattr__(self, item):
        return super(Configuration, self).__delitem__(item)


config = Configuration(
    handler=log_system.push_stream_handler(),
    device=torch.device('cpu'),
    chapter='train',
)


def manual_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(device_id: str, *moduels: nn.Module) -> None:
    config['device'] = torch.device(device_id)
    for module in moduels:
        module.to(config['device'])


# tensor type definitions

def i8_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.int8)


def i16_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.int16)


def i32_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.int32)


def i64_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.int64)


def u8_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.uint8)


def u16_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.uint16)


def u32_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.uint32)


def u64_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.uint64)


def f16_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.float16)


def f32_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.float32)


def f64_tensor(tensor) -> Tensor:
    return torch.tensor(tensor, device=config['device'], dtype=torch.float64)


byte_tensor = u8_tensor
long_tensor = i64_tensor
float_tensor = f32_tensor
