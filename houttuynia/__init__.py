import random

import torch
from torch import cuda
import numpy as np

from . import datasets
from . import models
from . import nn


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)


class Configuration(dict):
    def __getattr__(self, item):
        return super(Configuration, self).__getitem__(item)

    def __setattr__(self, key, value):
        return super(Configuration, self).__setitem__(key, value)

    def __delattr__(self, item):
        return super(Configuration, self).__delitem__(item)


config = Configuration(
    chapter='train',
    device=torch.device('cpu'),
)


def i8_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.int8)


def i16_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.int16)


def i32_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.int)


def i64_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.long)


def u8_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.uint8)


def u16_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.uint16)


def u32_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.uint)


def u64_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.ulong)


def f16_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.float16)


def f32_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.float32)


def f64_tensor(tensor):
    return torch.tensor(tensor, device=config.device, dtype=torch.float64)


def using_config(**kwargs):
    global config
    old_config = {**config}
    config = {**old_config, **kwargs}
    try:
        yield
    finally:
        config = old_config
