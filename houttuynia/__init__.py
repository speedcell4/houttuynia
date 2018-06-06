import random

import torch
from torch import cuda
from torch.nn import Module
import numpy as np


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


def to_device(device_id: str, *moduels: Module):
    config['device'] = torch.device(device_id)
    for module in moduels:
        module.to(config['device'])


def using_config(**kwargs):
    global config
    old_config = {**config}
    config = {**old_config, **kwargs}
    try:
        yield
    finally:
        config = old_config
