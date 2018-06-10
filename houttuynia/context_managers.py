from typing import Any

from torch import nn

from houttuynia import Configuration, config

__all__ = [
    'using_config',
    'switching_mode',
]


class using_config(object):
    def __init__(self, config: Configuration = config, **kwargs: Any) -> None:
        self.config = config
        self.new_kwargs = kwargs

    def __enter__(self):
        self.old_kwargs = {}
        for key, value in self.new_kwargs.items():
            self.old_kwargs[key] = self.config[key]
            self.config[key] = value

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.old_kwargs.items():
            self.config[key] = value
        del self.old_kwargs


class switching_mode(object):
    def __init__(self, module: nn.Module, mode: bool) -> None:
        self.mode = mode
        self.module = module

    def __enter__(self):
        self.training = self.module.training
        self.module.train(self.mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.module.train(self.training)
