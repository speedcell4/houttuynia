from enum import IntEnum
from typing import List, Tuple

from torch.utils.data import DataLoader
from torch import nn, optim

from .monitor import Monitor

__all__ = ['Moment', 'Trigger', 'Extension', 'Schedule']


class Moment(IntEnum):
    BEFORE_RUN: int = 1
    AFTER_RUN: int = 2

    BEFORE_EPOCH: int = 3
    AFTER_EPOCH: int = 4

    BEFORE_EPISODE: int = 5
    AFTER_EPISODE: int = 6

    BEFORE_ITERATION: int = 7
    AFTER_ITERATION: int = 8

    BEFORE_BACKWARD: int = 9
    AFTER_BACKWARD: int = 10


class Trigger(object):
    def __init__(self, *moments: Moment) -> None:
        self.moments = moments

    def __call__(self, moment: Moment, schedule: 'Schedule') -> bool:
        raise NotImplementedError


class Extension(object):
    def __call__(self, schedule: 'Schedule') -> None:
        raise NotImplementedError


class Schedule(object):
    def __init__(self, estimator: nn.Module, optimizer: optim.Optimizer, monitor: Monitor) -> None:
        super(Schedule, self).__init__()

        self.monitor = monitor
        self.estimator = estimator
        self.optimizer = optimizer

        self.extensions: List[Tuple[Trigger, Extension]] = []

        self.iteration = 0

    def register_extension(self, trigger: Trigger):
        def wrapper(extension: Extension) -> Extension:
            self.extensions.append((trigger, extension))
            return extension

        return wrapper

    def trigger_extension(self, moment: Moment) -> None:
        for trigger, extension in self.extensions:
            if moment in trigger.moments and trigger(moment, schedule=self):
                extension(schedule=self)

    def run(self, data_loader: DataLoader, num_epochs: int):
        raise NotImplementedError
