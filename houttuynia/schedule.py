import enum
from typing import List, Tuple

from houttuynia.monitors import Monitor
from houttuynia.nn import Architecture

__all__ = ['Moment', 'Trigger', 'Extension', 'Pipeline', 'Schedule']


@enum.unique
class Moment(enum.IntEnum):
    BEFORE_RUN: int = enum.auto()
    AFTER_RUN: int = enum.auto()

    BEFORE_EPOCH: int = enum.auto()
    AFTER_EPOCH: int = enum.auto()

    BEFORE_EPISODE: int = enum.auto()
    AFTER_EPISODE: int = enum.auto()

    BEFORE_ITERATION: int = enum.auto()
    AFTER_ITERATION: int = enum.auto()

    BEFORE_BACKWARD: int = enum.auto()
    AFTER_BACKWARD: int = enum.auto()


class Trigger(object):
    def __init__(self, *moments: Moment) -> None:
        self.moments = moments

    def __call__(self, moment: Moment, schedule: 'Schedule') -> bool:
        raise NotImplementedError


class Extension(object):
    def __call__(self, schedule: 'Schedule') -> None:
        raise NotImplementedError


class Pipeline(Extension):
    def __init__(self, *extensions: Extension) -> None:
        super(Pipeline, self).__init__()
        self._extensions = extensions

    def __call__(self, schedule: 'Schedule') -> None:
        for extension in self._extensions:
            extension.__call__(schedule=schedule)


class Schedule(object):
    def __init__(self, estimator: Architecture, optimizer, monitor: Monitor) -> None:
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

    def run(self, data_loader, num_epochs: int):
        raise NotImplementedError
