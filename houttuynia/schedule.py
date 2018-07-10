import enum
from typing import List, Tuple

from torch import optim
from torch.utils.data import DataLoader

from .monitors import Monitor
from .nn import Architecture

__all__ = ['Moment', 'Trigger', 'Extension', 'Schedule', 'EpochalSchedule']


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


class Periodic(Trigger):
    def __init__(self, moments: Moment, **intervals) -> None:
        super().__init__(moments)
        self.intervals = intervals

    def __call__(self, moment: Moment, schedule: 'Schedule') -> bool:
        for key, interval in self.intervals.items():
            # TODO warning for not hasattr
            if getattr(schedule, key) % interval != 0:
                return False
        return True


class Extension(object):
    def __call__(self, schedule: 'Schedule') -> None:
        raise NotImplementedError


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

    def before_run(self):
        return self.register_extension(Periodic(Moment.BEFORE_RUN))

    def after_run(self):
        return self.register_extension(Periodic(Moment.AFTER_RUN))

    def before_epoch(self, *, epoch, **intervals):
        return self.register_extension(Periodic(Moment.BEFORE_EPOCH, epoch=epoch, **intervals))

    def after_epoch(self, *, epoch, **intervals):
        return self.register_extension(Periodic(Moment.AFTER_EPOCH, epoch=epoch, **intervals))

    def before_iteration(self, *, iteration, **intervals):
        return self.register_extension(Periodic(Moment.BEFORE_ITERATION, iteration=iteration, **intervals))

    def after_iteration(self, *, iteration, **intervals):
        return self.register_extension(Periodic(Moment.AFTER_ITERATION, iteration=iteration, **intervals))

    def before_backward(self, *, iteration, **intervals):
        return self.register_extension(Periodic(Moment.BEFORE_BACKWARD, iteration=iteration, **intervals))

    def after_backward(self, *, iteration, **intervals):
        return self.register_extension(Periodic(Moment.AFTER_BACKWARD, iteration=iteration, **intervals))

    def trigger_extension(self, moment: Moment) -> None:
        for trigger, extension in self.extensions:
            if moment in trigger.moments and trigger(moment, schedule=self):
                extension(schedule=self)

    def run(self, data_loader, num_epochs: int):
        raise NotImplementedError


from .extensions import StartWatch, StopWatch, WarningUnused


class EpochalSchedule(Schedule):
    def __init__(self, estimator: Architecture, optimizer: optim.Optimizer, monitor: Monitor) -> None:
        super().__init__(estimator=estimator, optimizer=optimizer, monitor=monitor)

        self.epoch = 0

        self.before_epoch(epoch=1)(StartWatch('epoch'))
        self.after_epoch(epoch=1)(StopWatch('epoch'))

        self.after_run()(WarningUnused())

    def run(self, data_loader: DataLoader, num_epochs: int):
        self.trigger_extension(Moment.BEFORE_RUN)

        for _ in range(num_epochs):
            self.epoch += 1
            self.trigger_extension(Moment.BEFORE_EPOCH)

            for batch in data_loader:
                self.iteration += 1
                self.trigger_extension(Moment.BEFORE_ITERATION)

                self.estimator.train()
                self.optimizer.zero_grad()
                self.criterion, self.metrics = self.estimator.fit(batch)
                self.monitor.report_scalars(**self.metrics)

                self.trigger_extension(Moment.BEFORE_BACKWARD)
                self.criterion.backward()
                self.trigger_extension(Moment.AFTER_BACKWARD)
                self.optimizer.step()

                self.trigger_extension(Moment.AFTER_ITERATION)
                del self.criterion
                del self.metrics

            self.trigger_extension(Moment.AFTER_EPOCH)

        self.trigger_extension(Moment.AFTER_RUN)
