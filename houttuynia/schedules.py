from torch import optim
from torch.utils.data import DataLoader

from houttuynia.nn import Architecture
from houttuynia.monitors import Monitor
from houttuynia.schedule import Moment, Schedule
from houttuynia.extensions import StartWatch, StopWatch, WarningUnused
from houttuynia.triggers import Periodic

__all__ = [
    'EpochalSchedule',
]


class EpochalSchedule(Schedule):
    def __init__(self, estimator: Architecture, optimizer: optim.Optimizer, monitor: Monitor) -> None:
        super().__init__(estimator=estimator, optimizer=optimizer, monitor=monitor)

        self.epoch = 0

        self.register_extension(Periodic(Moment.BEFORE_EPOCH, epoch=1))(StartWatch('epoch'))
        self.register_extension(Periodic(Moment.AFTER_EPOCH, epoch=1))(StopWatch('epoch'))

        self.register_extension(Periodic(Moment.AFTER_RUN))(WarningUnused())

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
                self.criterion, metrics = self.estimator.fit(*batch)
                self.monitor.report_scalars(**metrics)

                self.trigger_extension(Moment.BEFORE_BACKWARD)
                self.criterion.backward()
                self.trigger_extension(Moment.AFTER_BACKWARD)
                self.optimizer.step()

                self.trigger_extension(Moment.AFTER_ITERATION)
                del self.criterion

            self.trigger_extension(Moment.AFTER_EPOCH)

        self.trigger_extension(Moment.AFTER_RUN)


