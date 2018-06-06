from torch import nn, optim
from torch.utils.data import DataLoader

from schedules import Moment, Monitor, Schedule
from .extensions import StartWatch, StopWatch
from .triggers import PeriodicalTrigger

__all__ = [
    'EpochalSchedule',
]


class EpochalSchedule(Schedule):
    def __init__(self, estimator: nn.Module, optimizer: optim.Optimizer, monitor: Monitor) -> None:
        super().__init__(estimator=estimator, optimizer=optimizer, monitor=monitor)

        self.register_extension(PeriodicalTrigger(Moment.BEFORE_EPOCH, epoch=1))(StartWatch('epoch'))
        self.register_extension(PeriodicalTrigger(Moment.AFTER_EPOCH, epoch=1))(StopWatch('epoch'))

    def run(self, data_loader: DataLoader, num_epochs: int):
        self.trigger_extension(Moment.BEFORE_RUN)

        for self.epoch in range(num_epochs):
            self.trigger_extension(Moment.BEFORE_EPOCH)

            for batch in data_loader:
                self.trigger_extension(Moment.BEFORE_ITERATION)

                self.optimizer.zero_grad()
                self.estimator.train()
                self.criterion = self.estimator.fit(batch)

                self.trigger_extension(Moment.BEFORE_BACKWARD)
                self.criterion.backward()
                self.trigger_extension(Moment.AFTER_BACKWARD)
                self.optimizer.step()

                self.trigger_extension(Moment.AFTER_ITERATION)
                del self.criterion

            self.trigger_extension(Moment.AFTER_EPOCH)

        self.trigger_extension(Moment.AFTER_RUN)
