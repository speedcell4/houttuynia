from datetime import datetime

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, optim
import logbook as log

_observation = {}


def report_scalar(name: str, value: float):
    global _observation
    return _observation.setdefault(name, []).append(value)


def average_scalar(name: str) -> float:
    obs = _observation.get(name, [])
    return sum(obs) / max(1, len(obs))


class Extension(object):
    def __call__(self, schedule: 'Schedule') -> None:
        raise NotImplementedError


class Schedule(object):
    accumulated_loss: torch.Tensor

    def __init__(self, module: nn.Module, optimizer: optim.Optimizer):
        super(Schedule, self).__init__()

        self.module = module
        self.optimizer = optimizer

        self.before_epoch_extensions = []
        self.after_epoch_extensions = []
        self.before_iteration_extensions = []
        self.after_iteration_extensions = []

        self.reset_parameters()

    def reset_parameters(self):
        self.epoch = 0
        self.iteration = 0

    def register_extension(self, extension: 'Extension',
                           epoch: int = None, iteration: int = None, before: bool = False) -> None:
        raise NotImplementedError

    def before_epoch(self):
        self.epoch += 1
        self.epoch_start_tm = datetime.now()
        log.notice(f'[{self.epoch} epoch] begin, current time: {self.epoch_start_tm}')

        for extension in self.before_epoch_extensions:
            extension.__call__(self)

    def after_epoch(self):
        for extension in self.after_epoch_extensions:
            extension.__call__(self)
        log.notice(f'[{self.epoch} epoch] done, elapse time: {datetime.now() - self.epoch_start_tm}')

    def before_iteration(self):
        self.iteration += 1

        for extension in self.before_iteration_extensions:
            extension.__call__(self)

    def after_iteration(self):
        loss = self.accumulated_loss.item()
        log.info(f'[{self.epoch}/{self.iteration}] loss => {loss:.4f}')

        for extension in self.after_iteration_extensions:
            extension.__call__(self)

        return self.optimizer.step()

    def train(self, data_loader: DataLoader, num_epochs: int):
        for _ in tqdm(range(num_epochs), desc='train', unit='epoch'):
            self.before_epoch()
            for batch in tqdm(data_loader, desc='batch', unit='batch'):
                self.before_iteration()
                self.module.train()
                self.accumulated_loss = self.module(batch)
                self.after_iteration()
            self.after_epoch()

    def eval(self, data_loader: DataLoader):
        raise NotImplementedError
