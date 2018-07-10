from click import Path
import torch
from torch import nn

from houttuynia.schedule import Extension, Schedule

__all__ = [
    'Snapshot',
]


class Snapshot(Extension):
    format_pattern = '{name}_epoch{epoch:02d}.pkl'

    def __init__(self, expt_dir: Path, **estimators: nn.Module) -> None:
        super(Snapshot, self).__init__()

        if expt_dir.name != self.__class__.__name__.lower():
            expt_dir /= self.__class__.__name__.lower()
        expt_dir.mkdir(parents=True, exist_ok=True)

        self.out_dir = expt_dir
        self.estimators = estimators

    def __call__(self, schedule: 'Schedule') -> None:
        for name, estimator in self.estimators.items():
            filename = self.format_pattern.format(name=name, epoch=schedule.epoch)
            torch.save(estimator, (self.out_dir / filename).__str__())
