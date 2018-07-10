from torch.utils.data import DataLoader
import torch

from houttuynia.schedule import Extension, Schedule
from houttuynia.context_managers import using_config

__all__ = [
    'Evaluation',
]


class Evaluation(Extension):
    def __init__(self, data_loader: DataLoader, chapter: str) -> None:
        super().__init__()
        self.chapter = chapter
        self.data_loader = data_loader

    def __call__(self, schedule: 'Schedule') -> None:
        with using_config(chapter=self.chapter) as _, torch.no_grad() as _:
            schedule.estimator.eval()

            for batch in self.data_loader:
                metrics = schedule.estimator.evaluate(batch)
                schedule.monitor.report_scalars(**metrics)
