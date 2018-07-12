from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Type
from functools import wraps
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import numpy as np
import torch
from tensorboardX import SummaryWriter

from houttuynia import config

__all__ = [
    'Monitor',
    'FilesystemMonitor', 'TensorboardMonitor',
    'get_monitor',
]


def unwrap_chapter(func):
    @wraps(func)
    def wrapper(*args, chapter: str = None, **kwargs):
        if chapter is None:
            chapter = config['chapter']
        return func(*args, chapter=chapter, **kwargs)

    return wrapper


class Monitor(object):
    def __init__(self) -> None:
        self.memory: Dict[Tuple[str, str], List[float]] = {}

    def query(self, chapter: str, *names: str, remove: bool = True) -> Iterable[Tuple[str, List[float]]]:
        for name in names:
            key = (name, chapter)
            if key in self.memory:
                yield name, self.memory[key]
                if remove:
                    del self.memory[key]

    @unwrap_chapter
    def report_scalars(self, chapter: str = None, **values: float) -> None:
        """ save `values` to memory.

        Args:
            chapter: chapter name
            **values: the name and value dict of criterion or metric
        """
        for name, value in values.items():
            if isinstance(value, (list, tuple)):
                self.memory.setdefault((name, chapter), []).extend(value)
            else:
                self.memory.setdefault((name, chapter), []).append(value)

    @unwrap_chapter
    def commit_scalars(self, iteration: int, chapter: str = None, **values: float) -> None:
        """ save `values` to disk with (iteration, chapter) information.

        Args:
            iteration: global iteration (step)
            chapter: chapter name
            **values: the name and value dict of criterion or metric
        """
        raise NotImplementedError

    @unwrap_chapter
    def commit_kwargs(self, iteration: int, chapter: str = None, **kwargs: Any) -> None:
        """ save the argument options to disk

        Args:
            iteration: global iteration (step)
            chapter: chapter name
            **kwargs: args dictionary
        """
        raise NotImplementedError

    @unwrap_chapter
    def commit_pr_curve(self, name: str, iteration: int, predictions: torch.Tensor, targets: torch.Tensor,
                        num_thresholds: int = 127, weights: torch.Tensor = None, chapter: str = None) -> None:
        """ save the predictions and targets to disk and plot the PR-curve

        Args:
            name: the meaning
            iteration: global iteration (step)
            predictions: the predictions
            targets: the target labels
            num_thresholds: how many thresholds used to draw this curve
            weights: the weights
            chapter: chapter name
        """
        raise NotImplementedError

    @unwrap_chapter
    def commit_embedding(self, iteration: int, embedding: torch.Tensor, name: str, label_text: List[str] = None,
                         label_image: torch.Tensor = None, chapter: str = None) -> None:
        """ save the embedding to disk and plot it

        Args:
            iteration: global iteration (step)
            embedding: the embedding tensor
            name: the name of embedding
            label_text: the label list (optional)
            label_image: the label image (optional)
            chapter: chapter name
        """
        raise NotImplementedError

    @unwrap_chapter
    def commit_histogram(self, iteration: int, values: np.ndarray, chapter: str = None) -> None:
        raise NotImplementedError

    @unwrap_chapter
    def commit_image(self):
        raise NotImplementedError

    @unwrap_chapter
    def commit_audio(self):
        raise NotImplementedError

    @unwrap_chapter
    def commit_video(self):
        raise NotImplementedError


class FilesystemMonitor(Monitor):
    def __init__(self, expt_dir: Path, name: str = 'log.txt', encoding: str = 'utf-8') -> None:
        super(FilesystemMonitor, self).__init__()
        self.expt_dir = expt_dir
        self.log_file = expt_dir / name
        self.stream = self.log_file.open(mode='w', encoding=encoding)

    def __del__(self):
        self.stream.close()

    _iteration_format = '[iteration {iteration}] {name}/{chapter}\t{value:.06f}'

    @unwrap_chapter
    def commit_scalars(self, iteration: int, chapter: str = None, **values: float) -> None:
        for name, value in values.items():
            info = self._iteration_format.format(
                iteration=iteration, name=name, chapter=chapter, value=value)
            print(info, file=self.stream)

    # @unwrap_chapter
    # def commit_pr_curve(self, name: str, iteration: int, predictions: torch.Tensor, targets: torch.Tensor,
    #                     num_thresholds: int = 127, weights: torch.Tensor = None, chapter: str = None) -> None:
    #     targets = targets.numpy()
    #     predictions = predictions.numpy()
    #     precision, recall, _ = precision_recall_curve(targets, predictions)
    #     average_precision = average_precision_score(targets, predictions)
    #     plt.step(recall, precision, color='b', alpha=0.2,
    #              where='post')
    #     plt.fill_between(recall, precision, step='post', alpha=0.2,
    #                      color='b')
    #
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.ylim([0.0, 1.05])
    #     plt.xlim([0.0, 1.0])
    #     plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
    #         average_precision))
    #
    #     return plt.imsave(self.log_file.with_name(f'{name}-prcurve-{iteration}.eps').__str__())


class TensorboardMonitor(Monitor):
    def __init__(self, log_dir: Path, comment: str = '') -> None:
        super(TensorboardMonitor, self).__init__()
        self.summary_writer = SummaryWriter(log_dir=log_dir.__str__(), comment=comment)

    def __del__(self):
        self.summary_writer.close()

    @unwrap_chapter
    def commit_scalars(self, iteration: int, chapter: str = None, **values: float) -> None:
        for name, value in values.items():
            self.summary_writer.add_scalars(
                main_tag=name, tag_scalar_dict={chapter: value}, global_step=iteration)

    # @unwrap_chapter
    # def commit_pr_curve(self, name: str, iteration: int, predictions: torch.Tensor, targets: torch.Tensor,
    #                     num_thresholds: int = 127, weights: torch.Tensor = None, chapter: str = None) -> None:
    #     return self.summary_writer.add_pr_curve(
    #         labels=targets, predictions=predictions, tag=name,
    #         global_step=iteration, num_thresholds=num_thresholds,
    #     )


def get_monitor(name: str) -> Type[Monitor]:
    return {
        'filesystem': FilesystemMonitor,
        'tensorboard': TensorboardMonitor,
    }.get(name.strip().lower(), FilesystemMonitor)
