from pathlib import Path
from typing import List, MutableMapping

import torch
from tensorboardX import SummaryWriter

from houttuynia import config


class Monitor(object):
    chapter_name_format: str = r'{name}/{chapter}'

    def __init__(self) -> None:
        self.history: MutableMapping[str, List[float]] = {}

    def __contains__(self, chapter_name: str) -> bool:
        return chapter_name in self.history

    def clear_observation(self) -> None:
        return self.history.clear()

    def report_scalars(self, *, chapter: str = None, global_step: int = None, **values) -> None:
        raise NotImplementedError


class InMemoryMonitor(Monitor):
    def __contains__(self, chapter_name: str) -> bool:
        return chapter_name in self.history

    def report_scalars(self, *, chapter: str = None, global_step: int = None, **values) -> None:
        if chapter is None:
            chapter = config['chapter']
        for name, value in values.items():
            if torch.is_tensor(value):
                value = value.item()
            chapter_name = self.chapter_name_format.format(name=name, chapter=chapter)
            self.history.setdefault(chapter_name, []).append(value)


class TensorboardMonitor(Monitor):
    def __init__(self, dir: Path, comment: str):
        super(TensorboardMonitor, self).__init__()
        self.writer = SummaryWriter(log_dir=dir.__str__(), comment=comment)

    def report_scalars(self, *, chapter: str = None, global_step: int = None, **values) -> None:
        if chapter is None:
            chapter = config['chapter']
        tag_scalar_dict = {
            name: value.item() if torch.is_tensor(value) else value
            for name, value in values.items()
        }
        return self.writer.add_scalars(chapter, tag_scalar_dict, global_step)


if __name__ == '__main__':
    reporter = InMemoryMonitor()
    reporter.report_scalars(chapter='train', loss=1.0, accuracy=0.9)
