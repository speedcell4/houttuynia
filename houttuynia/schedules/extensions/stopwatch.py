from datetime import datetime

from houttuynia import logging
from ..schedule import Extension, Schedule

__all__ = [
    'StartWatch', 'StopWatch',
]

attr_format: str = r'_watch_{key}'


class StartWatch(Extension):
    def __init__(self, key: str):
        super(StartWatch, self).__init__()
        self.key = key

    def __call__(self, schedule: 'Schedule') -> None:
        attr_key = attr_format.format(key=self.key)
        value = getattr(schedule, self.key)
        tm = datetime.now()
        logging.notice(f'{self.key}-{value} start => {tm}')
        return setattr(schedule, attr_key, tm)


class StopWatch(Extension):
    def __init__(self, key: str):
        super(StopWatch, self).__init__()
        self.key = key

    def __call__(self, schedule: 'Schedule') -> None:
        attr_key = attr_format.format(key=self.key)
        value = getattr(schedule, self.key)
        tm = datetime.now() - getattr(schedule, attr_key)
        logging.notice(f'{self.key}-{value} finished, time elapsed => {tm}')
        return delattr(schedule, attr_key)
