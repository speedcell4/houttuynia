from datetime import datetime

from houttuynia import logging
from ..schedule import Extension, Schedule

__all__ = [
    'StartWatch', 'StopWatch',
]

attr_string: str = r'_watch_{key}'


class StartWatch(Extension):
    def __init__(self, key: str):
        super(StartWatch, self).__init__()
        self.key = key

    def __call__(self, schedule: 'Schedule') -> None:
        current = datetime.now()
        logging.notice(f'{self.key}-{getattr(schedule, self.key)} start => {current}')
        return setattr(schedule, attr_string.format(key=self.key), current)


class StopWatch(Extension):
    def __init__(self, key: str):
        super(StopWatch, self).__init__()
        self.key = key

    def __call__(self, schedule: 'Schedule') -> None:
        timedelta = getattr(schedule, attr_string.format(key=self.key)) - datetime.now()
        logging.notice(f'{self.key}-{getattr(schedule, self.key)} finished, time elapsed => {timedelta}')
        return delattr(schedule, attr_string.format(key=self.key))
