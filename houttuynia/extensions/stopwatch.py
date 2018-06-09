from datetime import datetime

from houttuynia import log_system
from ..schedule import Extension, Schedule

__all__ = [
    'StartWatch', 'StopWatch',
]

attr_format: str = r'_watch_{key}'


# TODO timezone
class StartWatch(Extension):
    def __init__(self, key: str):
        super(StartWatch, self).__init__()
        self.key = key
        self.attr = attr_format.format(key=key)

    def __call__(self, schedule: 'Schedule') -> None:
        value = getattr(schedule, self.key)
        tm = datetime.now()
        log_system.notice(f'[{self.key} {value}] start => {tm}')
        return setattr(schedule, self.attr, tm)


class StopWatch(Extension):
    def __init__(self, key: str):
        super(StopWatch, self).__init__()
        self.key = key
        self.attr = attr_format.format(key=key)

    def __call__(self, schedule: 'Schedule') -> None:
        value = getattr(schedule, self.key)
        tm = datetime.now() - getattr(schedule, self.attr)
        log_system.notice(f'[{self.key} {value}] finished, time elapsed => {tm}')
        return delattr(schedule, self.attr)
