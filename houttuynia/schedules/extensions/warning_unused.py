import warnings

from houttuynia.schedules import Extension, Schedule

__all__ = [
    'WarningUnused',
]


class WarningUnused(Extension):
    def __call__(self, schedule: 'Schedule') -> None:
        if schedule.monitor.memory.__len__() != 0:
            warnings.warn(f'there are unused information in Monitor memory => {schedule.monitor.memory}')
