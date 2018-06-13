from houttuynia.schedule import *

__all__ = [
    'Periodic',
]


class Periodic(Trigger):
    def __init__(self, moments: Moment, **intervals) -> None:
        super().__init__(moments)
        self.intervals = intervals

    def __call__(self, moment: Moment, schedule: 'Schedule') -> bool:
        for key, interval in self.intervals.items():
            # TODO warning for not hasattr
            if getattr(schedule, key) % interval != 0:
                return False
        return True
