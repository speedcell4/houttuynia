from .schedule import *

__all__ = [
    'Periodic',
]


class Periodic(Trigger):
    def __init__(self, moments: Moment, **intervals) -> None:
        super().__init__(moments)
        assert intervals.__len__() > 0, 'you should set at least one interval value'

        self.intervals = intervals

    def __call__(self, moment: Moment, schedule: 'Schedule') -> bool:
        for key, interval in self.intervals.items():
            # TODO warning for not hasattr
            if getattr(schedule, key) % interval != 0:
                return False
        return True


class Once(Trigger):
    def __init__(self, moment: Moment, **intervals) -> None:
        super(Once, self).__init__(moment)
