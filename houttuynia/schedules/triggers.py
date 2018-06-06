from .schedule import *

__all__ = [
    'PeriodicalTrigger',
]


class PeriodicalTrigger(Trigger):
    def __init__(self, moments: Moment, **kwargs) -> None:
        super().__init__(moments)
        self.kwargs = kwargs

    def __call__(self, moment: Moment, schedule: 'Schedule') -> bool:
        for key, interval in self.kwargs.items():
            if getattr(schedule, key) % interval != 0:
                return False
        return True
