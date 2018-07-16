from houttuynia.schedule import Extension, Schedule
from houttuynia import long_tensor

__all__ = [
    'CommitScalarBySum',
    'CommitScalarByMean',
    'CommitPRCurve',
]


class CommitScalarBySum(Extension):
    def __init__(self, *names: str, chapter: str):
        super(CommitScalarBySum, self).__init__()
        self.names = names
        self.chapter = chapter

    def __call__(self, schedule: 'Schedule') -> None:
        return schedule.monitor.commit_scalars(
            global_step=schedule.instance, chapter=self.chapter, **{
                name: sum(values)
                for name, values in schedule.monitor.query(self.chapter, *self.names)
            })


class CommitScalarByMean(Extension):
    def __init__(self, *names: str, chapter: str):
        super(CommitScalarByMean, self).__init__()
        self.names = names
        self.chapter = chapter

    def __call__(self, schedule: 'Schedule') -> None:
        return schedule.monitor.commit_scalars(
            global_step=schedule.instance, chapter=self.chapter, **{
                name: sum(values) / len(values)
                for name, values in schedule.monitor.query(self.chapter, *self.names)
            })


class CommitPRCurve(Extension):
    def __init__(self, name: str, chapter: str):
        super(CommitPRCurve, self).__init__()
        self.name = name
        self.chapter = chapter

    def __call__(self, schedule: 'Schedule') -> None:
        [(_, targets)] = schedule.monitor.query(self.chapter, f'{self.name}_targets')
        [(_, predictions)] = schedule.monitor.query(self.chapter, f'{self.name}_predictions')
        return schedule.monitor.commit_pr_curve(
            name=self.name, global_step=schedule.instance, chapter=self.chapter,
            targets=long_tensor(targets),
            predictions=long_tensor(predictions),
        )
