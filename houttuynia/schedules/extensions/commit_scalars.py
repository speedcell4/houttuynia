from houttuynia.schedules import Extension, Schedule

__all__ = [
    'CommitScalarBySum',
    'CommitScalarByMean',
]


class CommitScalarBySum(Extension):
    def __init__(self, *names: str, chapter: str):
        super(CommitScalarBySum, self).__init__()
        self.names = names
        self.chapter = chapter

    def __call__(self, schedule: 'Schedule') -> None:
        return schedule.monitor.commit_scalars(
            global_step=schedule.iteration, chapter=self.chapter, **{
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
            global_step=schedule.iteration, chapter=self.chapter, **{
                name: sum(values) / len(values)
                for name, values in schedule.monitor.query(self.chapter, *self.names)
            })


