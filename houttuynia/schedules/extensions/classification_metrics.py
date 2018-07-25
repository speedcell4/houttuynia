from houttuynia.schedules import Extension, Schedule

import numpy as np

from sklearn.metrics import precision_recall_fscore_support

__all__ = [
    'ClassificationMetrics',
]


class ClassificationMetrics(Extension):
    def __init__(self, name: str, chapter: str, average: str = 'micro', dump_data: bool = False):
        super(ClassificationMetrics, self).__init__()

        self.name = name
        self.chapter = chapter
        self.average = average
        self.dump_data = dump_data

    def __call__(self, schedule: 'Schedule') -> None:
        targets = schedule.monitor.get(chapter=self.chapter, name=f'{self.name}_targets')
        outputs = schedule.monitor.get(chapter=self.chapter, name=f'{self.name}_outputs')
        targets = np.array(targets, dtype=np.int)
        outputs = np.array(outputs, dtype=np.int)

        if self.dump_data:
            self.dump_arrays(schedule=schedule, **{
                f'{self.name}_targets': targets,
                f'{self.name}_outputs': outputs,
            })

        acc, rec, f1, _ = precision_recall_fscore_support(
            y_true=targets, y_pred=outputs, average=self.average)

        return schedule.monitor.commit_scalars(global_step=schedule.iteration, chapter=self.chapter, **{
            f'{self.name}_acc': acc,
            f'{self.name}_rec': rec,
            f'{self.name}_f1': f1,
        })
