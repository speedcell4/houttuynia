from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

from houttuynia.schedules import Extension, Schedule


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
def _measure_and_plot(y_score, y_test, n_classes: int, path: Path):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    precision['micro'], recall['micro'], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision['micro'] = average_precision_score(y_test, y_score, average='micro')

    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
    plt.fill_between(recall['micro'], precision['micro'], step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'
              .format(average_precision['micro']))
    plt.savefig(path.__str__())
    plt.close()

    return average_precision['micro']


class PRCurve(Extension):
    filename = r'pr_curve_{iteration}.png'

    def __init__(self, num_classes: int, chapter: str, name: str):
        super(PRCurve, self).__init__()
        self.num_classes = num_classes
        self.chapter = chapter
        self.name = name

    def __call__(self, schedule: 'Schedule') -> None:
        if not schedule.monitor.contains(self.chapter, f'{self.name}_probs'):
            raise KeyError(f'{self.name}_probs not found')
        if not schedule.monitor.contains(self.chapter, f'{self.name}_targets'):
            raise KeyError(f'{self.name}_targets not found')

        if schedule.monitor.contains(self.chapter, f'{self.name}_targets'):
            probs = schedule.monitor.get(self.chapter, f'{self.name}_probs')
            targets = schedule.monitor.get(self.chapter, f'{self.name}_targets')

            probs = np.array(probs, dtype=np.float)
            targets = np.array(targets, dtype=np.int)

            np.save(str(schedule.monitor.expt_dir / f'pr_probs_{schedule.iteration}.npy'), probs)
            np.save(str(schedule.monitor.expt_dir / f'pr_targets_{schedule.iteration}.npy'), targets)

            targets = label_binarize(targets, range(self.num_classes))
            path = schedule.monitor.expt_dir / self.filename.format(iteration=schedule.iteration)
            micro = _measure_and_plot(y_test=targets, y_score=probs, n_classes=self.num_classes, path=path)
            schedule.monitor.commit_scalars(global_step=schedule.iteration, chapter=self.chapter, **{
                f'{self.name}_pr_micro_auc': micro,
            })
