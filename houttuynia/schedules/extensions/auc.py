from pathlib import Path

import numpy as np
from numpy import interp
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt

from houttuynia.schedules import Extension, Schedule


# from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_roc(y_test, y_score, n_classes: int, path: Path):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc['micro']),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr['macro'], tpr['macro'],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc['macro']),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc='lower right')
    plt.savefig(path.__str__())
    plt.close()

    return float(roc_auc['micro']), float(roc_auc['macro'])


class AUC(Extension):
    def __init__(self, num_classes: int, chapter: str, name: str):
        super(AUC, self).__init__()
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
            targets = label_binarize(targets, range(self.num_classes))
            path = schedule.monitor.expt_dir / f'roc_{schedule.iteration}.png'
            micro, macro = plot_roc(y_test=targets, y_score=probs, n_classes=self.num_classes, path=path)
            schedule.monitor.commit_scalars(global_step=schedule.iteration, chapter=self.chapter, **{
                f'{self.name}_micro': micro,
                f'{self.name}_macro': macro,
            })
