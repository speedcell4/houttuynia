from pathlib import Path

import aku
from torch import nn, optim

from houttuynia.monitors import get_monitor
from houttuynia.schedule import EpochalSchedule
from houttuynia import to_device
from houttuynia.data_loader import iris_data_loader
from houttuynia.extensions import ClipGradNorm, CommitScalarByMean, Evaluation, Snapshot
from houttuynia.utils import launch_expt


class IrisEstimator(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_features: int, dropout: float,
                 bias: bool, negative_slope: float) -> None:
        super(IrisEstimator, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.negative_slope = negative_slope

        self.estimator = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_features, bias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(hidden_features, hidden_features, bias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(hidden_features, num_classes, bias),
        )

        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, data):
        return self.estimator(data)

    def fit(self, batch):
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets)
        classification_pred = outputs.max(-1)[-1]
        acc = (classification_pred == targets).float()
        return loss.sum(0) / loss.size(0), {
            'loss': loss.tolist(),
            'acc': acc.tolist(),
            # 'classification_targets': targets.long().tolist(),
            # 'classification_predictions': classification_pred.long().tolist(),
        }

    def evaluate(self, batch):
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets)
        classification_pred = outputs.max(-1)[-1]
        acc = (classification_pred == targets).float()
        return {
            'loss': loss.tolist(),
            'acc': acc.tolist(),
            # 'classification_targets': targets.long().tolist(),
            # 'classification_predictions': classification_pred.long().tolist(),
        }


app = aku.App(__file__)


@app.register
def train(hidden_features: int = 100, dropout: float = 0.05,
          bias: bool = True, negative_slope: float = 0.05,
          seed: int = 42, device: int = -1, batch_size: int = 1, num_epochs: int = 50, commit_inr: int = 5,
          out_dir: Path = Path('../out'), monitor: ('filesystem', 'tensorboard') = 'tensorboard'):
    """ train iris classifier

    Args:
        hidden_features: the size of hidden layers
        dropout: the dropout ratio
        bias:  whether or not use the bias in hidden layers
        negative_slope: the ratio of negative part
        seed: the random seed number
        device: device id
        batch_size: the size of each batch
        num_epochs: the total numbers of epochs
        commit_inr: commit interval
        out_dir: the root path of output
        monitor: the type of monitor
    """
    expt_dir = launch_expt(**locals())

    train_loader, test_loader = iris_data_loader(batch_size=batch_size)

    estimator = IrisEstimator(4, 3, 100, dropout=dropout, bias=bias, negative_slope=negative_slope)
    optimizer = optim.Adam(estimator.parameters())
    monitor = get_monitor(monitor)(expt_dir=expt_dir)

    to_device(device, estimator)

    schedule = EpochalSchedule(estimator, optimizer, monitor)
    schedule.before_backward(iteration=1)(ClipGradNorm(max_norm=4.))

    schedule.after_epoch(epoch=1)(Snapshot(expt_dir=expt_dir, iris_estimator=estimator))
    schedule.after_epoch(epoch=1)(Evaluation(data_loader=test_loader, chapter='test'))

    schedule.after_iteration(iteration=commit_inr)(
        CommitScalarByMean('loss', 'acc', chapter='train'),
    )
    schedule.after_epoch(epoch=1)(
        CommitScalarByMean('loss', 'acc', chapter='test'),
    )

    return schedule.run(train_loader, num_epochs)


if __name__ == '__main__':
    app.run()
