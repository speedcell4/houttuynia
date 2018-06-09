from pathlib import Path

import aku
from torch import nn, optim

from houttuynia.monitors import TensorboardMonitor
from houttuynia.schedules import EpochalSchedule
from houttuynia.nn import Classifier
from houttuynia.datasets import prepare_iris_dataset
from houttuynia.schedule import Moment, Pipeline
from houttuynia.extensions import CommitScalarByMean, Evaluation
from houttuynia.triggers import Periodic


class IrisEstimator(Classifier):
    def __init__(self, in_features: int, num_classes: int, hidden_features: int, bias: bool = True,
                 negative_slope: float = 0.2) -> None:
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.negative_slope = negative_slope

        super(IrisEstimator, self).__init__(estimator=nn.Sequential(
            nn.Linear(in_features, hidden_features, bias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(hidden_features, hidden_features, bias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(hidden_features, num_classes, bias),
        ))


app = aku.App(__file__)


@app.register
def train(hidden_features: int = 100, bias: bool = True, negative_slope: float = 0.2,
          batch_size: int = 1, num_epochs: int = 50, log_dir: Path = Path('log_dir')):
    train, test = prepare_iris_dataset(batch_size)

    estimator = IrisEstimator(
        in_features=4, num_classes=3, hidden_features=hidden_features,
        negative_slope=negative_slope, bias=bias
    )
    optimizer = optim.Adam(estimator.parameters())
    monitor = TensorboardMonitor(log_dir=log_dir)

    schedule = EpochalSchedule(estimator, optimizer, monitor)
    schedule.register_extension(Periodic(Moment.AFTER_ITERATION, iteration=5))(CommitScalarByMean(
        'criterion', 'acc', chapter='train',
    ))
    schedule.register_extension(Periodic(Moment.AFTER_EPOCH, epoch=1))(Pipeline(
        Evaluation(data_loader=test, chapter='test'),
        CommitScalarByMean('criterion', 'acc', chapter='test'),
    ))

    return schedule.run(train, num_epochs)


if __name__ == '__main__':
    app.run()
