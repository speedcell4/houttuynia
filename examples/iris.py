from pathlib import Path

import aku
from torch import nn, optim

from houttuynia.monitors import get_monitor
from houttuynia.schedules import EpochalSchedule
from houttuynia.nn import Classifier
from houttuynia import log_system, to_device
from houttuynia.datasets import prepare_iris_dataset
from houttuynia.schedule import Moment, Pipeline
from houttuynia.extensions import CommitScalarByMean, Evaluation
from houttuynia.triggers import Periodic
from houttuynia.utils import experiment_hash, ensure_output_dir


class IrisEstimator(Classifier):
    def __init__(self, in_features: int, num_classes: int, hidden_features: int, dropout: float,
                 bias: bool = True, negative_slope: float = 0.2) -> None:
        self.dropout = dropout
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.negative_slope = negative_slope

        super(IrisEstimator, self).__init__(estimator=nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_features, bias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(hidden_features, hidden_features, bias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(hidden_features, num_classes, bias),
        ))


app = aku.App(__file__)


@app.register
def train(hidden_features: int = 100, dropout: float = 0.2, bias: bool = True, negative_slope: float = 0.2,
          device: str = 'cpu',
          batch_size: int = 1, num_epochs: int = 50, out_dir: Path = Path('log_dir'),
          monitor: ('filesystem', 'tensorboard') = 'tensorboard'):
    """ train iris classifier

    Args:
        hidden_features: the size of hidden layers
        dropout: the dropout ratio
        bias:  whether or not use the bias in hidden layers
        negative_slope: the ratio of negative part
        device: device id
        batch_size: the size of each batch
        num_epochs: the total numbers of epochs
        out_dir: the root path of output
        monitor: the type of monitor
    """
    out_dir /= experiment_hash(**locals())
    ensure_output_dir(out_dir)
    log_system.notice(f'experiment output dir => {out_dir}')

    train, test = prepare_iris_dataset(batch_size)

    estimator = IrisEstimator(
        in_features=4, dropout=dropout, num_classes=3, hidden_features=hidden_features,
        negative_slope=negative_slope, bias=bias
    )
    optimizer = optim.Adam(estimator.parameters())
    monitor = get_monitor(monitor)(log_dir=out_dir)

    to_device(device, estimator)

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
