from torch import nn

from .functional import accuracy_metric, accuracy_metric_with_logits

__all__ = [
    'AccuracyMetric', 'AccuracyMetricWithLogits',
]


class AccuracyMetric(nn.Module):
    def __init__(self, average_size: bool = True, reduce: bool = True) -> None:
        super(AccuracyMetric, self).__init__()

        self.reduce = reduce
        self.average_size = average_size

    def forward(self, output, target, weight=None):
        return accuracy_metric(
            output=output, target=target, weight=weight,
            average_size=self.average_size, reduce=self.reduce)


class AccuracyMetricWithLogits(nn.Module):
    def __init__(self, dim: int = -1, average_size: bool = True, reduce: bool = True) -> None:
        super(AccuracyMetricWithLogits, self).__init__()

        self.dim = dim
        self.reduce = reduce
        self.average_size = average_size

    def forward(self, output, target, weight=None):
        return accuracy_metric_with_logits(
            output=output, target=target, weight=weight,
            average_size=self.average_size, reduce=self.reduce, dim=self.dim)
