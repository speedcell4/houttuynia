from typing import Dict, Tuple

from torch import nn
from torch import Tensor

from houttuynia import log_system
from houttuynia.nn.modules.metrics import AccuracyMetricWithLogits

__all__ = [
    'Architecture', 'Classifier',
]


class Architecture(nn.Module):
    def forward(self, *input) -> Tensor:
        raise NotImplementedError

    def fit(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, float]]:
        raise NotImplementedError

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        _, metrics = self.fit(*args, **kwargs)
        return metrics

    def setattr_with_notice(self, name: str, value) -> None:
        setattr(self, name, value)
        return log_system.notice(f'{self.__class__.__name__}.{name} := {value.__class__.__name__}')


class Classifier(Architecture):
    num_classes: int

    def __init__(self, estimator: nn.Module, criterion=None, **kwargs):
        super(Classifier, self).__init__()
        self.estimator = estimator

        if criterion is None:
            if self.num_classes == 1:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
        self.setattr_with_notice('criterion', criterion)

        for name, metric in kwargs.items():
            if name.endswith('_metric'):
                self.setattr_with_notice(name, metric)
        else:
            self.setattr_with_notice('acc_metric', AccuracyMetricWithLogits())

    def forward(self, inputs):
        return self.estimator(inputs)

    def fit(self, inputs, targets, *args, **kwargs) -> Tuple[Tensor, Dict[str, float]]:
        outputs = self(inputs, *args, **kwargs)
        criterion = self.criterion(outputs, targets)

        metrics = {
            name[:-'_metric'.__len__()]: metric(outputs, targets).item()
            for name, metric in self.named_modules() if name.endswith('_metric')
        }
        metrics.update(criterion=criterion.item())
        return criterion, metrics


class EncoderDecoder(Architecture):
    pass


class TransitionParser(Architecture):
    pass


class VAE(Architecture):
    pass


class GAN(Architecture):
    pass
