import numpy as np
from numpy.random import RandomState
from matplotlib import pyplot as plt

import torch
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn, cuda, initial_seed, autograd, optim

from houttuynia import config


class ClassifierMixin(object):
    def __init__(self, criterion=None, metric=None, *args, **kwargs):
        super(ClassifierMixin, self).__init__(*args, **kwargs)

        if criterion is None:
            if self.num_classes == 1:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()

        if metric is None:
            if self.num_classes == 1:
                metric = nn.BinaryAccuracy()
            else:
                metric = nn.Accuracy()

        self.criterion = criterion
        self.metric = metric

    def fit(self, inputs, targets, global_step):
        logits = self(inputs)
        criterion = self.criterion(logits, targets)
        metric = self.metric(logits, targets)

        config['reporter'].report_scalars(
            criterion=criterion.item(),
            metric=metric.item(),
            global_step=global_step,
        )

        return criterion

    def transform(self, inputs, targets, global_step):
        logits = self.__call__(inputs)
        criterion = self.criterion(logits, targets)
        metric = self.metric(logits, targets)

        config['reporter'].report_scalars(
            criterion=criterion.item(),
            metric=metric.item(),
            global_step=global_step,
        )

        return criterion


def classifier(criterion=None, metric=None):
    def wrapper(cls):
        return type(cls.__name__, (ClassifierMixin, cls))
