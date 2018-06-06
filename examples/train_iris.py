import houttuynia as ho

import numpy as np
from numpy.random import RandomState
from matplotlib import pyplot as plt

import torch
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn, cuda, initial_seed, autograd, optim

from houttuynia.schedules import

class MultiClassifierMixin(object):
    def fit(self, X, t):
        y = self(X)
        ho


class IrisClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(IrisClassifier, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, batch):
        return
