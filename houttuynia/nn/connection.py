import torch
from torch.nn import init
from torch import nn

__all__ = [
    'Highway',
]


class Highway(nn.Module):
    def __init__(self, in_features: int, bias: bool = True) -> None:
        super(Highway, self).__init__()

        self.in_features = in_features
        self.out_in_features = in_features
        self.bias = bias

        self.fc = nn.Linear(in_features, in_features, bias=bias)
        self.carry = nn.Linear(in_features, in_features, bias=bias)
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.fc.weight)
        gain = init.calculate_gain(self.sigmoid.__class__.__name__.lower())
        init.xavier_uniform_(self.carry.weight, gain)
        if self.bias:
            init.constant_(self.fc.bias, 0.)
            init.constant_(self.carry.bias, 1.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        t = self.sigmoid(self.carry(x))

        return t * y + (1. - t) * x
