import torch
from torch import nn
from torch.nn import init

__all__ = [
    'Highway',
]


class Highway(nn.Module):
    def __init__(self, in_features: int, bias: bool = True) -> None:
        super(Highway, self).__init__()

        self.in_features = in_features
        self.out_features = in_features
        self.bias = bias

        self.transform = nn.Linear(in_features, in_features, bias=bias)
        self.carry = nn.Linear(in_features, in_features, bias=bias)
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        name = self.sigmoid.__class__.__name__.lower()
        gain = init.calculate_gain(name)

        init.xavier_uniform_(self.transform.weight, 1.0)
        init.xavier_uniform_(self.carry.weight, gain)
        if self.bias:
            init.constant_(self.transform.bias, 0.)
            init.constant_(self.carry.bias, 1.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transform = self.transform(x)
        carry = self.sigmoid(self.carry(x))
        return carry * transform + (1 - carry) * x
