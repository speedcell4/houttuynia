import torch
from torch import nn

from houttuynia.nn import init

__all__ = [
    'Conv1d', 'Conv2d', 'Conv3d', 'GramConv1',
]


class Conv1d(nn.Conv1d):
    def reset_parameters(self) -> None:
        return init.keras_conv_(self)


class Conv2d(nn.Conv2d):
    def reset_parameters(self) -> None:
        return init.keras_conv_(self)


class Conv3d(nn.Conv3d):
    def reset_parameters(self) -> None:
        return init.keras_conv_(self)


class GramConv1(nn.Sequential):
    def __init__(self, in_features: int, num_grams: int, out_features: int = None, bias: bool = True) -> None:
        if out_features is None:
            out_features = in_features

        self.num_grams = num_grams
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        super(GramConv1, self).__init__(
            Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            Conv1d(out_features, out_features, kernel_size=num_grams, stride=1, padding=num_grams // 2, bias=bias),
            nn.ReLU(inplace=True),
            Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self[0].reset_parameters()
        self[2].reset_parameters()
        self[4].reset_parameters()

    def forward(self, inputs: torch.Tensor, dim: int = -1) -> torch.Tensor:
        inputs = inputs.transpose(-2, dim)
        outputs = super(GramConv1, self).forward(inputs)
        return outputs.transpose(-2, dim)
