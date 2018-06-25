import torch
from torch import nn

from houttuynia.nn import init

__all__ = [
    'Conv1d', 'Conv2d', 'Conv3d', 'GramConv1d',
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


class GramConv1d(nn.Sequential):
    def __init__(self, in_features: int, num_grams: int = 5, out_features: int = None, bias: bool = False) -> None:
        if out_features is None:
            out_features = in_features

        self.num_grams = num_grams
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        super(GramConv1d, self).__init__(
            Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            Conv1d(out_features, out_features, kernel_size=num_grams, stride=1, padding=num_grams // 2, bias=bias),
            nn.ReLU(inplace=True),
            Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def reset_parameters(self) -> None:
        self[0].reset_parameters()
        self[2].reset_parameters()
        self[4].reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.transpose(-2, -1)
        outputs = super(GramConv1d, self).forward(inputs)
        return outputs.transpose(-2, -1)
