import torch
from torch import nn

from houttuynia.nn import init

__all__ = [
    'Conv1d', 'Conv2d', 'Conv3d',
    'GramConv1d',
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
    def __init__(self, in_features: int, num_grams: int = 5,
                 out_features: int = None, hidden_features: int = None,
                 bias: bool = False, negative_slope: float = 0.) -> None:
        if out_features is None:
            out_features = in_features
        if hidden_features is None:
            hidden_features = max(in_features, out_features)

        self.num_grams = num_grams
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        super(GramConv1d, self).__init__(
            Conv1d(in_channels=in_features, out_channels=hidden_features,
                   stride=1, bias=bias, kernel_size=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            Conv1d(in_channels=hidden_features, out_channels=hidden_features,
                   stride=1, bias=bias, kernel_size=num_grams, padding=num_grams // 2),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            Conv1d(in_channels=hidden_features, out_channels=hidden_features,
                   stride=1, bias=bias, kernel_size=1, padding=0),
        )

    def reset_parameters(self) -> None:
        self[0].reset_parameters()
        self[2].reset_parameters()
        self[4].reset_parameters()

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """

        Args:
            inputs: (batch, in_features, times)

        Returns:
            (batch, out_features, times)
        """
        return super(GramConv1d, self).forward(inputs)
