import torch
from torch.nn import functional as F
from torch import nn

__all__ = [
    'SkipConnection',
    'ResNet', 'DenseNet', 'DPN', 'MixNet',
]


class SkipConnection(nn.Module):
    def forward(self, inputs: torch.Tensor, *, inner: torch.Tensor, outer: torch.Tensor) -> torch.Tensor:
        """

        Args:
            inputs: (*batches, in_features)
            inner: (*batches, inner_features)
            outer: (*batches, outer_features)
        Returns:
            (*batch, in_features + outer_features)
        """
        raise NotImplementedError


class ResNet(SkipConnection):

    def forward(self, inputs: torch.Tensor, *, inner: torch.Tensor, outer: torch.Tensor = None) -> torch.Tensor:
        """ (He+ 2015, Deep Residual Learning for Image Recognition)

        Args:
            inputs: (*batches, in_features)
            inner: (*batches, in_features)
        Returns:
            (*batch, in_features)
        """
        return inputs + inner


class DenseNet(SkipConnection):
    def forward(self, inputs: torch.Tensor, *, inner: torch.Tensor = None, outer: torch.Tensor) -> torch.Tensor:
        """ (Huang+ 2017, Densely Connected Convolutional Networks)

        Args:
            inputs: (*batches, in_features)
            outer: (*batches, outer_features)
        Returns:
            (*batch, in_features + outer_features)
        """
        return torch.cat([inputs, outer], dim=-1)


class DPN(SkipConnection):
    def forward(self, inputs: torch.Tensor, *, inner: torch.Tensor, outer: torch.Tensor) -> torch.Tensor:
        """ (Chen+ 2017, Dual Path Networks)

        Args:
            inputs: (*batches, in_features)
            inner: (*batches, inner_features)
            outer: (*batches, outer_features)
        Returns:
            (*batch, in_features + outer_features)
        """
        padded_inner = F.pad(inner, (0, inputs.size(-1) - inner.size(-1)), mode='constant', value=0.)
        return torch.cat([inputs + padded_inner, outer], dim=-1)


class MixNet(SkipConnection):
    def forward(self, inputs: torch.Tensor, *, inner: torch.Tensor, outer: torch.Tensor) -> torch.Tensor:
        """ (Wang+ 2018, Mixed Link Networks)

        Args:
            inputs: (*batches, in_features)
            inner: (*batches, inner_features)
            outer: (*batches, outer_features)
        Returns:
            (*batch, in_features + outer_features)
        """
        padded_inner = F.pad(inner, (inputs.size(-1) - inner.size(-1), 0), mode='constant', value=0.)
        return torch.cat([inputs + padded_inner, outer], dim=-1)
