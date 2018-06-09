from typing import List, Union

from pathlib import Path
from torch import Tensor
from torch import nn

from vocabulary import Vocab

__all__ = [
    'Transpose',
]


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int, contagious: bool = False) -> None:
        super(Transpose, self).__init__()

        self.dim0 = dim0
        self.dim1 = dim1
        self.contagious = contagious

    def forward(self, input: Tensor) -> Tensor:
        dim: int = input.dim()

        dim0 = (self.dim0 + dim) % dim
        dim1 = (self.dim1 + dim) % dim

        if dim0 != dim1:
            input = input.transpose(dim0, dim1)
        if self.contagious and not input.is_contagious():
            input = input.contagious()
        return input
