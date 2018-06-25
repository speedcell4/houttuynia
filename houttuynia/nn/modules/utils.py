from torch import Tensor
import numpy as np
import torch
from torch import nn

from houttuynia import long_tensor

__all__ = [
    'Transpose',

    'convert_lengths_to_mask',
    'masked_sentences', 'masked_sentences_',
    'truncated_sentences', 'truncated_sentences_',
]


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int, contagious: bool = False) -> None:
        super(Transpose, self).__init__()

        self.dim0 = dim0
        self.dim1 = dim1
        self.contagious = contagious

    def forward(self, inputs: Tensor) -> Tensor:
        dim: int = inputs.dim()

        dim0 = (self.dim0 + dim) % dim
        dim1 = (self.dim1 + dim) % dim

        if dim0 != dim1:
            inputs = inputs.transpose(dim0, dim1)
        if self.contagious and not inputs.is_contagious():
            inputs = inputs.contagious()
        return inputs


def convert_lengths_to_mask(
        lengths: torch.LongTensor, total_length: int = None,
        batch_first: bool = False, flap: bool = False) -> torch.ByteTensor:
    with torch.no_grad():
        if total_length is None:
            total_length = lengths.max().item()
        indexes = long_tensor(np.arange(0, total_length))
        if batch_first:
            expand_size = lengths.size(0), total_length
            indexes = indexes.view(1, -1).expand(expand_size)
            lengths = lengths.view(-1, 1).expand(expand_size)
        else:
            expand_size = total_length, lengths.size(0)
            indexes = indexes.view(-1, 1).expand(expand_size)
            lengths = lengths.view(1, -1).expand(expand_size)
        if not flap:
            return (indexes < lengths).byte()
        return (indexes >= lengths).byte()


def masked_sentences(
        sentences: torch.FloatTensor,
        mask: torch.ByteTensor, value: float = 0.) -> torch.FloatTensor:
    return sentences.masked_fill(mask.unsqueeze(-1).expand_as(sentences), value)


def masked_sentences_(
        sentences: torch.FloatTensor,
        mask: torch.ByteTensor, value: float = 0.) -> torch.FloatTensor:
    return sentences.masked_fill_(mask.unsqueeze(-1).expand_as(sentences), value)


def truncated_sentences(
        sentences: torch.FloatTensor,
        lengths: torch.LongTensor, value: float = 0., batch_first: bool = False) -> torch.FloatTensor:
    mask = convert_lengths_to_mask(lengths, batch_first=batch_first, flap=True)
    return masked_sentences(sentences, mask, value=value)


def truncated_sentences_(
        sentences: torch.FloatTensor,
        lengths: torch.LongTensor, value: float = 0., batch_first: bool = False) -> torch.FloatTensor:
    mask = convert_lengths_to_mask(lengths, batch_first=batch_first, flap=True)
    return masked_sentences_(sentences, mask, value=value)
