from typing import Tuple

import numpy as np
from more_itertools import islice_extended
import torch

from houttuynia import long_tensor

__all__ = [
    'lens_to_mask',

    'cartesian_view',
    'expanded_masked_fill',

    'masked_sentences', 'masked_sentences_',
    'truncated_sentences', 'truncated_sentences_',
]


def lens_to_mask(
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


def cartesian_view(tensor1: torch.Tensor, tensor2: torch.Tensor,
                   start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert tensor1.dim() == tensor2.dim()

    start1, start2 = tensor1.size()[:start], tensor2.size()[:start]
    end1, end2 = tensor1.size()[end:], tensor2.size()[end:]
    mid1, mid2, mid_common = [], [], []

    for dim1, dim2 in islice_extended(zip(tensor1.size(), tensor2.size()), start, end):
        if dim1 != dim2:
            mid1.extend((dim1, 1))
            mid2.extend((1, dim2))
            mid_common.extend((dim1, dim2))
        else:
            mid1.append(dim1)
            mid2.append(dim2)
            mid_common.append(dim1)

    tensor1 = tensor1.view(*start1, *mid1, *end1).expand(*start1, *mid_common, *end1)
    tensor2 = tensor2.view(*start2, *mid2, *end2).expand(*start2, *mid_common, *end2)
    return tensor1, tensor2


def expanded_masked_fill(tensor: torch.Tensor, mask: torch.ByteTensor,
                         filling_value: float = -float('inf')) -> torch.Tensor:
    """

    Args:
        tensor: (*batches, *nom1, nom2)
        mask: (*batches, nom2)
        filling_value: (,)

    Returns:
        (*batches, *nom1, nom2)
    """
    *batch, dim = mask.size()
    mask = mask.view(*batch, *(1,) * (tensor.dim() - mask.dim()), dim).expand_as(tensor)
    return tensor.masked_fill(mask, filling_value)


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
    mask = lens_to_mask(lengths, batch_first=batch_first, flap=True)
    return masked_sentences(sentences, mask, value=value)


def truncated_sentences_(
        sentences: torch.FloatTensor,
        lengths: torch.LongTensor, value: float = 0., batch_first: bool = False) -> torch.FloatTensor:
    mask = lens_to_mask(lengths, batch_first=batch_first, flap=True)
    return masked_sentences_(sentences, mask, value=value)
