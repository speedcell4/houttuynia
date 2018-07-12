from typing import List

import torch

import houttuynia as ho


def position_indexes(sequence: List, token=None, total_length: int = None,
                     offset: int = None, min_value: int = None, max_value: int = None):
    if offset is None:
        offset = 0
    if total_length is None:
        total_length = len(sequence)

    token = sequence.index(token) if token is not None else 0

    with torch.no_grad():
        idx = ho.long_tensor(torch.arange(0, total_length)) - token
        if min_value is not None or max_value is not None:
            idx = torch.clamp(idx, min=min_value, max=max_value)
        return idx + offset
