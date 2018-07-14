from typing import List

import torch

import houttuynia as ho


def idx_pos_indexing(token_idx: int, total_length: int,
                     offset: int, min_value: int = None, max_value: int = None):
    if offset is None:
        offset = 0
    with torch.no_grad():
        idx = ho.long_tensor(torch.arange(0, total_length)) - token_idx
        if min_value is not None or max_value is not None:
            idx = torch.clamp(idx, min=min_value, max=max_value)
        return idx + offset


def seq_pos_indexing(sequence: List, token=None, total_length: int = None,
                     offset: int = None, min_value: int = None, max_value: int = None):
    if total_length is None:
        total_length = len(sequence)

    token_idx = sequence.index(token) if token is not None else 0
    return idx_pos_indexing(token_idx=token_idx, total_length=total_length,
                            offset=offset, min_value=min_value, max_value=max_value)
