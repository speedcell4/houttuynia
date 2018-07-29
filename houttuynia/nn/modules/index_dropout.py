from typing import Union

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.distributions import Bernoulli

__all__ = [
    'IndexDropout',
]


class IndexDropout(nn.Module):
    def __init__(self, prob: float, filling_idx: int):
        super(IndexDropout, self).__init__()
        self.prob = prob
        self.filling_idx = filling_idx
        self.dist = Bernoulli(probs=prob)

    def forward(self, index: Union[torch.LongTensor, PackedSequence]) -> Union[torch.LongTensor, PackedSequence]:
        inputs = index[1] if isinstance(index, PackedSequence) else index
        mask = self.dist.sample(inputs.size()).byte()
        outputs = inputs.masked_fill(mask, self.filling_idx)

        if isinstance(index, PackedSequence):
            return PackedSequence(outputs, index[1])
        return outputs
