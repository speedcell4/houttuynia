from typing import Tuple, Union

import torch
from torch.nn import init
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from houttuynia.nn.init import keras_lstm_

__all__ = [
    'LSTM',
]


class LSTM(nn.LSTM):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True,
                 batch_first: bool = True, dropout: float = 0., bidirectional: bool = False):
        super(LSTM, self).__init__(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
            batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.h0 = nn.Parameter(torch.FloatTensor(num_layers * num_directions, 1, hidden_size))
        self.c0 = nn.Parameter(torch.FloatTensor(num_layers * num_directions, 1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'h0'):
            init.uniform_(self.h0, a=-0.05, b=+0.05)
            init.uniform_(self.c0, a=-0.05, b=+0.05)
        return keras_lstm_(self)

    def forward(self, input: Union[torch.Tensor, PackedSequence],
                hx: Tuple[torch.Tensor, torch.Tensor] = None) -> Union[torch.Tensor, PackedSequence]:
        if hx is None:
            if isinstance(input, PackedSequence):
                h0 = self.h0.expand(-1, input[1][0], -1)
                c0 = self.c0.expand(-1, input[1][0], -1)
            else:
                h0 = self.h0.expand(-1, input.size(0), -1)
                c0 = self.c0.expand(-1, input.size(0), -1)
            hx = (h0, c0)
        return super().forward(input, hx)
