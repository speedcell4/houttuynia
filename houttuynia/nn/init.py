from typing import Union

import torch
from torch.nn import init
from torch import nn

__all__ = [
    'keras_lstm_',
]


def keras_lstm_(lstm: Union[nn.LSTMCell, nn.LSTM]) -> None:
    for name, tensor in lstm.named_parameters():  # type: str, torch.Tensor
        if name.startswith('weight_ih'):
            init.xavier_uniform_(tensor)
        elif name.startswith('weight_hh'):
            init.orthogonal_(tensor)
        elif name.startswith('bias_ih'):
            init.constant_(tensor, 0.)
        elif name.startswith('bias_hh'):
            init.constant_(tensor, 0.)
            tensor[lstm.hidden_size:lstm.hidden_size * 2] = 1.0
