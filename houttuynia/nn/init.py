from typing import Union

import torch
from torch.nn import init
from torch import nn

__all__ = [
    'keras_conv_', 'keras_lstm_',
    'positional_',
]


def keras_conv_(module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
                mode: str = 'fan_out', nonlinearity: str = 'relu') -> None:
    for name, tensor in module.named_parameters():  # type: str, torch.Tensor
        if name.startswith('weight'):
            init.kaiming_uniform_(tensor, mode=mode, nonlinearity=nonlinearity)
        elif name.startswith('bias'):
            init.constant_(tensor, 0.)


def keras_lstm_(module: Union[nn.LSTMCell, nn.LSTM]) -> None:
    for name, tensor in module.named_parameters():  # type: str, torch.Tensor
        if name.startswith('weight_ih'):
            init.xavier_uniform_(tensor)
        elif name.startswith('weight_hh'):
            init.orthogonal_(tensor)
        elif name.startswith('bias_'):
            init.constant_(tensor, 0.)
            tensor[module.hidden_size:module.hidden_size * 2] = 0.5


def positional_(tensor: torch.Tensor) -> None:
    assert tensor.dim() == 2

    sentence, features = tensor.size()
    pos = torch.arange(0, sentence).float()
    ixs = torch.arange(0, features).float()
    ixs = 1. / torch.pow(10000., torch.floor(ixs / 2) * 2 / features)
    with torch.no_grad():
        tensor = pos.view(-1, 1) @ ixs.view(1, -1)
        tensor[..., 0::2].sin_()
        tensor[..., 1::2].cos_()
