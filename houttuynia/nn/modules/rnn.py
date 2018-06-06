from torch import nn

from houttuynia.nn import init

__all__ = [
    'LSTM', 'LSTMCell',
]


class LSTM(nn.LSTM):
    def reset_parameters(self):
        return init.keras_lstm_(self)


class LSTMCell(nn.LSTMCell):
    def reset_parameters(self):
        return init.keras_lstm_(self)
