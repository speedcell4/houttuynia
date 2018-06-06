from torch import nn

from houttuynia.nn.init import keras_lstm_

__all__ = [
    'LSTM', 'LSTMCell',
]


class LSTM(nn.LSTM):
    def reset_parameters(self):
        return keras_lstm_(self)


class LSTMCell(nn.LSTMCell):
    def reset_parameters(self):
        return keras_lstm_(self)
