from torch import nn

from houttuynia.nn import init


class Conv1d(nn.Conv1d):
    def reset_parameters(self):
        return init.keras_conv_(self)


class Conv2d(nn.Conv2d):
    def reset_parameters(self):
        return init.keras_conv_(self)


class Conv3d(nn.Conv3d):
    def reset_parameters(self):
        return init.keras_conv_(self)
