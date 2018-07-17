from typing import Iterable, Union

import torch
from torch import nn

from houttuynia.schedules import Extension, Schedule

__all__ = [
    'ClipGradNorm',
    'ClipGradValue',
]


class ClipGradNorm(Extension):
    def __init__(self, parameters: Iterable[nn.Parameter] = None,
                 max_norm: Union[int, float] = 4., norm_type: Union[int, float] = 2) -> None:
        super(ClipGradNorm, self).__init__()
        if parameters is not None:
            parameters = list(parameters)

        self.max_norm = max_norm
        self.norm_type = norm_type
        self.parameters = parameters

    def __call__(self, schedule: 'Schedule') -> None:
        if self.parameters is None:
            parameters = schedule.estimator.parameters()
        else:
            parameters = self.parameters

        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=self.max_norm, norm_type=self.norm_type)


class ClipGradValue(Extension):
    def __init__(self, parameters: Iterable[nn.Parameter] = None, clip_value: Union[int, float] = 4.) -> None:
        super(ClipGradValue, self).__init__()
        if parameters is not None:
            parameters = list(parameters)

        self.clip_value = clip_value
        self.parameters = parameters

    def __call__(self, schedule: 'Schedule') -> None:
        if self.parameters is None:
            parameters = schedule.estimator.parameters()
        else:
            parameters = self.parameters

        return torch.nn.utils.clip_grad_value_(
            parameters, clip_value=self.clip_value)
