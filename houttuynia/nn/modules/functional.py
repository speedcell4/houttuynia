import torch
from torch.nn import functional as F

from torch import Tensor

__all__ = [
    'accuracy_metric', 'accuracy_metric_with_logits',
    'dot_product_attention',
]


def accuracy_metric(
        output: Tensor, target: Tensor,
        weight: Tensor = None, average_size: bool = True, reduce: bool = True) -> Tensor:
    assert output.size() == target.size()
    assert output.dtype == target.dtype == torch.int64

    ret = (output == target).float()

    if weight is not None:
        ret = ret * weight
    if not reduce:
        return ret
    elif average_size:
        return ret.mean()
    else:
        return ret.sum()


def accuracy_metric_with_logits(
        output: Tensor, target: Tensor, dim: int = -1,
        weight: Tensor = None, average_size: bool = True, reduce: bool = True) -> Tensor:
    return accuracy_metric(
        output=output.argmax(dim=dim), target=target,
        weight=weight, average_size=average_size, reduce=reduce)


def dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, temperature: float) -> Tensor:
    return F.softmax(Q @ K.transpose(-2, -1) * temperature, dim=-1) @ V
