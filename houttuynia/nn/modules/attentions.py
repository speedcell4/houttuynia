import torch
from torch.nn import init
from torch import nn
from torch import Tensor

from houttuynia.nn import dot_product_attention

__all__ = [
    'Attention',
    'DotProduct',
    'MultiHead',
]


class Attention(nn.Module):
    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        raise NotImplementedError


class DotProduct(Attention):
    def __init__(self, in_features: float):
        super(DotProduct, self).__init__()
        self.temperature = 1 / in_features ** 0.5

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        return dot_product_attention(Q, K, V, self.temperature)


class MultiHead(Attention):
    def __init__(self, key_features: int, value_feature: int, out_features: int, num_heads: int) -> None:
        super(MultiHead, self).__init__()
        assert out_features % num_heads == 0

        self.key_features = key_features
        self.value_feature = value_feature
        self.out_features = out_features
        self.model_feature = out_features // num_heads
        self.num_heads = num_heads

        self.W = nn.Parameter(torch.Tensor(out_features, out_features).float())
        self.Q = nn.Parameter(torch.Tensor(key_features, out_features).float())
        self.K = nn.Parameter(torch.Tensor(key_features, out_features).float())
        self.V = nn.Parameter(torch.Tensor(value_feature, out_features).float())
        self.attention = DotProduct(in_features=self.model_feature)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.Q)
        init.xavier_uniform_(self.K)
        init.xavier_uniform_(self.V)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        Q = (Q @ self.Q).view([*Q.size()[:-2], self.num_heads, self.model_feature])
        K = (K @ self.K).view([*K.size()[:-2], self.num_heads, self.model_feature])
        V = (V @ self.V).view([*V.size()[:-2], self.num_heads, self.model_feature])
        A = self.attention(
            Q=Q.transpose(-3, -2),
            K=K.transpose(-3, -2),
            V=V.transpose(-3, -2),
        ).transpose(-3, -2).contiguous()
        return (A @ self.W).view([*A.size()[:-2], self.out_features])
