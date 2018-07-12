import torch
from torch.nn import functional as F
from torch import nn
from torch.nn import init

from houttuynia.nn.modules.utils import expanded_masked_fill


class Attention(nn.Module):
    def forward(self, Q: torch.FloatTensor, K: torch.FloatTensor, V: torch.FloatTensor,
                mask: torch.ByteTensor = None) -> torch.FloatTensor:
        """
        Args:
            Q: (*batch, nom1, vector1)
            K: (*batch, nom2, vector1)
            V: (*batch, nom2, vector2)
            mask: (*batch, nom2)
        Returns:
            (*batch, nom1, vector2)
        """
        raise NotImplementedError


class FacetAttention(Attention):
    def __init__(self, in_features: int, negative_slope: float = 0, bias: int = False) -> None:
        super(FacetAttention, self).__init__()

        self.in_features = in_features
        self.negative_slope = negative_slope

        self.fc = nn.Sequential(
            nn.Linear(in_features * self.num_facts, in_features * self.num_facts, bias=bias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(in_features * self.num_facts, 1, bias=bias),
        )

    def forward(self, Q: torch.FloatTensor, K: torch.FloatTensor, V: torch.FloatTensor,
                mask: torch.ByteTensor = None) -> torch.FloatTensor:
        """
        Args:
            Q: (*batch, nom1, vector1)
            K: (*batch, nom2, vector1)
            V: (*batch, nom2, vector2)
            mask: (*batch, nom2)
        Returns:
            (*batch, nom1, vector2)
        """

        *batch, nom1, vector1 = Q.size()
        *batch, nom2, vector2 = V.size()

        Q = Q.view(*batch, nom1, 1, vector1).expand(*batch, nom1, nom2, vector1)
        K = K.view(*batch, 1, nom2, vector1).expand(*batch, nom1, nom2, vector1)
        A = self.fc(self.facets(Q, K)).squeeze(-1)  # (*batch, nom1, nom2)
        if mask is not None:
            A = expanded_masked_fill(A, ~mask, filling_value=-float('inf'))
        return F.softmax(A, dim=-1) @ V

    num_facts: int = 4

    @classmethod
    def facets(cls, Q: torch.FloatTensor, K: torch.FloatTensor) -> torch.FloatTensor:
        return torch.cat([Q, K, torch.abs(Q - K), Q * K], dim=-1)


class DotProductAttention(Attention):
    def __init__(self):
        super(DotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q: torch.FloatTensor, K: torch.FloatTensor, V: torch.FloatTensor,
                mask: torch.ByteTensor = None) -> torch.FloatTensor:
        """

        Args:
            Q: (*batch, nom1, vector1)
            K: (*batch, nom2, vector1)
            V: (*batch, nom2, vector2)
            mask: (*batch, nom2)
        Returns:
            (*batch, nom1, vector2)
        """
        A = Q @ K.transpose(-2, -1) / Q.size(-1) ** 0.5
        if mask is not None:
            A = expanded_masked_fill(A, ~mask, filling_value=-float('inf'))
        return self.softmax(A) @ V


class MultiHeadAttention(Attention):
    def __init__(self, num_heads: int, key_features: int, value_features: int, out_features: int = None):
        super(MultiHeadAttention, self).__init__()

        if out_features is None:
            out_features = value_features
        assert out_features % num_heads == 0

        self.num_heads = num_heads
        self.out_features = out_features
        self.key_features = key_features
        self.value_features = value_features

        self.attention = DotProductAttention()
        self.W = nn.Parameter(torch.Tensor(out_features, out_features))
        self.Q = nn.Parameter(torch.Tensor(key_features, out_features))
        self.K = nn.Parameter(torch.Tensor(key_features, out_features))
        self.V = nn.Parameter(torch.Tensor(value_features, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.Q)
        init.xavier_uniform_(self.K)
        init.xavier_uniform_(self.V)

    def forward(self, Q: torch.FloatTensor, K: torch.FloatTensor, V: torch.FloatTensor,
                mask: torch.ByteTensor = None) -> torch.FloatTensor:
        """

        Args:
            Q: (*batch, nom1, key_features)
            K: (*batch, nom2, key_features)
            V: (*batch, nom2, value_features)
            mask: (*batch, nom2)

        Returns:
            (*batch, nom1, out_features)

        """
        Q = (Q @ self.Q).unsqueeze(-1).view(*Q.size()[:-1], self.num_heads, -1)
        K = (K @ self.K).unsqueeze(-1).view(*K.size()[:-1], self.num_heads, -1)
        V = (V @ self.V).unsqueeze(-1).view(*V.size()[:-1], self.num_heads, -1)
        A = self.attention(
            Q.transpose(-2, -3),
            K.transpose(-2, -3),
            V.transpose(-2, -3),
            mask=mask,
        ).transpose(-2, -3)
        return A.contiguous().view(*A.size()[:-2], -1) @ self.W
