from torch import Tensor
import torch
from torch.nn import init
from torch.nn import functional as F
from torch import nn

import houttuynia as ho
from houttuynia.nn.init import positional_

__all__ = [
    'EmbeddingLayer',
    'MultiHead',
    'TransformerEncoderBlock',
    'TransformerEncoderLayer',
]


class MultiHead(nn.Module):
    def __init__(self, num_heads: int, out_features: int,
                 key_features: int = None, value_features: int = None) -> None:
        super(MultiHead, self).__init__()
        assert out_features % num_heads == 0

        if key_features is None:
            key_features = out_features
        if value_features is None:
            value_features = out_features

        self.key_features = key_features
        self.value_features = value_features
        self.out_features = out_features
        self.model_features = out_features // num_heads
        self.num_heads = num_heads

        self.W = nn.Parameter(torch.Tensor(out_features, out_features).float())
        self.Q = nn.Parameter(torch.Tensor(key_features, out_features).float())
        self.K = nn.Parameter(torch.Tensor(key_features, out_features).float())
        self.V = nn.Parameter(torch.Tensor(value_features, out_features).float())

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.Q)
        init.xavier_uniform_(self.K)
        init.xavier_uniform_(self.V)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        Q = (Q @ self.Q).view([*Q.size()[:-1], self.num_heads, self.model_feature])
        K = (K @ self.K).view([*K.size()[:-1], self.num_heads, self.model_feature])
        V = (V @ self.V).view([*V.size()[:-1], self.num_heads, self.model_feature])
        A = torch.einsum('bqhf,bkhf->bqhk', (Q, K))
        A = F.softmax(A / (self.model_features ** 0.5), dim=-1)
        A = torch.einsum('bqhk,bkhf->bqhf', (A, V))
        return A.contiguous().view([*A.size()[:-2], -1]) @ self.W


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_heads: int, in_features: int,
                 n_gram: int = 3, bias: bool = True) -> None:
        super(TransformerEncoderBlock, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features

        self.head = MultiHead(num_heads=num_heads, out_features=in_features)
        self.norm = nn.LayerNorm(normalized_shape=in_features)
        self.fc = nn.Sequential(
            ho.nn.Conv1d(in_features, in_features, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            ho.nn.Conv1d(in_features, in_features, kernel_size=n_gram, stride=1, padding=n_gram // 2, bias=bias),
            nn.ReLU(inplace=True),
            ho.nn.Conv1d(in_features, in_features, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.head.reset_parameters()
        self.norm.reset_parameters()
        self.fc[0].reset_parameters()
        self.fc[2].reset_parameters()
        self.fc[4].reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.norm(self.head(inputs, inputs, inputs) + inputs)
        return self.fc(outputs.transpose(-2, -1)).transpose(-2, -1)


class EmbeddingLayer(nn.Module):
    def __init__(self, num_tokens: int, token_features: int, freeze_position: bool = True) -> None:
        super(EmbeddingLayer, self).__init__()

        self.freeze_position = freeze_position
        self.token_embedding = nn.Embedding(num_tokens, token_features)
        self.position_embedding = nn.Embedding(num_tokens, token_features)

        self.reset_parameters()

    def reset_parameters(self):
        self.token_embedding.reset_parameters()
        positional_(self.position_embedding.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sentence, features = inputs.shape
        position = self.position_embedding.weight[sentence:, features:]
        if self.freeze_position:
            position = position.detach()
        return self.token_embedding(inputs) + position


class TransformerEncoderLayer(nn.Sequential):
    def __init__(self, num_layers: int, num_heads: int, in_features: int,
                 n_gram: int = 3, bias: bool = True) -> None:
        super(TransformerEncoderLayer, self).__init__([
            TransformerEncoderBlock(num_heads, in_features, n_gram, bias)
            for _ in range(num_layers)
        ])


if __name__ == '__main__':
    EmbeddingLayer(100, 20)
