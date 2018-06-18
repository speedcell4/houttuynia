import torch
from torch import nn

from houttuynia.nn.init import positional_

__all__ = [
    'PositionalEmbedding',
]


class PositionalEmbedding(nn.Module):
    def __init__(self, sentence_length: int, token_features: int, freeze_position: bool = True) -> None:
        super(PositionalEmbedding, self).__init__()

        self.token_features = token_features
        self.sentence_length = sentence_length
        self.freeze_position = freeze_position
        self.position_embedding = nn.Embedding(sentence_length, token_features)

        self.reset_parameters()

    def reset_parameters(self):
        positional_(self.position_embedding.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sentence, features = inputs.shape[-2:]
        position = self.position_embedding.weight[:sentence, :features]
        if self.freeze_position:
            position = position.detach()
        return inputs + position


if __name__ == '__main__':
    embedding = PositionalEmbedding(100, 20, True)
    x = torch.rand(4, 12, 20)
    y = embedding(x)
    print(x.shape)
    print(y.shape)
