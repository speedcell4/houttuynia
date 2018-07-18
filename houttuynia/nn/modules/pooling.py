import torch
from torch import nn


class PiecewiseMaxPool1d(nn.Module):
    def __init__(self, out_features: int, num_pieces: int):
        super(PiecewiseMaxPool1d, self).__init__()

        self.out_features = out_features
        self.num_pieces = num_pieces

        self.pool = nn.AdaptiveMaxPool1d(out_features, return_indices=False)

    def forward(self, inputs: torch.FloatTensor, mask: torch.LongTensor) -> torch.FloatTensor:
        """

        Args:
            inputs: (batch, channel, in_features)
            mask: (batch, channel)

        Returns:
            (batch, channel, num_pieces, out_features)
        """
        outputs = []
        for ix in range(1, self.num_pieces + 1):
            y = self.pool(inputs.masked_fill(mask.unsqueeze(-1) != ix, -float('inf')))
            outputs.append(y.masked_fill_(y == -float('inf'), 0.))
        return torch.stack(outputs, dim=-2)
