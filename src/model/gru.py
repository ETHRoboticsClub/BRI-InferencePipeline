"""GRU encoder network processing temporal sequences of tangent vectors."""

from typing import Sequence

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """Dense projection followed by two stacked GRU layers, returning the last time step."""

    def __init__(
        self,
        in_dim: int = 36,
        dense: int = 256,
        units: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(in_dim, dense), nn.Tanh())
        self.gru1 = nn.GRU(dense, units[0], batch_first=True)
        self.gru2 = nn.GRU(units[0], units[1], batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = units[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform sequence [B, S, in_dim] or [S, in_dim] -> [B, out_dim] or [out_dim]."""
        is_2d = x.ndim == 2
        if is_2d:
            x = x.unsqueeze(0)

        h1, _ = self.gru1(self.dense(x))
        h2, _ = self.gru2(self.dropout(h1))
        out = self.dropout(h2[:, -1])
        return out.squeeze(0) if is_2d else out
