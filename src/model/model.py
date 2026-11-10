from __future__ import annotations

import torch
from torch import nn

from src.model.base import BaseEMGModel


class Model(BaseEMGModel):
    """Template sEMG-to-kinematics model `(B, C, W) -> (B, 15)`.

    Replace this class on a feature branch with your own architecture
    (e.g. CNN, TCN, Transformer, GRU, or Riemannian decoder).
    """

    def __init__(
        self,
        channels: int = 8,
        targets: int = 15,
        window_size: int = 240,
        hidden: int = 64,
        **_kwargs: object,
    ) -> None:
        super().__init__(channels=channels, targets=targets, window_size=window_size)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.channels * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.targets),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 170.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        rms = torch.sqrt(torch.mean(x**2, dim=-1) + 1e-6)
        mav = torch.mean(torch.abs(x), dim=-1)
        feats = torch.cat([rms, mav], dim=-1)
        return torch.clamp(self.net(feats), 0.0, 180.0)
