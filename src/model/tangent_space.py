"""Tangent space projection module for sEMG covariance features."""

import torch
import torch.nn as nn


class TangentSpace(nn.Module):
    """Covariance of causal sub-windows, shrunk with OAS and mapped to tangent space at identity."""

    def __init__(self, channels: int = 8, sub_window: int = 60, step: int = 20) -> None:
        super().__init__()
        self.channels = channels
        self.sub_window = sub_window
        self.step = step

        row, col = torch.triu_indices(channels, channels)
        self.register_buffer("row", row)
        self.register_buffer("col", col)
        self.register_buffer("weight", torch.where(row == col, 1.0, 2.0**0.5))
        self.out_dim = len(row)

    def covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute OAS-regularized covariance of x [..., C, T]."""
        c, t = x.shape[-2:]
        x_centered = x - x.mean(-1, keepdim=True)
        s = x_centered @ x_centered.transpose(-1, -2) / (t - 1)
        mu = s.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
        tr_s2 = (s * s).sum((-2, -1), keepdim=True)
        num = (1.0 - 2.0 / c) * tr_s2 + (mu * c) ** 2
        den = (t + 1.0 - 2.0 / c) * (tr_s2 - (mu * c) ** 2 / c)
        rho = (num / (den + 1e-8)).clamp(0.0, 1.0)
        eye = torch.eye(c, device=x.device, dtype=x.dtype)
        return (1.0 - rho) * s + rho * mu * eye + 1e-5 * eye

    def project(self, cov: torch.Tensor) -> torch.Tensor:
        """Project covariance matrix/matrices [..., C, C] onto tangent space at identity."""
        evals, evecs = torch.linalg.eigh(cov)
        log_evals = evals.clamp(min=1e-6).log()
        log_cov = evecs @ torch.diag_embed(log_evals) @ evecs.transpose(-1, -2)
        return log_cov[..., self.row, self.col] * self.weight

    @torch.no_grad()
    def compute_single_subwindow(self, sub_window_data: torch.Tensor) -> torch.Tensor:
        """Compute tangent vector for a single sub-window [C, W] or batch [B, C, W]."""
        return self.project(self.covariance(sub_window_data))

    @torch.no_grad()
    def compute_sequence(self, sub_windows_seq: torch.Tensor) -> torch.Tensor:
        """Compute tangent vectors for sequence of sub-windows [B, S, C, W] or [S, C, W]."""
        return self.project(self.covariance(sub_windows_seq))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG window [B, C, W] or [C, W] -> tangent sequence [B, S, out_dim] or [S, out_dim]."""
        is_2d = x.ndim == 2
        if is_2d:
            x = x.unsqueeze(0)

        x_aligned = x[..., (x.shape[-1] - self.sub_window) % self.step :]
        subs = x_aligned.unfold(-1, self.sub_window, self.step).transpose(1, 2)
        out = self.project(self.covariance(subs))
        return out.squeeze(0) if is_2d else out
