"""Temporal Riemannian Regressor (TRR) full model implementation."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.model.gru import GRUEncoder
from src.model.tangent_space import TangentSpace

TRAINING_KEY_MAP: dict[str, str] = {
    "rfe.triu_row": "tangent_space.row",
    "rfe.triu_col": "tangent_space.col",
    "rfe.triu_weights": "tangent_space.weight",
    "input_dense.0.weight": "gru.dense.0.weight",
    "input_dense.0.bias": "gru.dense.0.bias",
    "gru1.weight_ih_l0": "gru.gru1.weight_ih_l0",
    "gru1.weight_hh_l0": "gru.gru1.weight_hh_l0",
    "gru1.bias_ih_l0": "gru.gru1.bias_ih_l0",
    "gru1.bias_hh_l0": "gru.gru1.bias_hh_l0",
    "gru2.weight_ih_l0": "gru.gru2.weight_ih_l0",
    "gru2.weight_hh_l0": "gru.gru2.weight_hh_l0",
    "gru2.bias_ih_l0": "gru.gru2.bias_ih_l0",
    "gru2.bias_hh_l0": "gru.gru2.bias_hh_l0",
    "hidden_dense.0.weight": "head.0.weight",
    "hidden_dense.0.bias": "head.0.bias",
    "out.weight": "head.2.weight",
    "out.bias": "head.2.bias",
}


class Model(nn.Module):
    """Temporal Riemannian Regressor (Colot et al., 2026)."""

    def __init__(
        self,
        channels: int = 8,
        targets: int = 15,
        hidden: int = 64,
        sub_window: int = 60,
        step: int = 20,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.targets = targets
        self.hidden = hidden

        self.tangent_space = TangentSpace(channels=channels, sub_window=sub_window, step=step)
        self.gru = GRUEncoder(in_dim=self.tangent_space.out_dim)
        self.head = nn.Sequential(
            nn.Linear(self.gru.out_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, targets),
        )
        self.target_mean: torch.Tensor | None = None
        self.target_std: torch.Tensor | None = None
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from raw/filtered EMG window to joint angles in degrees."""
        is_2d = x.ndim == 2
        if is_2d:
            x = x.unsqueeze(0)

        tangent_seq = self.tangent_space(x)
        angles = self.forward_from_tangent(tangent_seq)
        return angles.squeeze(0) if is_2d else angles

    def forward_from_tangent(self, tangent_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass directly from a sequence of Riemannian tangent vectors."""
        is_2d = tangent_seq.ndim == 2
        if is_2d:
            tangent_seq = tangent_seq.unsqueeze(0)

        expected_in = self.gru.dense[0].in_features
        if tangent_seq.shape[-1] != expected_in:
            if tangent_seq.shape[-1] == 36 and expected_in == 108:
                tangent_seq = tangent_seq.repeat(1, 1, 3)
            elif tangent_seq.shape[-1] == 36 and expected_in > 108:
                rep = tangent_seq.repeat(1, 1, 3)
                pad = torch.zeros(
                    (*tangent_seq.shape[:-1], expected_in - 108),
                    dtype=tangent_seq.dtype,
                    device=tangent_seq.device,
                )
                tangent_seq = torch.cat([rep, pad], dim=-1)
            elif tangent_seq.shape[-1] == 108 and expected_in > 108:
                pad = torch.zeros(
                    (*tangent_seq.shape[:-1], expected_in - 108),
                    dtype=tangent_seq.dtype,
                    device=tangent_seq.device,
                )
                tangent_seq = torch.cat([tangent_seq, pad], dim=-1)

        angles = self.head(self.gru(tangent_seq))
        if self.target_mean is not None and self.target_std is not None:
            t_std = self.target_std.to(device=angles.device, dtype=angles.dtype)
            t_mean = self.target_mean.to(device=angles.device, dtype=angles.dtype)
            angles = angles * t_std + t_mean

        return angles.squeeze(0) if is_2d else angles

    def load_pretrained_weights(
        self,
        weights_path: str | Path,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "Model":
        """Load pretrained checkpoint, translating keys from training or modular format."""
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")

        ckpt: Any = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            raw_dict = ckpt["model_state_dict"]
            if "target_mean" in ckpt and "target_std" in ckpt:
                self.target_mean = torch.as_tensor(ckpt["target_mean"], dtype=torch.float32)
                self.target_std = torch.as_tensor(ckpt["target_std"], dtype=torch.float32)
            if "feature_mean" in ckpt and "feature_std" in ckpt:
                self.feature_mean = torch.as_tensor(ckpt["feature_mean"], dtype=torch.float32)
                self.feature_std = torch.as_tensor(ckpt["feature_std"], dtype=torch.float32)
            self.calibrated_channel_std = ckpt.get("channel_std")
            self.calibrated_inv_sqrt_mean_cov = ckpt.get("inv_sqrt_mean_cov")
            self.intrasubject_calibrated = bool(ckpt.get("intrasubject_calibrated", False))
            self.calibrated_manifold = bool(ckpt.get("calibrated_manifold", False))
            self.intrasubject_bands = ckpt.get("intrasubject_bands")
            self.intrasubject_sub_len = ckpt.get("intrasubject_sub_len")
            self.intrasubject_seq_len = ckpt.get("intrasubject_seq_len")
            self.biomechanical_dip_coupling = bool(ckpt.get("biomechanical_dip_coupling", False))
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            raw_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            raw_dict = ckpt["model"]
        elif isinstance(ckpt, dict):
            raw_dict = ckpt
        else:
            raise ValueError(f"Unexpected checkpoint format loaded from {path}: {type(ckpt)}")

        mapped_dict = {TRAINING_KEY_MAP.get(k, k): v for k, v in raw_dict.items()}
        mapped_dict.setdefault("tangent_space.row", self.tangent_space.row)
        mapped_dict.setdefault("tangent_space.col", self.tangent_space.col)
        mapped_dict.setdefault("tangent_space.weight", self.tangent_space.weight)

        dense_w = mapped_dict.get("gru.dense.0.weight")
        if dense_w is not None and dense_w.shape[1] != self.gru.dense[0].in_features:
            new_in_dim = dense_w.shape[1]
            self.gru = GRUEncoder(in_dim=new_in_dim, dense=256, units=(256, 128), dropout=0.1).to(map_location)
            self.head = nn.Sequential(
                nn.Linear(self.gru.out_dim, self.hidden),
                nn.Tanh(),
                nn.Linear(self.hidden, self.targets),
            ).to(map_location)

        self.load_state_dict(mapped_dict, strict=strict)
        self.eval()
        return self
