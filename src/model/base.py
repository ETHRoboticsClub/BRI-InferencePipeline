from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class BaseEMGModel(nn.Module):
    """Model-agnostic base interface for continuous sEMG-to-kinematics decoders.

    To add a custom model on a feature branch:
    1. Subclass `BaseEMGModel` in `src/model/model.py` (as `class Model(BaseEMGModel)`).
    2. Implement `forward(self, x: torch.Tensor) -> torch.Tensor` mapping
       `(batch, channels, window_samples)` -> `(batch, 15)` joint angles in degrees `[0, 180]`,
       or override `predict_window(self, window: np.ndarray) -> np.ndarray` for stateful or
       non-PyTorch decoders.
    """

    def __init__(self, channels: int = 8, targets: int = 15, window_size: int = 240) -> None:
        super().__init__()
        self.channels = int(channels)
        self.targets = int(targets)
        self.window_size = int(window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward(x) -> (batch, targets).")

    @torch.no_grad()
    def predict_window(self, window: np.ndarray) -> np.ndarray:
        """Predict 15 finger joint angles in degrees `(15,)` from a filtered sEMG window `(C, W)`."""
        self.eval()
        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        x = torch.from_numpy(arr)
        out = self.forward(x).detach().cpu().numpy()
        return np.clip(np.squeeze(out, axis=0), 0.0, 180.0).astype(np.float32)

    def reset(self) -> None:
        """Reset any internal stateful buffers when streaming restarts."""

    def load_pretrained_weights(
        self,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        if isinstance(state_dict, dict):
            cleaned = {k.removeprefix("model.").removeprefix("module."): v for k, v in state_dict.items()}
            self.load_state_dict(cleaned, strict=False)
        return ckpt if isinstance(ckpt, dict) else {}
