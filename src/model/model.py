from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.model.base import BaseEMGModel
from src.model.kernelized import KernelizedLinear
from src.model.ssm import SSMEncoder
from src.model.tangent_space import STEP, SUB_WINDOW, TangentSpace, n_steps, raw_length


class Model(BaseEMGModel):
    """Tangent-Space + Diagonal State-Space Model (SSM) continuous sEMG decoder.

    Maps a filtered 8-channel sEMG window `(B, C, W)` -> `(B, 15)` finger joint
    angles in degrees `[0, 180]`. Z-scores input using training channel stats
    embedded in the checkpoint (`channel_mean`, `channel_std`).
    """

    handles_normalization: bool = True

    def __init__(
        self,
        channels: int = 8,
        targets: int = 15,
        window_size: int | None = None,
        length: int | None = None,
        activation: str = "relu",
        hidden: int = 64,
        downsample: int = 1,
        ssm_speed: float = 4.0,
        **_kwargs: object,
    ) -> None:
        if length is not None:
            resolved_length = int(length)
            resolved_window = raw_length(resolved_length) if window_size is None else int(window_size)
        elif window_size is not None:
            resolved_window = int(window_size)
            if resolved_window < SUB_WINDOW:
                raise ValueError(f"window_size must be >= {SUB_WINDOW}, got {resolved_window}")
            resolved_length = n_steps(resolved_window)
        else:
            resolved_length = 50
            resolved_window = raw_length(resolved_length)

        super().__init__(channels=channels, targets=targets, window_size=resolved_window)
        self.length = resolved_length
        self.activation_name = str(activation)
        self.hidden = int(hidden)
        self.downsample = int(downsample)
        self.ssm_speed = float(ssm_speed)

        self.tangent_space = TangentSpace(self.channels)
        self._build_layers()

        self.register_buffer("channel_mean", torch.zeros(1, self.channels, 1, dtype=torch.float32))
        self.register_buffer("channel_std", torch.ones(1, self.channels, 1, dtype=torch.float32))

    def _build_layers(self) -> None:
        self.encoder = SSMEncoder(
            self.tangent_space.out_dim,
            self.length,
            self.activation_name,
            downsample=self.downsample,
            speed=self.ssm_speed,
        )
        self.head = KernelizedLinear(self.encoder.out_dim, self.targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        z = (x - self.channel_mean) / torch.clamp(self.channel_std, min=1e-3)
        steps = (z.shape[-1] - SUB_WINDOW) // STEP + 1
        if steps != self.length:
            raise ValueError(f"expected {self.length} tangent steps, got {steps} from {z.shape[-1]} samples")
        out = self.head(self.encoder(self.tangent_space(z)))
        return torch.clamp(out, 0.0, 180.0)

    def load_pretrained_weights(
        self,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        if isinstance(ckpt, dict) and isinstance(ckpt.get("model_config"), dict):
            cfg = ckpt["model_config"]
            rebuilt = False
            for attr, key in [
                ("activation_name", "activation"),
                ("hidden", "hidden"),
                ("downsample", "downsample"),
                ("ssm_speed", "ssm_speed"),
                ("length", "length"),
            ]:
                if key in cfg and getattr(self, attr) != cfg[key]:
                    setattr(self, attr, type(getattr(self, attr))(cfg[key]))
                    rebuilt = True
            if "length" in cfg:
                self.window_size = int(cfg.get("window_size", raw_length(self.length)))
            if rebuilt:
                self._build_layers()

        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        if isinstance(state_dict, dict):
            cleaned = {k.removeprefix("model.").removeprefix("module."): v for k, v in state_dict.items()}
            self.load_state_dict(cleaned, strict=False)

        if isinstance(ckpt, dict):
            if "channel_mean" in ckpt and ckpt["channel_mean"] is not None:
                mean_arr = np.asarray(ckpt["channel_mean"], dtype=np.float32).reshape(1, self.channels, 1)
                self.channel_mean.copy_(torch.from_numpy(mean_arr).to(self.channel_mean.device))
            if "channel_std" in ckpt and ckpt["channel_std"] is not None:
                std_arr = np.asarray(ckpt["channel_std"], dtype=np.float32).reshape(1, self.channels, 1)
                std_arr = np.maximum(std_arr, 1e-3)
                self.channel_std.copy_(torch.from_numpy(std_arr).to(self.channel_std.device))

        return ckpt if isinstance(ckpt, dict) else {}
