"""UDP datagram transmitter for MuJoCo Shadow Hand controls."""

from __future__ import annotations

import socket
import struct
from typing import Any

import numpy as np

from src.kinematics.mapping import map_angles_to_controls


class UDPSender:
    """Non-blocking UDP socket client transmitting 20-float actuator control packets."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5005) -> None:
        self.host = host
        self.port = port
        self.sock: socket.socket | None = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def send(self, controls: np.ndarray | list[float] | tuple[float, ...]) -> int:
        """Pack 20 control floats into 80 bytes little-endian and transmit over UDP."""
        if self.sock is None:
            raise RuntimeError("UDPSender socket is closed")

        if hasattr(controls, "__len__") and len(controls) != 20:
            raise ValueError(f"Expected exactly 20 control values, got {len(controls)}")

        payload = struct.pack("<20f", *controls)
        try:
            return self.sock.sendto(payload, (self.host, self.port))
        except BlockingIOError:
            return 0

    def send_angles(
        self,
        angles_deg: np.ndarray | list[float] | tuple[float, ...],
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Translate 15 joint angles to 20 controls and transmit in one step."""
        controls = map_angles_to_controls(angles_deg, params=params)
        self.send(controls)
        return controls

    def close(self) -> None:
        """Close and release the underlying UDP socket."""
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def __enter__(self) -> "UDPSender":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
