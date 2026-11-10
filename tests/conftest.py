"""Pytest fixtures and test utilities for BRI-InferencePipeline E2E Test Suite.

Provides:
- Project paths and configuration fixtures
- High-fidelity synthetic sEMG and tangent sequence generators
- Frequency-specific sinusoid generators for filter verification
- Canonical joint angle profiles (extended, flexed, neutral, isolated digits)
- Mock UDP receiver utility for capturing and verifying binary datagrams
"""

import pathlib
import socket
import struct
from typing import Generator

import numpy as np
import pytest
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAINING_PIPELINE_ROOT = PROJECT_ROOT.parent / "BRI-TrainingPipeline"


@pytest.fixture(scope="session")
def project_root() -> pathlib.Path:
    """Return the absolute path to the BRI-InferencePipeline repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def training_pipeline_root() -> pathlib.Path:
    """Return the absolute path to the reference BRI-TrainingPipeline root."""
    return TRAINING_PIPELINE_ROOT


@pytest.fixture(scope="session")
def default_config_path() -> pathlib.Path:
    """Return the path to src/params.yaml."""
    return PROJECT_ROOT / "src" / "params.yaml"


@pytest.fixture(scope="session")
def sample_config_dict() -> dict:
    """Return a canonical configuration dictionary matching requirements R1-R5."""
    return {
        "preprocessing": {
            "fs": 200,
            "bandpass": [15, 95],
            "notch": 50,
            "channels": 8,
        },
        "windowing": {
            "sub_window": 60,
            "step": 20,
            "queue_len": 10,
        },
        "socket": {
            "host": "127.0.0.1",
            "port": 5005,
        },
        "joint_mapping": {
            "neutral_wrist": [0.750, 0.588],
            "neutral_abduction": 0.500,
            "neutral_mcp": 0.143,
            "neutral_tendon": 0.000,
        },
    }


@pytest.fixture
def temp_config_file(tmp_path: pathlib.Path, sample_config_dict: dict) -> pathlib.Path:
    """Write sample_config_dict into a temporary params.yaml file."""
    cfg_file = tmp_path / "params.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(sample_config_dict, f, default_flow_style=False)
    return cfg_file


@pytest.fixture(scope="session")
def synthetic_emg_generator():
    """Factory fixture generating deterministic synthetic 8-channel sEMG arrays."""

    def _generator(
        channels: int = 8,
        samples: int = 240,
        fs: float = 200.0,
        noise_std: float = 15.0,
        seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t = np.arange(samples) / fs
        # Realistic amplitude envelope modulation across channels
        envelopes = np.stack([0.3 + 0.7 * (np.sin(2 * np.pi * (0.2 + 0.04 * ch) * t) ** 2) for ch in range(channels)])
        raw = rng.standard_normal((channels, samples)) * noise_std * envelopes
        return np.clip(raw, -128.0, 127.0).astype(np.float32)

    return _generator


@pytest.fixture(scope="session")
def sinusoid_generator():
    """Factory fixture generating pure sinusoids to evaluate digital filter response."""

    def _generator(
        freq: float,
        fs: float = 200.0,
        duration: float = 1.0,
        amplitude: float = 10.0,
        channels: int = 8,
    ) -> np.ndarray:
        t = np.arange(int(duration * fs)) / fs
        wave = amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)
        return np.tile(wave, (channels, 1))

    return _generator


@pytest.fixture(scope="session")
def canonical_hand_angles() -> dict[str, np.ndarray]:
    """Return dictionary of canonical 15-DOF finger angle postures in degrees."""
    # 15 DOFs: Thumb(CMC, MCP, IP), Index(MCP, PIP, DIP), Middle(MCP, PIP, DIP),
    # Ring(MCP, PIP, DIP), Pinky(MCP, PIP, DIP)
    open_hand = np.full(15, 180.0, dtype=np.float32)

    closed_fist = np.array(
        [
            120.0,
            100.0,
            100.0,  # Thumb (CMC, MCP, IP flexed)
            90.0,
            70.0,
            90.0,  # Index
            90.0,
            70.0,
            90.0,  # Middle
            90.0,
            70.0,
            90.0,  # Ring
            90.0,
            70.0,
            90.0,  # Pinky
        ],
        dtype=np.float32,
    )

    half_flexed = np.full(15, 135.0, dtype=np.float32)

    index_pointing = np.array(
        [
            120.0,
            100.0,
            100.0,  # Thumb flexed
            180.0,
            180.0,
            180.0,  # Index fully extended
            90.0,
            70.0,
            90.0,  # Middle flexed
            90.0,
            70.0,
            90.0,  # Ring flexed
            90.0,
            70.0,
            90.0,  # Pinky flexed
        ],
        dtype=np.float32,
    )

    return {
        "open_hand": open_hand,
        "closed_fist": closed_fist,
        "half_flexed": half_flexed,
        "index_pointing": index_pointing,
    }


@pytest.fixture(scope="session")
def expected_open_hand_controls() -> np.ndarray:
    """Expected 20 normalized Shadow Hand controls [0.0, 1.0] for fully open hand (180 deg)."""
    ctrl = np.zeros(20, dtype=np.float32)
    # Wrist neutral: WRJ2=0.750, WRJ1=0.588
    ctrl[0] = 0.750
    ctrl[1] = 0.588
    # Thumb: THJ5=0.500, THJ4=0.000, THJ3=0.500, THJ2=0.500, THJ1=0.143
    ctrl[2] = 0.500
    ctrl[3] = 0.000
    ctrl[4] = 0.500
    ctrl[5] = 0.500
    ctrl[6] = 0.143
    # Index: FFJ4=0.500, FFJ3=0.143, FFJ0=0.000
    ctrl[7] = 0.500
    ctrl[8] = 0.143
    ctrl[9] = 0.000
    # Middle: MFJ4=0.500, MFJ3=0.143, MFJ0=0.000
    ctrl[10] = 0.500
    ctrl[11] = 0.143
    ctrl[12] = 0.000
    # Ring: RFJ4=0.500, RFJ3=0.143, RFJ0=0.000
    ctrl[13] = 0.500
    ctrl[14] = 0.143
    ctrl[15] = 0.000
    # Pinky: LFJ5=0.000, LFJ4=0.500, LFJ3=0.143, LFJ0=0.000
    ctrl[16] = 0.000
    ctrl[17] = 0.500
    ctrl[18] = 0.143
    ctrl[19] = 0.000
    return ctrl


class MockUDPReceiver:
    """Helper class to receive UDP datagrams in automated tests without port conflicts."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow immediate reuse of address/port
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.settimeout(1.5)
        self.port = self.sock.getsockname()[1]
        self.host = host

    def recv_packet(self, timeout: float = 1.0) -> bytes:
        self.sock.settimeout(timeout)
        data, _ = self.sock.recvfrom(256)
        return data

    def unpack_packet(self, data: bytes) -> tuple[float, ...]:
        assert len(data) == 80, f"Expected 80 bytes for 20f payload, got {len(data)}"
        return struct.unpack("<20f", data)

    def close(self):
        self.sock.close()


@pytest.fixture
def mock_udp_receiver() -> Generator[MockUDPReceiver, None, None]:
    """Yield a MockUDPReceiver bound to an ephemeral loopback port and clean up after."""
    receiver = MockUDPReceiver()
    yield receiver
    receiver.close()
