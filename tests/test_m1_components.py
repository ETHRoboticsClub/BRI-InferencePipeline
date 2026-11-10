"""Unit and regression tests for core preprocessing and model-agnostic interface."""

import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.model import BaseEMGModel, Model
from src.preprocess import Preprocess

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_params_yaml_structure() -> None:
    """Verify src/params.yaml contains all required fields and valid types."""
    params_path = REPO_ROOT / "src" / "params.yaml"
    assert params_path.is_file(), f"Missing params.yaml at {params_path}"

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    assert "preprocessing" in params
    prep = params["preprocessing"]
    assert prep["fs"] == 200
    assert prep["bandpass"] == [15, 95]
    assert prep["notch"] == 50
    assert prep["channels"] == 8

    assert "windowing" in params
    win = params["windowing"]
    assert win["sub_window"] == 60
    assert win["step"] == 20
    assert win["queue_len"] == 10
    assert win["total_window"] == (win["queue_len"] - 1) * win["step"] + win["sub_window"]

    assert "socket" in params
    sock = params["socket"]
    assert sock["host"] == "127.0.0.1"
    assert sock["port"] == 5005

    assert "kinematics" in params
    kin = params["kinematics"]
    assert kin["rest_angle_deg"] == 180.0
    assert "neutral_wrist" in kin
    assert "flex_ranges" in kin
    assert "tendon_weights" in kin


def test_preprocess_filter_design() -> None:
    """Verify SOS filter shapes and properties."""
    prep = Preprocess(fs=200, bandpass=[15, 95], notch=50, channels=8)
    assert prep.notch.shape == (1, 6)
    assert prep.bandpass.shape == (4, 6)
    assert prep.zi_notch.shape == (1, 8, 2)
    assert prep.zi_bandpass.shape == (4, 8, 2)


def test_preprocess_batch_filtering() -> None:
    """Verify batch filtering on 2D and 3D arrays."""
    prep = Preprocess(fs=200, bandpass=[15, 95], notch=50, channels=8)

    x_2d = np.random.randn(8, 400).astype(np.float32)
    y_2d = prep(x_2d)
    assert y_2d.shape == (8, 400)
    assert y_2d.dtype == np.float32

    x_3d = np.random.randn(3, 8, 400).astype(np.float32)
    y_3d = prep(x_3d)
    assert y_3d.shape == (3, 8, 400)
    assert y_3d.dtype == np.float32


def test_preprocess_stateful_streaming_equivalence() -> None:
    """Verify that causal streaming chunk filtering matches whole-array filtering."""
    prep_batch = Preprocess(fs=200, bandpass=[15, 95], notch=50, channels=8)
    prep_stream = Preprocess(fs=200, bandpass=[15, 95], notch=50, channels=8)

    np.random.seed(42)
    total_samples = 400
    chunk_size = 20
    x = np.random.randn(8, total_samples).astype(np.float64)

    y_batch = prep_batch(x)

    prep_stream.reset_state(channels=8)
    chunks = []
    for i in range(0, total_samples, chunk_size):
        chunk = x[:, i : i + chunk_size]
        filtered_chunk = prep_stream.filter_chunk(chunk)
        chunks.append(filtered_chunk)
    y_stream = np.concatenate(chunks, axis=-1)

    max_diff = np.max(np.abs(y_batch - y_stream))
    assert max_diff < 1e-6, f"Streaming and batch filtering diverged! Max diff: {max_diff}"


def test_preprocess_single_sample_and_reset() -> None:
    """Verify single-sample filtering and state reset."""
    prep = Preprocess(channels=8)
    sample = np.random.randn(8).astype(np.float32)
    filtered = prep.filter_chunk(sample)
    assert filtered.shape == (8,)

    prep.reset_state()
    assert np.all(prep.zi_notch == 0.0)
    assert np.all(prep.zi_bandpass == 0.0)


def test_base_emg_model_custom_subclass() -> None:
    """Verify developers can subclass BaseEMGModel to plug in any custom architecture."""

    class CustomConvModel(BaseEMGModel):
        def __init__(self) -> None:
            super().__init__(channels=8, targets=15, window_size=240)
            self.conv = torch.nn.Conv1d(8, 15, kernel_size=5, padding=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.conv(x).mean(dim=-1)
            return torch.sigmoid(feat) * 180.0

    custom = CustomConvModel()
    win = np.random.randn(8, 240).astype(np.float32)
    angles = custom.predict_window(win)
    assert angles.shape == (15,)
    assert np.all((angles >= 0.0) & (angles <= 180.0))


def test_template_model_forward_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Verify template Model forward pass, predict_window, and checkpoint save/load."""
    model = Model(channels=8, targets=15, hidden=64)

    x_batch = torch.randn(2, 8, 240)
    y_batch = model(x_batch)
    assert y_batch.shape == (2, 15)

    win = np.random.randn(8, 240).astype(np.float32)
    angles = model.predict_window(win)
    assert angles.shape == (15,)
    assert (angles >= 0.0).all() and (angles <= 180.0).all()

    ckpt_path = tmp_path / "test_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    model2 = Model(channels=8, targets=15, hidden=64)
    model2.load_pretrained_weights(ckpt_path, map_location="cpu")
    np.testing.assert_allclose(model.predict_window(win), model2.predict_window(win), atol=1e-5)


def test_model_forward_latency_cpu() -> None:
    """Verify model forward pass latency on CPU is well under 10 ms."""
    model = Model(channels=8, targets=15)
    model.eval()
    win = np.random.randn(8, 240).astype(np.float32)

    for _ in range(5):
        _ = model.predict_window(win)

    n_iters = 50
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = model.predict_window(win)
    mean_latency_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
    assert mean_latency_ms < 10.0, f"Latency {mean_latency_ms:.2f} ms exceeds 10 ms!"
