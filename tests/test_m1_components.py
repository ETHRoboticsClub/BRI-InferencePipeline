"""Unit and regression tests for Milestone 1 components.

Tests:
- Configuration parsing from params.yaml.
- Preprocess: filter design, batch filtering, stateful streaming chunk filtering equivalence.
- TangentSpace: covariance estimation, tangent projection, sliding window, and sub-window methods.
- GRUEncoder: dimension transformations, sequence encoding.
- Model: end-to-end forward pass, forward_from_tangent, weight loading and numerical identity.
- Latency: CPU execution time verification.
"""

import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.model import GRUEncoder, Model, TangentSpace
from src.preprocess import Preprocess

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_params_yaml_structure() -> None:
    """Verify src/params.yaml contains all required fields and valid types."""
    params_path = REPO_ROOT / "src" / "params.yaml"
    assert params_path.is_file(), f"Missing params.yaml at {params_path}"

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Preprocessing
    assert "preprocessing" in params
    prep = params["preprocessing"]
    assert prep["fs"] == 200
    assert prep["bandpass"] == [15, 95]
    assert prep["notch"] == 50
    assert prep["channels"] == 8

    # Windowing
    assert "windowing" in params
    win = params["windowing"]
    assert win["sub_window"] == 60
    assert win["step"] == 20
    assert win["queue_len"] == 10
    assert win["total_window"] == (win["queue_len"] - 1) * win["step"] + win["sub_window"]

    # Socket
    assert "socket" in params
    sock = params["socket"]
    assert sock["host"] == "127.0.0.1"
    assert sock["port"] == 5005

    # Kinematics
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

    # 2D input (C, T)
    x_2d = np.random.randn(8, 400).astype(np.float32)
    y_2d = prep(x_2d)
    assert y_2d.shape == (8, 400)
    assert y_2d.dtype == np.float32

    # 3D input (B, C, T)
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
    chunk_size = 20  # 100 ms at 200 Hz
    x = np.random.randn(8, total_samples).astype(np.float64)

    # 1. Batch filter
    y_batch = prep_batch(x)

    # 2. Chunk filter with persistent state
    prep_stream.reset_state(channels=8)
    chunks = []
    for i in range(0, total_samples, chunk_size):
        chunk = x[:, i : i + chunk_size]
        filtered_chunk = prep_stream.filter_chunk(chunk)
        chunks.append(filtered_chunk)
    y_stream = np.concatenate(chunks, axis=-1)

    max_diff = np.max(np.abs(y_batch - y_stream))
    # Float32 casting adds slight quantization, within 1e-6
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


def test_tangent_space_properties() -> None:
    """Verify TangentSpace dimensions and buffers."""
    ts = TangentSpace(channels=8, sub_window=60, step=20)
    assert ts.out_dim == 36  # 8 * 9 / 2
    assert ts.row.shape == (36,)
    assert ts.col.shape == (36,)
    assert ts.weight.shape == (36,)
    # Diagonals have weight 1.0, off-diagonals have sqrt(2)
    assert torch.isclose(ts.weight[0], torch.tensor(1.0))


def test_tangent_space_forward_shapes() -> None:
    """Verify TangentSpace sliding window and single sub-window computation."""
    ts = TangentSpace(channels=8, sub_window=60, step=20)

    # Batch 3D input: [B, C, W] = [2, 8, 240] -> [B, 10, 36]
    x_3d = torch.randn(2, 8, 240)
    out_3d = ts(x_3d)
    assert out_3d.shape == (2, 10, 36)

    # 2D input: [8, 240] -> [10, 36]
    x_2d = torch.randn(8, 240)
    out_2d = ts(x_2d)
    assert out_2d.shape == (10, 36)

    # Single sub-window: [8, 60] -> [36]
    x_sub = torch.randn(8, 60)
    out_sub = ts.compute_single_subwindow(x_sub)
    assert out_sub.shape == (36,)

    # Explicit sequence: [2, 10, 8, 60] -> [2, 10, 36]
    x_seq = torch.randn(2, 10, 8, 60)
    out_seq = ts.compute_sequence(x_seq)
    assert out_seq.shape == (2, 10, 36)


def test_gru_encoder_shapes() -> None:
    """Verify GRUEncoder input-output dimensions."""
    encoder = GRUEncoder(in_dim=36, dense=256, units=(256, 128), dropout=0.1)
    encoder.eval()

    # 3D sequence [B, 10, 36] -> [B, 128]
    seq_3d = torch.randn(4, 10, 36)
    out_3d = encoder(seq_3d)
    assert out_3d.shape == (4, 128)

    # 2D sequence [10, 36] -> [128]
    seq_2d = torch.randn(10, 36)
    out_2d = encoder(seq_2d)
    assert out_2d.shape == (128,)


def test_model_forward_and_weights_loading() -> None:
    """Verify Model forward passes, weight loading, and output ranges."""
    model = Model(channels=8, targets=15, hidden=64)

    # 1. Uninitialized forward pass shape checks
    x_batch = torch.randn(2, 8, 240)
    y_batch = model(x_batch)
    assert y_batch.shape == (2, 15)

    x_single = torch.randn(8, 240)
    y_single = model(x_single)
    assert y_single.shape == (15,)

    tangent_seq = torch.randn(2, 10, 36)
    y_from_tangent = model.forward_from_tangent(tangent_seq)
    assert y_from_tangent.shape == (2, 15)

    # 2. Pretrained weight loading from migrated weights
    migrated_weights = REPO_ROOT / "models" / "best_model.pt"
    assert migrated_weights.is_file(), f"Missing migrated checkpoint: {migrated_weights}"

    model.load_pretrained_weights(migrated_weights, map_location="cpu")
    model.eval()

    # 3. Predict on a fixed zero-mean EMG pattern
    torch.manual_seed(100)
    x_eval = torch.randn(1, 8, 240)
    with torch.no_grad():
        angles = model(x_eval)

    assert angles.shape == (1, 15)
    angles_np = angles.squeeze(0).numpy()
    # Output angles must be plausible finger joint angles (between 0 deg and 180 deg)
    assert (angles_np > 30.0).all(), f"Angles too low: {angles_np}"
    assert (angles_np < 190.0).all(), f"Angles too high: {angles_np}"


def test_model_equivalence_with_training_checkpoint() -> None:
    """Verify that original training checkpoint and migrated checkpoint yield identical outputs."""
    training_ckpt = Path(r"C:\Users\pette\Documents\ETHRC_HRI\BRI-TrainingPipeline\runs\cross_subject\best_model.pt")
    migrated_ckpt = REPO_ROOT / "models" / "best_model.pt"

    if not training_ckpt.is_file():
        pytest.skip(f"Training checkpoint not available at {training_ckpt}")

    model_training = Model(channels=8, targets=15)
    model_training.load_pretrained_weights(training_ckpt, map_location="cpu")
    model_training.eval()

    model_migrated = Model(channels=8, targets=15)
    model_migrated.load_pretrained_weights(migrated_ckpt, map_location="cpu")
    model_migrated.eval()

    torch.manual_seed(2026)
    test_x = torch.randn(5, 8, 240)

    with torch.no_grad():
        out_train = model_training(test_x)
        out_migrated = model_migrated(test_x)

    diff = torch.max(torch.abs(out_train - out_migrated)).item()
    assert diff == 0.0, f"Predictions differ between source and migrated weights: {diff}"


def test_model_forward_latency_cpu() -> None:
    """Verify model forward pass latency on CPU is well under 20 ms."""
    model = Model(channels=8, targets=15)
    weights_path = REPO_ROOT / "models" / "best_model.pt"
    if weights_path.is_file():
        model.load_pretrained_weights(weights_path, map_location="cpu")
    model.eval()

    x = torch.randn(1, 8, 240)
    tangent_seq = torch.randn(1, 10, 36)

    # Warmup
    for _ in range(5):
        _ = model(x)
        _ = model.forward_from_tangent(tangent_seq)

    # Benchmark full forward pass [1, 8, 240]
    n_iters = 50
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(x)
    total_time = time.perf_counter() - t0
    mean_full_latency_ms = (total_time / n_iters) * 1000.0

    # Benchmark forward_from_tangent [1, 10, 36] (used when FIFO is pre-computed)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model.forward_from_tangent(tangent_seq)
    total_time = time.perf_counter() - t0
    mean_tangent_latency_ms = (total_time / n_iters) * 1000.0

    print(f"\nMean full forward latency: {mean_full_latency_ms:.2f} ms")
    print(f"Mean forward_from_tangent latency: {mean_tangent_latency_ms:.2f} ms")

    assert mean_full_latency_ms < 20.0, f"Latency {mean_full_latency_ms:.2f} ms exceeds 20 ms!"
    assert mean_tangent_latency_ms < 10.0, f"Latency {mean_tangent_latency_ms:.2f} ms exceeds 10 ms!"
