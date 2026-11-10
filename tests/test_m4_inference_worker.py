"""Unit, regression, and end-to-end integration tests for Milestone 4.

Verifies:
1. StreamingInferenceWorker initialization, buffer allocations, and custom parameters.
2. Synchronous step execution and StepResult dataclass contracts.
3. Rolling raw EMG buffer (60 samples) and persistent SOS filter state continuity.
4. 10-step Riemannian tangent-space covariance FIFO queue maintenance.
5. Latency monitoring statistics (mean, p50, p95, p99, min, max, std, jitter, thresholds < 25 ms).
6. Thread lifecycle: start, stop, join, is_alive, context manager, and clean cleanup.
7. 10 Hz loop execution cadence and drift-compensated timing.
8. End-to-end integration: MockMyoStreamer -> EMGRingBuffer -> StreamingInferenceWorker ->
   UDPSender -> MockUDPReceiver.
9. Worker warmup methods (from array and ring buffer).
10. Reset and reusability.
11. Input validation and error handling.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest
import torch

from src.kinematics.udp_sender import UDPSender
from src.model.model import Model
from src.preprocess import Preprocess
from src.stream.mock_myo import MockMyoStreamer
from src.stream.ring_buffer import EMGRingBuffer
from src.stream.worker import LatencyMonitor, StepResult, StreamingInferenceWorker
from tests.conftest import MockUDPReceiver

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_ROOT / "models" / "best_model.pt"


@pytest.fixture
def loaded_model() -> Model:
    """Return TRR Model loaded with pretrained weights if available."""
    model = Model(channels=8, targets=15, hidden=64)
    if WEIGHTS_PATH.is_file():
        model.load_pretrained_weights(WEIGHTS_PATH, map_location="cpu")
    model.eval()
    return model


@pytest.fixture
def ring_buffer() -> EMGRingBuffer:
    """Return standard 8-channel EMGRingBuffer."""
    return EMGRingBuffer(capacity=2000, channels=8)


# 1. Initialization and Buffer Allocations
def test_worker_initialization_defaults_and_buffers(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify default initialization parameters, rolling buffer shapes, and initial state."""
    worker = StreamingInferenceWorker(
        ring_buffer=ring_buffer,
        model=loaded_model,
    )

    assert worker.channels == 8
    assert worker.sub_window == 60
    assert worker.step_size == 20
    assert worker.queue_len == 10
    assert worker.fs == 200
    assert worker.cadence_hz == 10.0
    assert worker.steps_completed == 0
    assert worker.samples_processed == 0
    assert not worker.is_warmed_up
    assert worker.latest_angles is None
    assert worker.latest_controls is None
    assert not worker.is_alive()

    # Buffer shapes
    raw_buf = worker.raw_buffer
    filt_buf = worker.filtered_buffer
    fifo_seq = worker.tangent_fifo

    assert raw_buf.shape == (8, 60)
    assert filt_buf.shape == (8, 60)
    assert fifo_seq.shape == (1, 10, 36)
    assert np.all(raw_buf == 0.0)
    assert np.all(filt_buf == 0.0)
    assert torch.all(fifo_seq == 0.0)


def test_worker_initialization_custom_overrides(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify custom sub_window, step, queue_len, and cadence parameters."""
    worker = StreamingInferenceWorker(
        ring_buffer=ring_buffer,
        model=loaded_model,
        sub_window=80,
        step=10,
        queue_len=6,
        fs=200,
        cadence_hz=20.0,
    )

    assert worker.sub_window == 80
    assert worker.step_size == 10
    assert worker.queue_len == 6
    assert worker.cadence_hz == 20.0
    assert worker.raw_buffer.shape == (8, 80)
    assert worker.filtered_buffer.shape == (8, 80)
    assert worker.tangent_fifo.shape == (1, 6, 36)


# 2. Synchronous Step Execution and Contracts
def test_worker_step_synchronous_execution(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify single-step synchronous execution produces valid StepResult."""
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)

    chunk = np.random.randn(8, 20).astype(np.float32)
    ts = time.perf_counter() - np.arange(20, 0, -1) * 0.005

    result = worker.step(raw_chunk=chunk, timestamps=ts, transmit=False)

    assert isinstance(result, StepResult)
    assert result.angles.shape == (15,)
    assert result.controls.shape == (20,)
    assert result.tangent_vector.shape == (36,)
    assert 0.0 <= result.compute_latency_ms < 50.0
    assert result.transport_latency_ms >= 0.0

    # Actuator controls bounded in [0.0, 1.0]
    assert np.all(result.controls >= 0.0)
    assert np.all(result.controls <= 1.0)

    # Dictionary-like indexing support
    assert np.array_equal(result["angles"], result.angles)
    assert np.array_equal(result["controls"], result.controls)

    # Progress tracking updated
    assert worker.steps_completed == 1
    assert worker.samples_processed == 20
    assert worker.latest_step_result is not None


# 3. Rolling Raw Buffer & Filter State Continuity
def test_rolling_raw_buffer_and_filter_continuity(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify 60-sample rolling raw buffer shifts correctly and SOS filter states persist."""
    prep = Preprocess(fs=200, bandpass=[15, 95], notch=50, channels=8)
    worker = StreamingInferenceWorker(
        ring_buffer=ring_buffer,
        model=loaded_model,
        preprocessor=prep,
    )

    # Chunk 1: all 1.0
    c1 = np.full((8, 20), 1.0, dtype=np.float32)
    worker.step(raw_chunk=c1, transmit=False)
    buf = worker.raw_buffer
    assert np.all(buf[:, :40] == 0.0)
    assert np.all(buf[:, 40:60] == 1.0)

    # Chunk 2: all 2.0
    c2 = np.full((8, 20), 2.0, dtype=np.float32)
    worker.step(raw_chunk=c2, transmit=False)
    buf = worker.raw_buffer
    assert np.all(buf[:, :20] == 0.0)
    assert np.all(buf[:, 20:40] == 1.0)
    assert np.all(buf[:, 40:60] == 2.0)

    # Chunk 3: all 3.0 (buffer now fully filled)
    c3 = np.full((8, 20), 3.0, dtype=np.float32)
    worker.step(raw_chunk=c3, transmit=False)
    buf = worker.raw_buffer
    assert np.all(buf[:, :20] == 1.0)
    assert np.all(buf[:, 20:40] == 2.0)
    assert np.all(buf[:, 40:60] == 3.0)

    # Chunk 4: all 4.0 (chunk 1 discarded, chunk 4 inserted at tail)
    c4 = np.full((8, 20), 4.0, dtype=np.float32)
    worker.step(raw_chunk=c4, transmit=False)
    buf = worker.raw_buffer
    assert np.all(buf[:, :20] == 2.0)
    assert np.all(buf[:, 20:40] == 3.0)
    assert np.all(buf[:, 40:60] == 4.0)

    # Verify SOS filter states are non-zero (persisted and evolved)
    assert not np.all(prep.zi_notch == 0.0)
    assert not np.all(prep.zi_bandpass == 0.0)


# 4. 10-Step Tangent FIFO Queue Maintenance
def test_tangent_fifo_queue_maintenance_and_warmup(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify 10-step FIFO queue shifts correctly and warmup flag triggers at step 10."""
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)

    tangents = []
    for i in range(12):
        chunk = np.random.randn(8, 20).astype(np.float32) * (i + 1)
        res = worker.step(raw_chunk=chunk, transmit=False)
        tangents.append(res.tangent_vector)
        if i < 9:
            assert not worker.is_warmed_up
        else:
            assert worker.is_warmed_up

    assert worker.steps_completed == 12
    assert worker.samples_processed == 240

    # The latest tangent vector in FIFO must match tangents[-1]
    fifo = worker.tangent_fifo.numpy()
    latest_fifo_vec = fifo[0, -1]
    assert np.allclose(latest_fifo_vec, tangents[-1], atol=1e-5)

    # The second latest tangent vector in FIFO must match tangents[-2]
    prev_fifo_vec = fifo[0, -2]
    assert np.allclose(prev_fifo_vec, tangents[-2], atol=1e-5)


# 5. Latency Monitor Metrics & Thresholds
def test_latency_monitor_metrics_and_thresholds(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify latency monitor statistics accuracy and compliance with sub-25 ms threshold."""
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)

    for _ in range(30):
        chunk = np.random.randn(8, 20).astype(np.float32)
        worker.step(raw_chunk=chunk, transmit=False)

    stats = worker.get_latency_stats()
    assert "compute" in stats
    assert "transport" in stats

    comp = stats["compute"]
    assert comp["count"] == 30.0
    assert comp["mean"] > 0.0
    assert comp["min"] <= comp["mean"] <= comp["max"]
    assert comp["p50"] <= comp["p95"] <= comp["p99"]
    assert "jitter" in comp
    assert comp["mean"] < 20.0, f"Mean forward compute latency {comp['mean']} ms exceeds 20 ms!"

    # Verify worker convenience properties
    assert worker.mean_compute_latency_ms == comp["mean"]
    assert worker.p95_compute_latency_ms == comp["p95"]

    # Test LatencyMonitor reset and standalone usage
    worker.latency_monitor.reset()
    comp_reset = worker.latency_monitor.compute_stats()
    assert comp_reset["count"] == 0.0
    assert comp_reset["mean"] == 0.0

    standalone_mon = LatencyMonitor()
    standalone_mon.record(1.5, 2.5)
    assert standalone_mon.mean_compute_ms == 1.5
    assert standalone_mon.mean_transport_ms == 2.5


# 6. Thread Lifecycle (Start, Stop, Join, Context Manager)
def test_worker_thread_lifecycle(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify thread start, stop, is_alive, and context manager lifecycle."""
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)

    assert not worker.is_alive()
    worker.start()
    assert worker.is_alive()

    # Calling start while already running should log warning and be a no-op
    worker.start()
    assert worker.is_alive()

    worker.stop(timeout=1.0)
    assert not worker.is_alive()

    # Context manager test
    with StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model) as ctx_worker:
        assert ctx_worker.is_alive()
    assert not ctx_worker.is_alive()


# 7. 10 Hz Loop Execution Cadence
def test_10hz_loop_execution_cadence(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify worker thread maintains 10 Hz execution cadence (100 ms interval)."""
    worker = StreamingInferenceWorker(
        ring_buffer=ring_buffer,
        model=loaded_model,
        pace_loop=True,
    )

    # Pre-fill ring buffer with 100 samples (5 steps of 20 samples)
    data = np.random.randn(8, 100).astype(np.float32)
    ring_buffer.push_chunk(data)

    t0 = time.perf_counter()
    worker.start()

    # Wait for 5 steps to complete
    while worker.steps_completed < 5 and (time.perf_counter() - t0) < 2.0:
        time.sleep(0.02)

    elapsed = time.perf_counter() - t0
    worker.stop(timeout=1.0)

    # 5 steps @ 10 Hz should take ~0.4 to 0.6 seconds
    assert worker.steps_completed >= 5
    assert 0.35 <= elapsed <= 0.85, f"5 steps at 10 Hz took {elapsed:.3f} s (expected ~0.50 s)"


# 8. End-to-End Streaming with MockMyoStreamer and MockUDPReceiver
def test_end_to_end_streaming_mock_streamer_and_udp(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify live 200 Hz streamer driving 10 Hz inference worker and sending UDP datagrams."""
    receiver = MockUDPReceiver()
    sender = UDPSender(host="127.0.0.1", port=receiver.port)

    streamer = MockMyoStreamer(
        destination=ring_buffer,
        sample_rate=200.0,
        packet_mode=False,
    )

    worker = StreamingInferenceWorker(
        ring_buffer=ring_buffer,
        model=loaded_model,
        udp_sender=sender,
        pace_loop=True,
    )

    packets_received = []
    arrival_timestamps = []

    try:
        streamer.start()
        worker.start()

        # Listen for UDP packets for ~1.5 seconds (expect ~15 packets @ 10 Hz)
        t_deadline = time.perf_counter() + 1.5
        while time.perf_counter() < t_deadline:
            try:
                pkt = receiver.recv_packet(timeout=0.3)
                packets_received.append(pkt)
                arrival_timestamps.append(time.perf_counter())
            except Exception:
                continue

    finally:
        worker.stop(timeout=1.0)
        streamer.stop(timeout=1.0)
        sender.close()
        receiver.close()

    assert len(packets_received) >= 8, f"Expected >= 8 packets, got {len(packets_received)}"

    # Verify each packet: 80 bytes with 20 floats in [0, 1]
    for pkt in packets_received:
        assert len(pkt) == 80
        ctrls = np.array(receiver.unpack_packet(pkt), dtype=np.float32)
        assert ctrls.shape == (20,)
        assert np.all(ctrls >= 0.0)
        assert np.all(ctrls <= 1.0)
        # Wrist neutral controls
        assert np.isclose(ctrls[0], 0.750, atol=1e-3)
        assert np.isclose(ctrls[1], 0.588, atol=1e-3)

    # Verify packet arrival rate is ~10 Hz (mean interval ~0.10 s)
    if len(arrival_timestamps) >= 5:
        intervals = np.diff(arrival_timestamps)
        mean_interval = float(np.mean(intervals))
        assert 0.07 <= mean_interval <= 0.16, f"Mean packet interval {mean_interval:.3f} s not ~0.10 s"

    # Verify latency stats recorded
    stats = worker.get_latency_stats()
    assert stats["compute"]["count"] >= len(packets_received)
    assert stats["compute"]["mean"] < 20.0
    assert stats["transport"]["mean"] < 25.0


# 9. Warmup Methods
def test_worker_warmup_from_array_and_buffer(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify warmup fills rolling buffer and FIFO queue without sending UDP packets."""
    receiver = MockUDPReceiver()
    sender = UDPSender(host="127.0.0.1", port=receiver.port)

    worker = StreamingInferenceWorker(
        ring_buffer=ring_buffer,
        model=loaded_model,
        udp_sender=sender,
    )

    data = np.random.randn(8, 240).astype(np.float32)
    worker.warmup(initial_emg=data)

    assert worker.is_warmed_up
    assert worker.steps_completed == 12
    assert worker.samples_processed == 240

    # No UDP packets should have been transmitted during warmup
    with pytest.raises(Exception):
        receiver.recv_packet(timeout=0.1)

    sender.close()
    receiver.close()


# 10. Reset and Reusability
def test_worker_reset_and_reusability(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify reset zeroes all buffers and counters, allowing clean restart."""
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)

    chunk = np.random.randn(8, 20).astype(np.float32)
    worker.step(raw_chunk=chunk, transmit=False)
    assert worker.steps_completed == 1
    assert not np.all(worker.raw_buffer == 0.0)

    worker.reset()
    assert worker.steps_completed == 0
    assert worker.samples_processed == 0
    assert not worker.is_warmed_up
    assert np.all(worker.raw_buffer == 0.0)
    assert np.all(worker.filtered_buffer == 0.0)
    assert torch.all(worker.tangent_fifo == 0.0)

    # Step again after reset
    worker.step(raw_chunk=chunk, transmit=False)
    assert worker.steps_completed == 1


# 11. Input Validation and Error Handling
def test_worker_invalid_input_dimensions(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify worker raises ValueError on invalid channel count or 3D input."""
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)

    # Wrong channel count (4 channels instead of 8)
    with pytest.raises(ValueError, match="Channel count mismatch|Sample size"):
        worker.step(raw_chunk=np.random.randn(4, 20))

    # 3D input
    with pytest.raises(ValueError, match="Expected 2D chunk array"):
        worker.step(raw_chunk=np.random.randn(1, 8, 20))


def test_worker_pop_timeout_from_empty_buffer(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify step raises TimeoutError when popping from empty ring buffer."""
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)

    with pytest.raises(TimeoutError):
        worker.step(pop_timeout=0.05)


def test_worker_online_emg_normalization(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify online EMG normalization standardizes raw EMG and prevents saturation."""
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model, pace_loop=False)

    assert worker.norm_enabled is True
    # Feed raw Myo chunks with amplitude ~ 40
    for _ in range(15):
        chunk = np.random.randn(8, 20).astype(np.float32) * 40.0
        res = worker.step(raw_chunk=chunk, transmit=False)

    assert res.angles.shape == (15,)
    assert np.all((res.angles >= 0.0) & (res.angles <= 180.0))
    assert res.controls.shape == (20,)
    assert np.all((res.controls >= 0.0) & (res.controls <= 1.0))


def test_worker_channel_scale_setter(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify explicit channel scale setter and reset functionality."""
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model, pace_loop=False)

    worker.set_channel_scale(25.0)
    assert worker._user_channel_scale is not None
    assert np.allclose(worker._user_channel_scale, 25.0)

    scales = np.array([20.0, 22.0, 25.0, 28.0, 30.0, 32.0, 35.0, 38.0], dtype=np.float32)
    worker.set_channel_scale(scales)
    assert np.allclose(worker._user_channel_scale.flatten(), scales)

    worker.reset_channel_scale()
    assert worker._user_channel_scale is None


def test_guided_calibration_channel_alignment(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify calibrate_guided aligns rotated extensor/flexor pods and per-channel scales."""
    loaded_model.intrasubject_calibrated = False
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model, pace_loop=False)
    rng = np.random.default_rng(123)

    open_raw = rng.standard_normal((8, 600)).astype(np.float32) * 2.0
    open_raw[2] = rng.standard_normal(600).astype(np.float32) * 40.0
    open_raw[3] = rng.standard_normal(600).astype(np.float32) * 25.0

    fist_raw = rng.standard_normal((8, 600)).astype(np.float32) * 2.0
    fist_raw[6] = rng.standard_normal(600).astype(np.float32) * 55.0
    fist_raw[7] = rng.standard_normal(600).astype(np.float32) * 35.0

    summary = worker.calibrate_guided(open_raw, fist_raw)
    assert summary["extensor_pod"] == 3
    assert summary["flexor_pod"] == 7
    assert worker.channel_perm[0] == 2
    assert worker.channel_perm[4] == 6


def test_worker_remote_mujoco_calibration_and_peer_sync(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    """Verify remote UDP calibration trigger from MuJoCo button and peer state sync."""
    import json
    import socket

    worker1 = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model, pace_loop=False)
    rb2 = EMGRingBuffer(capacity=2000, channels=8)
    worker2 = StreamingInferenceWorker(ring_buffer=rb2, model=loaded_model, pace_loop=False)

    worker1._start_control_socket()
    worker2._start_control_socket()
    try:
        assert worker1._ctrl_port is not None
        assert worker2._ctrl_port is not None
        assert worker1._ctrl_port != worker2._ctrl_port

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(
                json.dumps({"cmd": "calibrate"}).encode("utf-8"),
                ("127.0.0.1", worker1._ctrl_port),
            )
        finally:
            s.close()

        chunk = np.random.randn(8, 20).astype(np.float32) * 10.0
        worker1.step(raw_chunk=chunk, transmit=False)
        assert worker1.is_calibrating
        assert worker1.calib_phase == 1

        worker1._calib_open_raw.append(np.random.randn(8, 200).astype(np.float32) * 15.0)
        worker1._calib_fist_raw.append(np.random.randn(8, 200).astype(np.float32) * 35.0)
        worker1._calib_open_angles.append(np.full(15, 175.0, dtype=np.float32))
        worker1._calib_fist_angles.append(np.full(15, 65.0, dtype=np.float32))
        worker1._calib_rms_samples.append(np.full(8, 22.0, dtype=np.float32))
        worker1._calib_start_time = time.perf_counter() - 4.5
        worker1.step(raw_chunk=chunk, transmit=False)

        assert not worker1.is_calibrating
        assert worker1.calib_phase == 3
        assert worker1.is_calibrated

        worker2.step(raw_chunk=chunk, transmit=False)
        assert worker2.is_calibrated
        assert np.array_equal(worker2.channel_perm, worker1.channel_perm)
        assert np.allclose(worker2.channel_std, worker1.channel_std)
    finally:
        worker1._stop_control_socket()
        worker2._stop_control_socket()
