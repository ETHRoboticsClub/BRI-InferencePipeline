"""Unit and integration tests for the model-agnostic StreamingInferenceWorker."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from src.kinematics.udp_sender import UDPSender
from src.model import BaseEMGModel, Model
from src.preprocess import Preprocess
from src.stream.mock_myo import MockMyoStreamer
from src.stream.ring_buffer import EMGRingBuffer
from src.stream.worker import LatencyMonitor, StepResult, StreamingInferenceWorker
from tests.conftest import MockUDPReceiver


@pytest.fixture
def loaded_model() -> Model:
    model = Model(channels=8, targets=15, hidden=64)
    model.eval()
    return model


@pytest.fixture
def ring_buffer() -> EMGRingBuffer:
    return EMGRingBuffer(capacity=2000, channels=8)


def test_worker_initialization_defaults_and_buffers(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)

    assert worker.channels == 8
    assert worker.window_size == 1040
    assert worker.step_size == 20
    assert worker.fs == 200
    assert worker.cadence_hz == 10.0
    assert worker.steps_completed == 0
    assert worker.samples_processed == 0
    assert not worker.is_warmed_up
    assert worker.latest_angles is None
    assert worker.latest_controls is None
    assert not worker.is_alive()

    assert worker.raw_buffer.shape == (8, 1040)
    assert worker.filtered_buffer.shape == (8, 1040)
    assert np.all(worker.raw_buffer == 0.0)
    assert np.all(worker.filtered_buffer == 0.0)


def test_worker_initialization_custom_overrides(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    worker = StreamingInferenceWorker(
        ring_buffer=ring_buffer,
        model=loaded_model,
        window_size=160,
        step=10,
        fs=200,
        cadence_hz=20.0,
    )

    assert worker.window_size == 160
    assert worker.step_size == 10
    assert worker.cadence_hz == 20.0
    assert worker.raw_buffer.shape == (8, 160)
    assert worker.filtered_buffer.shape == (8, 160)


def test_worker_step_synchronous_execution(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)

    chunk = np.random.randn(8, 20).astype(np.float32)
    ts = time.perf_counter() - np.arange(20, 0, -1) * 0.005

    result = worker.step(raw_chunk=chunk, timestamps=ts, transmit=False)

    assert isinstance(result, StepResult)
    assert result.angles.shape == (15,)
    assert result.controls.shape == (20,)
    assert 0.0 <= result.compute_latency_ms < 50.0
    assert result.transport_latency_ms >= 0.0
    assert np.all(result.controls >= 0.0) and np.all(result.controls <= 1.0)
    assert np.array_equal(result["angles"], result.angles)
    assert np.array_equal(result["controls"], result.controls)
    assert worker.steps_completed == 1
    assert worker.samples_processed == 20


def test_worker_accepts_custom_model_subclass(ring_buffer: EMGRingBuffer) -> None:
    """Verify any custom BaseEMGModel subclass works seamlessly inside StreamingInferenceWorker."""

    class ConstantAngleModel(BaseEMGModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], 15), 90.0, device=x.device)

    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=ConstantAngleModel(), pace_loop=False)
    chunk = np.random.randn(8, 20).astype(np.float32)
    res = worker.step(raw_chunk=chunk, transmit=False)
    np.testing.assert_allclose(res.angles, 90.0, atol=1e-4)


def test_rolling_raw_buffer_and_filter_continuity(ring_buffer: EMGRingBuffer) -> None:
    class DummyModel(BaseEMGModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], 15), 120.0, device=x.device)

    prep = Preprocess(fs=200, bandpass=[15, 95], notch=50, channels=8)
    worker = StreamingInferenceWorker(
        ring_buffer=ring_buffer,
        model=DummyModel(window_size=60),
        preprocessor=prep,
        window_size=60,
        step=20,
    )

    for val in (1.0, 2.0, 3.0):
        worker.step(raw_chunk=np.full((8, 20), val, dtype=np.float32), transmit=False)

    buf = worker.raw_buffer
    assert np.all(buf[:, :20] == 1.0)
    assert np.all(buf[:, 20:40] == 2.0)
    assert np.all(buf[:, 40:60] == 3.0)
    assert not np.all(prep.zi_notch == 0.0)
    assert not np.all(prep.zi_bandpass == 0.0)


def test_latency_monitor_metrics_and_thresholds(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)

    for _ in range(30):
        chunk = np.random.randn(8, 20).astype(np.float32)
        worker.step(raw_chunk=chunk, transmit=False)

    stats = worker.get_latency_stats()
    comp = stats["compute"]
    assert comp["count"] == 30.0
    assert comp["mean"] < 20.0
    assert worker.mean_compute_latency_ms == comp["mean"]
    assert worker.p95_compute_latency_ms == comp["p95"]

    standalone_mon = LatencyMonitor()
    standalone_mon.record(1.5, 2.5)
    assert standalone_mon.mean_compute_ms == 1.5
    assert standalone_mon.mean_transport_ms == 2.5


def test_worker_thread_lifecycle(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)
    assert not worker.is_alive()
    worker.start()
    assert worker.is_alive()
    worker.stop(timeout=1.0)
    assert not worker.is_alive()

    with StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model) as ctx_worker:
        assert ctx_worker.is_alive()
    assert not ctx_worker.is_alive()


def test_10hz_loop_execution_cadence(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model, pace_loop=True)
    ring_buffer.push_chunk(np.random.randn(8, 100).astype(np.float32))

    t0 = time.perf_counter()
    worker.start()
    while worker.steps_completed < 5 and (time.perf_counter() - t0) < 2.0:
        time.sleep(0.02)
    elapsed = time.perf_counter() - t0
    worker.stop(timeout=1.0)

    assert worker.steps_completed >= 5
    assert 0.35 <= elapsed <= 0.85


def test_end_to_end_streaming_mock_streamer_and_udp(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    receiver = MockUDPReceiver()
    sender = UDPSender(host="127.0.0.1", port=receiver.port)
    streamer = MockMyoStreamer(destination=ring_buffer, sample_rate=200.0, packet_mode=False)
    worker = StreamingInferenceWorker(
        ring_buffer=ring_buffer,
        model=loaded_model,
        udp_sender=sender,
        pace_loop=True,
    )

    packets_received = []
    try:
        streamer.start()
        worker.start()
        t_deadline = time.perf_counter() + 1.2
        while time.perf_counter() < t_deadline:
            try:
                pkt = receiver.recv_packet(timeout=0.3)
                packets_received.append(pkt)
            except Exception:
                continue
    finally:
        worker.stop(timeout=1.0)
        streamer.stop(timeout=1.0)
        sender.close()
        receiver.close()

    assert len(packets_received) >= 6
    for pkt in packets_received:
        assert len(pkt) == 80
        ctrls = np.array(receiver.unpack_packet(pkt), dtype=np.float32)
        assert ctrls.shape == (20,)
        assert np.all((ctrls >= 0.0) & (ctrls <= 1.0))


def test_worker_warmup_and_reset(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model)
    data = np.random.randn(8, 1040).astype(np.float32)
    worker.warmup(initial_emg=data)
    assert worker.is_warmed_up
    assert worker.steps_completed == 52

    worker.reset()
    assert worker.steps_completed == 0
    assert not worker.is_warmed_up
    assert np.all(worker.raw_buffer == 0.0)


def test_worker_invalid_input_and_channel_calibration(loaded_model: Model, ring_buffer: EMGRingBuffer) -> None:
    worker = StreamingInferenceWorker(ring_buffer=ring_buffer, model=loaded_model, pace_loop=False)

    with pytest.raises(ValueError):
        worker.step(raw_chunk=np.random.randn(4, 20))

    with pytest.raises(TimeoutError):
        worker.step(pop_timeout=0.05)

    rng = np.random.default_rng(123)
    open_raw = rng.standard_normal((8, 400)).astype(np.float32)
    open_raw[2] *= 20.0
    fist_raw = rng.standard_normal((8, 400)).astype(np.float32)
    fist_raw[6] *= 25.0

    summary = worker.calibrate_guided(open_raw, fist_raw)
    assert summary["extensor_pod"] == 3
    assert summary["flexor_pod"] == 7
    assert worker.is_calibrated
