"""Unit and integration tests for Milestone 3: BLE Myo Acquisition and Mock Streaming.

Verifies:
- EMGRingBuffer FIFO ordering, push/pop semantics, get_latest peek, overflow protection,
  underflow handling, and multi-threaded concurrency safety.
- MockMyoStreamer high-precision 200 Hz timing, drift compensation, recorded data replay,
  synthetic waveform fallback, and lifecycle management.
- BleMyoClient GATT profile configuration, 16-byte notification unpacking,
  and automatic fallback to MockMyoStreamer upon discovery timeout or connection error.
"""

from __future__ import annotations

import pathlib
import struct
import threading
import time

import numpy as np
import pytest

from src.stream import (
    CMD_NEVER_SLEEP,
    CMD_NORMAL_SLEEP,
    CMD_SET_EMG_FILTERED,
    CMD_SET_EMG_RAW,
    MYO_EMG_HANDLES,
    MYO_EMG_UUIDS,
    UUID_COMMAND,
    UUID_CONTROL_SVC,
    UUID_EMG0_DATA,
    UUID_EMG1_DATA,
    UUID_EMG2_DATA,
    UUID_EMG3_DATA,
    UUID_EMG_SVC,
    BleMyoClient,
    EMGRingBuffer,
    MockMyoStreamer,
    unpack_emg_notification,
)

# ---------------------------------------------------------------------------
# EMGRingBuffer Unit Tests
# ---------------------------------------------------------------------------


def test_ring_buffer_fifo_ordering() -> None:
    """Verify strictly causal FIFO ordering of samples and monotonic timestamps."""
    buf = EMGRingBuffer(capacity=100, channels=8)
    assert len(buf) == 0
    assert buf.is_empty
    assert not buf.is_full

    n_samples = 25
    for i in range(n_samples):
        sample = np.full(8, float(i), dtype=np.float32)
        buf.push(sample)
        time.sleep(0.0005)

    assert len(buf) == n_samples

    samples, timestamps = buf.pop_chunk(n_samples, timeout=0.1)
    assert samples.shape == (8, n_samples)
    assert timestamps.shape == (n_samples,)

    # Verify FIFO ordering
    for i in range(n_samples):
        np.testing.assert_array_equal(samples[:, i], np.full(8, float(i), dtype=np.float32))

    # Verify monotonic increasing timestamps
    assert np.all(np.diff(timestamps) > 0)
    assert len(buf) == 0
    assert buf.is_empty


def test_ring_buffer_push_pop_shapes() -> None:
    """Verify shapes and data types for single-sample and chunk push/pop operations."""
    buf = EMGRingBuffer(capacity=50, channels=8)

    # Push single sample
    sample = np.arange(8, dtype=np.float32)
    buf.push_sample(sample, timestamp=123.456)
    assert len(buf) == 1

    popped_sample, popped_ts = buf.pop_sample(timeout=0.1)
    assert popped_sample.shape == (8,)
    assert popped_sample.dtype == np.float32
    assert popped_ts == pytest.approx(123.456)
    np.testing.assert_array_equal(popped_sample, sample)


def test_ring_buffer_push_chunk_and_get_latest() -> None:
    """Verify push_chunk with both (channels, N) and (N, channels) shapes and get_latest peek."""
    buf = EMGRingBuffer(capacity=100, channels=8)

    # Push chunk shape (8, 20)
    chunk1 = np.ones((8, 20), dtype=np.float32) * 5.0
    buf.push_chunk(chunk1)
    assert len(buf) == 20

    # Push chunk shape (10, 8) -> should auto-transpose
    chunk2 = np.ones((10, 8), dtype=np.float32) * 9.0
    buf.push_chunk(chunk2)
    assert len(buf) == 30

    # Non-destructive get_latest(10)
    latest10, ts10 = buf.get_latest(10)
    assert latest10.shape == (8, 10)
    assert ts10.shape == (10,)
    np.testing.assert_allclose(latest10, 9.0)
    # Length must remain 30 (peek does not consume)
    assert len(buf) == 30

    # get_latest(30)
    latest30, ts30 = buf.get_latest(30)
    assert latest30.shape == (8, 30)
    np.testing.assert_allclose(latest30[:, :20], 5.0)
    np.testing.assert_allclose(latest30[:, 20:], 9.0)

    # get_latest requesting more than available must raise ValueError
    with pytest.raises(ValueError, match="Requested 35 samples, but buffer only contains 30"):
        buf.get_latest(35)


def test_ring_buffer_overflow_protection() -> None:
    """Verify that buffer wraps around, drops oldest samples, and tracks dropped count."""
    capacity = 10
    buf = EMGRingBuffer(capacity=capacity, channels=8)

    # Push 25 samples into a capacity 10 buffer
    for i in range(25):
        buf.push(np.full(8, float(i), dtype=np.float32))

    assert len(buf) == capacity
    assert buf.is_full
    assert buf.dropped_count == 15

    # Oldest 15 samples (0..14) were dropped; remaining 10 must be 15..24
    latest, _ = buf.get_latest(10)
    for col_idx, expected_val in enumerate(range(15, 25)):
        np.testing.assert_array_equal(latest[:, col_idx], np.full(8, float(expected_val), dtype=np.float32))

    # Pop all 10 remaining samples
    popped, _ = buf.pop_chunk(10, timeout=0.1)
    for col_idx, expected_val in enumerate(range(15, 25)):
        np.testing.assert_array_equal(popped[:, col_idx], np.full(8, float(expected_val), dtype=np.float32))
    assert len(buf) == 0


def test_ring_buffer_underflow_empty_pop() -> None:
    """Verify that popping from an empty or underflowing buffer blocks and raises TimeoutError."""
    buf = EMGRingBuffer(capacity=20, channels=8)

    t0 = time.perf_counter()
    with pytest.raises(TimeoutError, match="Timed out waiting for 5 samples"):
        buf.pop_chunk(5, timeout=0.05)
    elapsed = time.perf_counter() - t0
    assert elapsed >= 0.04

    # Single-sample pop underflow
    with pytest.raises(TimeoutError):
        buf.pop_sample(timeout=0.03)


def test_ring_buffer_invalid_arguments() -> None:
    """Verify validation of invalid parameters and out-of-range requests."""
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        EMGRingBuffer(capacity=0)

    with pytest.raises(ValueError, match="channels must be >= 1"):
        EMGRingBuffer(channels=0)

    buf = EMGRingBuffer(capacity=20, channels=8)

    with pytest.raises(ValueError, match="Sample size mismatch"):
        buf.push(np.zeros(7))

    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        buf.pop_chunk(0)

    with pytest.raises(ValueError, match="exceeds buffer capacity"):
        buf.pop_chunk(25)

    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        buf.get_latest(0)

    with pytest.raises(ValueError, match="exceeds buffer capacity"):
        buf.get_latest(25)


def test_ring_buffer_concurrent_read_write() -> None:
    """Verify thread safety during high-throughput concurrent producer and consumer operations."""
    buf = EMGRingBuffer(capacity=200, channels=8)
    n_total = 400
    chunk_size = 20
    received_samples: list[float] = []
    producer_done = threading.Event()

    def producer() -> None:
        for i in range(n_total):
            buf.push(np.full(8, float(i), dtype=np.float32))
            time.sleep(0.0001)
        producer_done.set()

    def consumer() -> None:
        consumed = 0
        while consumed < n_total:
            remaining = n_total - consumed
            pop_n = min(chunk_size, remaining)
            samples, _ = buf.pop_chunk(pop_n, timeout=1.0)
            for j in range(pop_n):
                received_samples.append(float(samples[0, j]))
            consumed += pop_n

    t_prod = threading.Thread(target=producer, daemon=True)
    t_cons = threading.Thread(target=consumer, daemon=True)

    t_prod.start()
    t_cons.start()

    t_prod.join(timeout=3.0)
    t_cons.join(timeout=3.0)

    assert len(received_samples) == n_total
    # Verify received samples in exact FIFO order without race condition corruption
    expected = [float(i) for i in range(n_total)]
    assert received_samples == expected


# ---------------------------------------------------------------------------
# Notification Unpacking Tests
# ---------------------------------------------------------------------------


def test_notification_unpacking_exact() -> None:
    """Verify unpacking of 16-byte Myo BLE notifications into two 8-channel int8 samples."""
    # Test values including negative, zero, and positive extremes
    s1_vals = [-128, -64, -1, 0, 1, 64, 126, 127]
    s2_vals = [127, 100, 50, 0, -25, -50, -100, -128]

    raw_bytes = struct.pack("<8b8b", *s1_vals, *s2_vals)
    assert len(raw_bytes) == 16

    s1, s2 = unpack_emg_notification(raw_bytes)

    assert s1.shape == (8,)
    assert s2.shape == (8,)
    assert s1.dtype == np.float32
    assert s2.dtype == np.float32

    np.testing.assert_array_equal(s1, np.array(s1_vals, dtype=np.float32))
    np.testing.assert_array_equal(s2, np.array(s2_vals, dtype=np.float32))


def test_notification_unpacking_invalid_length() -> None:
    """Verify unpack_emg_notification rejects non-16-byte inputs."""
    with pytest.raises(ValueError, match="Expected exactly 16 bytes"):
        unpack_emg_notification(bytes(15))

    with pytest.raises(ValueError, match="Expected exactly 16 bytes"):
        unpack_emg_notification(bytes(17))

    with pytest.raises(ValueError, match="Expected exactly 16 bytes"):
        unpack_emg_notification(b"")


# ---------------------------------------------------------------------------
# MockMyoStreamer Unit Tests
# ---------------------------------------------------------------------------


def test_mock_stream_synthetic_amplitude() -> None:
    """Verify synthetic waveform generator outputs valid 8-channel sEMG within int8 bounds."""
    duration = 2.0
    fs = 200.0
    synth = MockMyoStreamer.generate_synthetic(seconds=duration, fs=fs, channels=8, seed=123)

    assert synth.shape == (8, int(duration * fs))
    assert synth.dtype == np.float32
    # Verify values lie within int8 hardware bounds
    assert np.all(synth >= -128.0)
    assert np.all(synth <= 127.0)
    # Verify non-trivial signal variance on all 8 channels
    variances = np.var(synth, axis=1)
    assert np.all(variances > 1.0)


def test_mock_stream_sample_rate_200hz() -> None:
    """Verify MockMyoStreamer emits samples at approximately 200 Hz with drift compensation."""
    target_count = 100  # 100 samples at 200 Hz = 0.50 seconds
    streamer = MockMyoStreamer(sample_rate=200.0, packet_mode=False)

    t0 = time.perf_counter()
    with streamer:
        # Collect target_count samples from the buffer
        samples, timestamps = streamer.ring_buffer.pop_chunk(target_count, timeout=1.5)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    rate = target_count / elapsed

    assert samples.shape == (8, target_count)
    assert timestamps.shape == (target_count,)
    # Verify rate is near 200 Hz (allow ±10 Hz tolerance across test runner loads)
    assert 185.0 <= rate <= 215.0


def test_mock_stream_packet_mode_100hz() -> None:
    """Verify MockMyoStreamer packet mode emits dual samples matching Myo BLE 100 Hz cadence."""
    target_count = 60  # 30 packets = 60 samples
    streamer = MockMyoStreamer(sample_rate=200.0, packet_mode=True)

    with streamer:
        samples, _ = streamer.ring_buffer.pop_chunk(target_count, timeout=1.5)

    assert samples.shape == (8, target_count)


def test_mock_stream_real_data_replay() -> None:
    """Verify MockMyoStreamer replays authentic data from subject_01_trial_01.npz if present."""
    data_path = pathlib.Path(
        r"C:\Users\pette\Documents\ETHRC_HRI\BRI-TrainingPipeline\data\raw\emg_fk\subject_01_trial_01.npz"
    )

    if not data_path.exists():
        pytest.skip("Recorded test dataset subject_01_trial_01.npz not present.")

    raw_npz = np.load(data_path)
    expected_first_10 = raw_npz["emg"][:, :10].astype(np.float32)

    streamer = MockMyoStreamer(data_source=data_path, sample_rate=200.0)
    assert streamer.data_source_type == "real"
    assert streamer.data_path is not None

    with streamer:
        samples, _ = streamer.ring_buffer.pop_chunk(10, timeout=1.0)

    np.testing.assert_allclose(samples, expected_first_10, atol=1e-5)


def test_mock_stream_stop_event_responsiveness() -> None:
    """Verify MockMyoStreamer terminates promptly when stop() is called."""
    streamer = MockMyoStreamer(sample_rate=200.0)
    streamer.start()
    assert streamer.is_alive()

    time.sleep(0.05)
    t0 = time.perf_counter()
    streamer.stop(timeout=1.0)
    elapsed = time.perf_counter() - t0

    assert not streamer.is_alive()
    # Stopping should be responsive (< 0.25 s)
    assert elapsed < 0.25


def test_mock_stream_context_manager() -> None:
    """Verify MockMyoStreamer context manager starts and stops background worker."""
    streamer = MockMyoStreamer(sample_rate=200.0)
    assert not streamer.is_alive()

    with streamer:
        assert streamer.is_alive()

    assert not streamer.is_alive()


def test_mock_stream_invalid_params() -> None:
    """Verify invalid sample_rate or channels raise ValueError."""
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        MockMyoStreamer(sample_rate=0.0)

    with pytest.raises(ValueError, match="channels must be positive"):
        MockMyoStreamer(channels=-1)


# ---------------------------------------------------------------------------
# BleMyoClient Unit & Integration Tests
# ---------------------------------------------------------------------------


def test_ble_myo_constants_and_uuids() -> None:
    """Verify GATT UUIDs, Command bytes, and Handle specifications match Myo protocol."""
    assert UUID_CONTROL_SVC == "d5060001-a904-deb9-4748-2c7f4a124842"
    assert UUID_COMMAND == "d5060401-a904-deb9-4748-2c7f4a124842"
    assert UUID_EMG_SVC == "d5060005-a904-deb9-4748-2c7f4a124842"

    assert len(MYO_EMG_UUIDS) == 4
    assert UUID_EMG0_DATA in MYO_EMG_UUIDS
    assert UUID_EMG1_DATA in MYO_EMG_UUIDS
    assert UUID_EMG2_DATA in MYO_EMG_UUIDS
    assert UUID_EMG3_DATA in MYO_EMG_UUIDS

    assert MYO_EMG_HANDLES == (42, 45, 48, 51)
    assert bytes(CMD_SET_EMG_RAW) == bytes([0x01, 0x03, 0x03, 0x00, 0x00])
    assert bytes(CMD_SET_EMG_FILTERED) == bytes([0x01, 0x03, 0x02, 0x00, 0x00])
    assert bytes(CMD_NEVER_SLEEP) == bytes([0x09, 0x01, 0x01])
    assert bytes(CMD_NORMAL_SLEEP) == bytes([0x09, 0x01, 0x00])


def test_ble_myo_force_mock() -> None:
    """Verify BleMyoClient with force_mock=True skips BLE and runs MockMyoStreamer."""
    client = BleMyoClient(force_mock=True)
    assert client.is_mock
    assert not client.is_connected

    with client:
        assert client.is_alive()
        samples, _ = client.ring_buffer.pop_chunk(20, timeout=1.0)
        assert samples.shape == (8, 20)

    assert not client.is_alive()


def test_ble_myo_mock_fallback_on_timeout() -> None:
    """Verify BleMyoClient falls back to mock when no Myo is found within discovery_timeout."""
    # Fast discovery timeout (0.15s) when no physical Myo is present
    client = BleMyoClient(discovery_timeout=0.15, force_mock=False, mock_fallback=True)

    client.start(wait_ready=True, timeout=2.0)
    try:
        assert client.is_mock
        assert client.is_alive()

        # Verify samples are streaming into buffer
        samples, _ = client.ring_buffer.pop_chunk(20, timeout=1.0)
        assert samples.shape == (8, 20)
    finally:
        client.stop()

    assert not client.is_alive()


def test_ble_myo_notification_handler_dispatch() -> None:
    """Verify _notification_handler directly pushes 2 consecutive samples into ring buffer."""
    client = BleMyoClient(force_mock=True)

    s1_vals = [10, 20, 30, 40, 50, 60, 70, 80]
    s2_vals = [-10, -20, -30, -40, -50, -60, -70, -80]
    payload = struct.pack("<8b8b", *s1_vals, *s2_vals)

    client._notification_handler(None, payload)
    assert len(client.ring_buffer) == 2

    samples, _ = client.ring_buffer.pop_chunk(2, timeout=0.1)
    np.testing.assert_array_equal(samples[:, 0], np.array(s1_vals, dtype=np.float32))
    np.testing.assert_array_equal(samples[:, 1], np.array(s2_vals, dtype=np.float32))


def test_ble_myo_continuous_search_no_mock_fallback() -> None:
    """Verify BleMyoClient with mock_fallback=False does NOT fall back to mock signals."""
    client = BleMyoClient(discovery_timeout=0.15, force_mock=False, mock_fallback=False)
    client.start(wait_ready=False)
    try:
        # Give discovery timeout time to elapse
        time.sleep(0.35)
        # Verify it has NOT fallen back to mock streamer
        assert not client.is_mock
        # Thread should still be alive and actively searching
        assert client.is_alive()
        assert not client.is_connected
        # Ring buffer should be empty (no mock/random signals injected)
        assert len(client.ring_buffer) == 0
    finally:
        client.stop()

    assert not client.is_alive()


def test_ble_myo_shared_connection_primary_and_follower() -> None:
    """Verify two BleMyoClient instances share a single Myo stream via the local TCP hub."""
    test_port = 15188
    primary = BleMyoClient(
        discovery_timeout=0.2,
        force_mock=False,
        mock_fallback=False,
        share_connection=True,
        share_port=test_port,
    )
    follower = BleMyoClient(
        discovery_timeout=0.2,
        force_mock=False,
        mock_fallback=False,
        share_connection=True,
        share_port=test_port,
    )

    try:
        primary.start(wait_ready=False)
        # Wait for primary to bind the local hub server
        deadline = time.perf_counter() + 1.5
        while not primary.is_hub_primary and time.perf_counter() < deadline:
            time.sleep(0.02)
        assert primary.is_hub_primary

        follower.start(wait_ready=True, timeout=1.5)
        assert follower.is_shared_follower

        # Simulate primary connecting to physical Myo and receiving 16-byte notification
        primary._connected = True
        primary._broadcast_status()

        s1_vals = [12, 24, 36, 48, 60, 72, 84, 96]
        s2_vals = [-12, -24, -36, -48, -60, -72, -84, -96]
        payload = struct.pack("<8b8b", *s1_vals, *s2_vals)
        primary._notification_handler(None, payload)

        # Follower must receive the exact 2 samples in its own ring buffer
        f_samples, _ = follower.ring_buffer.pop_chunk(2, timeout=1.0)
        assert follower.is_connected
        np.testing.assert_array_equal(f_samples[:, 0], np.array(s1_vals, dtype=np.float32))
        np.testing.assert_array_equal(f_samples[:, 1], np.array(s2_vals, dtype=np.float32))
    finally:
        follower.stop()
        primary.stop()
