"""High-precision 200 Hz mock sEMG streamer."""

from __future__ import annotations

import logging
import pathlib
import queue
import threading
import time
from typing import Callable

import numpy as np

from src.stream.ring_buffer import EMGRingBuffer

logger = logging.getLogger(__name__)


def _set_windows_timer_resolution(period_ms: int = 1) -> bool:
    try:
        import ctypes

        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "winmm"):
            ctypes.windll.winmm.timeBeginPeriod(period_ms)
            return True
    except Exception:
        pass
    return False


def _reset_windows_timer_resolution(period_ms: int = 1) -> bool:
    try:
        import ctypes

        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "winmm"):
            ctypes.windll.winmm.timeEndPeriod(period_ms)
            return True
    except Exception:
        pass
    return False


class MockMyoStreamer:
    """High-precision 200 Hz sEMG streaming worker for replay or synthetic waveforms."""

    DEFAULT_CANDIDATE_PATHS = [
        pathlib.Path(__file__).resolve().parents[2] / "data" / "recordings" / "myo_intrasubject.npz",
        pathlib.Path(__file__).resolve().parents[3]
        / "BRI-TrainingPipeline"
        / "data"
        / "raw"
        / "emg_fk"
        / "subject_01_trial_01.npz",
    ]

    def __init__(
        self,
        data_source: str | pathlib.Path | np.ndarray | None = None,
        destination: (EMGRingBuffer | queue.Queue | Callable[[np.ndarray, float], None] | None) = None,
        sample_rate: float = 200.0,
        packet_mode: bool = False,
        loop_playback: bool = True,
        channels: int = 8,
        seed: int = 42,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.sample_rate = float(sample_rate)
        self.packet_mode = bool(packet_mode)
        self.loop_playback = bool(loop_playback)
        self.channels = int(channels)
        self.seed = int(seed)

        if destination is None:
            self._ring_buffer: EMGRingBuffer | None = EMGRingBuffer(capacity=2000, channels=self.channels)
            self._destination = self._ring_buffer
        elif isinstance(destination, EMGRingBuffer):
            self._ring_buffer = destination
            self._destination = destination
        else:
            self._ring_buffer = None
            self._destination = destination

        self.data_source_type: str = "synthetic"
        self.data_path: str | None = None
        self._data: np.ndarray = self._resolve_data(data_source)
        self._n_samples = self._data.shape[1]

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples_emitted = 0
        self._lock = threading.Lock()

    @property
    def ring_buffer(self) -> EMGRingBuffer:
        if self._ring_buffer is not None:
            return self._ring_buffer
        if isinstance(self._destination, EMGRingBuffer):
            return self._destination
        raise AttributeError("No EMGRingBuffer attached to this MockMyoStreamer.")

    @property
    def buffer(self) -> EMGRingBuffer:
        return self.ring_buffer

    @property
    def samples_emitted(self) -> int:
        with self._lock:
            return self._samples_emitted

    @classmethod
    def generate_synthetic(
        cls,
        seconds: float = 30.0,
        fs: float = 200.0,
        channels: int = 8,
        seed: int = 42,
        noise_std: float = 15.0,
    ) -> np.ndarray:
        """Generate 8-channel synthetic sEMG signals clipped to [-128.0, 127.0]."""
        rng = np.random.default_rng(seed)
        n = int(round(seconds * fs))
        t = np.arange(n) / fs

        envelopes = np.stack(
            [
                0.25
                + 0.75
                * (
                    0.6 * (np.sin(2 * np.pi * (0.20 + 0.03 * ch) * t) ** 2)
                    + 0.4 * (np.sin(2 * np.pi * (0.08 + 0.015 * ch) * t) ** 2)
                )
                for ch in range(channels)
            ]
        )
        raw = rng.standard_normal((channels, n)) * noise_std * envelopes
        burst_mask = rng.random((channels, n)) < 0.02
        raw += rng.standard_normal((channels, n)) * (noise_std * 2.5) * burst_mask
        return np.clip(raw, -128.0, 127.0).astype(np.float32)

    def _resolve_data(self, data_source: str | pathlib.Path | np.ndarray | None) -> np.ndarray:
        if isinstance(data_source, np.ndarray):
            arr = np.asarray(data_source, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Array data_source must be 2D (channels, N), got {arr.shape}")
            if arr.shape[0] != self.channels and arr.shape[1] == self.channels:
                arr = arr.T
            if arr.shape[0] != self.channels:
                raise ValueError(f"Expected {self.channels} channels, got shape {arr.shape}")
            self.data_source_type = "array"
            return arr

        candidate_paths: list[pathlib.Path] = (
            [pathlib.Path(data_source)] if data_source is not None else list(self.DEFAULT_CANDIDATE_PATHS)
        )

        for p in candidate_paths:
            try:
                resolved = p.resolve()
                if resolved.is_file():
                    data = np.load(resolved)
                    if "emg" in data:
                        arr = np.asarray(data["emg"], dtype=np.float32)
                        if arr.shape[0] != self.channels and arr.shape[1] == self.channels:
                            arr = arr.T
                        if arr.shape[0] == self.channels:
                            self.data_source_type = "real"
                            self.data_path = str(resolved)
                            return arr
            except Exception:
                pass

        self.data_source_type = "synthetic"
        self.data_path = None
        return self.generate_synthetic(seconds=60.0, fs=self.sample_rate, channels=self.channels, seed=self.seed)

    def _emit(self, sample: np.ndarray, timestamp: float) -> None:
        if isinstance(self._destination, EMGRingBuffer):
            self._destination.push(sample, timestamp)
        elif isinstance(self._destination, queue.Queue):
            self._destination.put((sample, timestamp))
        elif callable(self._destination):
            self._destination(sample, timestamp)

        with self._lock:
            self._samples_emitted += 1

    def _run_loop(self) -> None:
        _set_windows_timer_resolution(1)
        try:
            t_start = time.perf_counter()
            step = 0
            idx = 0

            if self.packet_mode:
                dt = 2.0 / self.sample_rate
                half_dt = 1.0 / self.sample_rate

                while not self._stop_event.is_set():
                    if not self.loop_playback and idx + 1 >= self._n_samples:
                        break

                    s1 = self._data[:, idx % self._n_samples]
                    s2 = self._data[:, (idx + 1) % self._n_samples]
                    t_now = time.perf_counter()
                    self._emit(s1, t_now)
                    self._emit(s2, t_now + half_dt)

                    idx += 2
                    step += 1

                    target = t_start + step * dt
                    rem = target - time.perf_counter()
                    if rem > 0.0015 and self._stop_event.wait(timeout=rem - 0.001):
                        break
                    while time.perf_counter() < target and not self._stop_event.is_set():
                        pass
            else:
                dt = 1.0 / self.sample_rate

                while not self._stop_event.is_set():
                    if not self.loop_playback and idx >= self._n_samples:
                        break

                    s = self._data[:, idx % self._n_samples]
                    self._emit(s, time.perf_counter())

                    idx += 1
                    step += 1

                    target = t_start + step * dt
                    rem = target - time.perf_counter()
                    if rem > 0.0015 and self._stop_event.wait(timeout=rem - 0.001):
                        break
                    while time.perf_counter() < target and not self._stop_event.is_set():
                        pass
        finally:
            _reset_windows_timer_resolution(1)

    def start(self) -> None:
        if self.is_alive():
            return
        with self._lock:
            self._samples_emitted = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="MockMyoStreamer")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> MockMyoStreamer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
