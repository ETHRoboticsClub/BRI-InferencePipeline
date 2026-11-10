"""Thread-safe FIFO circular ring buffer for multi-channel sEMG acquisition."""

from __future__ import annotations

import logging
import threading
import time
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class EMGRingBuffer:
    """Thread-safe bounded FIFO circular ring buffer for multi-channel sEMG samples."""

    def __init__(self, capacity: int = 2000, channels: int = 8) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        self._capacity = int(capacity)
        self._channels = int(channels)
        self._data = np.zeros((self._channels, self._capacity), dtype=np.float32)
        self._timestamps = np.zeros(self._capacity, dtype=np.float64)

        self._head = 0
        self._tail = 0
        self._count = 0
        self._dropped_count = 0

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._count >= self._capacity

    def __len__(self) -> int:
        with self._lock:
            return self._count

    def clear(self) -> None:
        """Clear all samples from the buffer and reset pointers."""
        with self._lock:
            self._head = 0
            self._tail = 0
            self._count = 0
            self._dropped_count = 0
            self._not_full.notify_all()

    def push_sample(self, sample: np.ndarray | Sequence[float], timestamp: float | None = None) -> None:
        self.push(sample, timestamp)

    def push(self, sample: np.ndarray | Sequence[float], timestamp: float | None = None) -> None:
        """Push a single multi-channel sample into the ring buffer."""
        arr = np.asarray(sample, dtype=np.float32)
        if arr.size != self._channels:
            raise ValueError(f"Sample size mismatch: expected {self._channels} elements, got {arr.size}")
        arr = arr.reshape(self._channels)
        ts = time.perf_counter() if timestamp is None else float(timestamp)

        with self._lock:
            if self._count == self._capacity:
                self._tail = (self._tail + 1) % self._capacity
                self._dropped_count += 1
                logger.warning(
                    "EMGRingBuffer overflow: buffer full (%d/%d), dropped oldest sample.",
                    self._count,
                    self._capacity,
                )
            else:
                self._count += 1

            self._data[:, self._head] = arr
            self._timestamps[self._head] = ts
            self._head = (self._head + 1) % self._capacity
            self._not_empty.notify()

    def push_chunk(
        self,
        samples: np.ndarray,
        timestamps: np.ndarray | Sequence[float] | None = None,
    ) -> None:
        """Push multiple multi-channel samples into the ring buffer."""
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for push_chunk, got shape {arr.shape}")

        if arr.shape[0] == self._channels:
            n_samples = arr.shape[1]
        elif arr.shape[1] == self._channels:
            arr = arr.T
            n_samples = arr.shape[1]
        else:
            raise ValueError(f"Sample dimensions {arr.shape} do not match channel count {self._channels}")

        if n_samples == 0:
            return

        if timestamps is None:
            t_now = time.perf_counter()
            ts = t_now - (n_samples - 1 - np.arange(n_samples, dtype=np.float64)) * 0.005
        else:
            ts = np.asarray(timestamps, dtype=np.float64)
            if ts.size != n_samples:
                raise ValueError(f"Timestamps length {ts.size} does not match sample count {n_samples}")
            ts = ts.reshape(n_samples)

        with self._lock:
            if n_samples > self._capacity:
                dropped_direct = n_samples - self._capacity
                self._dropped_count += dropped_direct + self._count
                logger.warning(
                    "EMGRingBuffer overflow: chunk size %d exceeds capacity %d.",
                    n_samples,
                    self._capacity,
                )
                arr = arr[:, -self._capacity :]
                ts = ts[-self._capacity :]
                n_samples = self._capacity
                self._head = 0
                self._tail = 0
                self._count = self._capacity
                self._data[:, :n_samples] = arr
                self._timestamps[:n_samples] = ts
                self._head = (self._head + n_samples) % self._capacity
                self._not_empty.notify_all()
                return

            overflow = (self._count + n_samples) - self._capacity
            if overflow > 0:
                self._tail = (self._tail + overflow) % self._capacity
                self._dropped_count += overflow
                self._count = self._capacity
                logger.warning(
                    "EMGRingBuffer overflow: incoming %d samples caused %d dropped.",
                    n_samples,
                    overflow,
                )
            else:
                self._count += n_samples

            part1 = min(n_samples, self._capacity - self._head)
            part2 = n_samples - part1

            self._data[:, self._head : self._head + part1] = arr[:, :part1]
            self._timestamps[self._head : self._head + part1] = ts[:part1]

            if part2 > 0:
                self._data[:, :part2] = arr[:, part1:]
                self._timestamps[:part2] = ts[part1:]

            self._head = (self._head + n_samples) % self._capacity
            self._not_empty.notify_all()

    def pop_chunk(self, n_samples: int, timeout: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Pop the oldest N samples from the ring buffer in FIFO order."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if n_samples > self._capacity:
            raise ValueError(f"n_samples {n_samples} exceeds buffer capacity {self._capacity}")

        deadline = time.perf_counter() + timeout

        with self._lock:
            while self._count < n_samples:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for {n_samples} samples; only {self._count} available.")
                self._not_empty.wait(timeout=remaining)

            out_samples = np.empty((self._channels, n_samples), dtype=np.float32)
            out_timestamps = np.empty(n_samples, dtype=np.float64)

            part1 = min(n_samples, self._capacity - self._tail)
            part2 = n_samples - part1

            out_samples[:, :part1] = self._data[:, self._tail : self._tail + part1]
            out_timestamps[:part1] = self._timestamps[self._tail : self._tail + part1]

            if part2 > 0:
                out_samples[:, part1:] = self._data[:, :part2]
                out_timestamps[part1:] = self._timestamps[:part2]

            self._tail = (self._tail + n_samples) % self._capacity
            self._count -= n_samples
            self._not_full.notify_all()
            return out_samples, out_timestamps

    def pop_sample(self, timeout: float = 0.5) -> tuple[np.ndarray, float]:
        """Pop a single oldest sample from the ring buffer."""
        samples, timestamps = self.pop_chunk(1, timeout=timeout)
        return samples[:, 0], float(timestamps[0])

    def get_latest(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the most recent N samples without removing them from the buffer."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if n_samples > self._capacity:
            raise ValueError(f"n_samples {n_samples} exceeds buffer capacity {self._capacity}")

        with self._lock:
            if self._count < n_samples:
                raise ValueError(f"Requested {n_samples} samples, but buffer only contains {self._count}")

            start_idx = (self._head - n_samples) % self._capacity
            out_samples = np.empty((self._channels, n_samples), dtype=np.float32)
            out_timestamps = np.empty(n_samples, dtype=np.float64)

            part1 = min(n_samples, self._capacity - start_idx)
            part2 = n_samples - part1

            out_samples[:, :part1] = self._data[:, start_idx : start_idx + part1]
            out_timestamps[:part1] = self._timestamps[start_idx : start_idx + part1]

            if part2 > 0:
                out_samples[:, part1:] = self._data[:, :part2]
                out_timestamps[part1:] = self._timestamps[:part2]

            return out_samples, out_timestamps
