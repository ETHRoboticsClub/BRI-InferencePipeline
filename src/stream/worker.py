"""Real-time model-agnostic streaming inference worker for continuous sEMG decoding."""

from __future__ import annotations

import collections
import dataclasses
import logging
import pathlib
import socket
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml

from src.kinematics.mapping import map_angles_to_controls
from src.kinematics.udp_sender import UDPSender
from src.model.model import Model
from src.preprocess import Preprocess
from src.stream.ring_buffer import EMGRingBuffer

logger = logging.getLogger(__name__)

DEFAULT_PARAMS_PATH = pathlib.Path(__file__).resolve().parent.parent / "params.yaml"


def _load_default_params(path: pathlib.Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    return {
        "preprocessing": {"fs": 200, "bandpass": [15, 95], "notch": 50, "channels": 8},
        "windowing": {"window_size": 1040, "sub_window": 60, "step": 20, "queue_len": 50},
        "socket": {"host": "127.0.0.1", "port": 5005},
        "model": {
            "channels": 8,
            "targets": 15,
            "length": 50,
            "activation": "relu",
            "hidden": 64,
            "downsample": 1,
            "ssm_speed": 4,
        },
    }


@dataclasses.dataclass(frozen=True)
class StepResult:
    """Telemetry snapshot captured at the completion of a single inference tick."""

    step_index: int
    angles: np.ndarray
    controls: np.ndarray
    compute_latency_ms: float
    transport_latency_ms: float
    timestamp: float

    @property
    def angles_deg(self) -> np.ndarray:
        return self.angles

    @property
    def actuators_ctrl(self) -> np.ndarray:
        return self.controls

    @property
    def timestamp_s(self) -> float:
        return self.timestamp

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


StepMetrics = StepResult


class LatencyMonitor:
    """Thread-safe rolling latency tracker for compute and transport metrics."""

    def __init__(
        self,
        maxlen: int = 1000,
        compute_budget_ms: float = 10.0,
        transport_budget_ms: float = 20.0,
    ) -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen must be > 0, got {maxlen}")
        self.maxlen = int(maxlen)
        self.compute_budget_ms = float(compute_budget_ms)
        self.transport_budget_ms = float(transport_budget_ms)

        self._compute_history: collections.deque[float] = collections.deque(maxlen=self.maxlen)
        self._transport_history: collections.deque[float] = collections.deque(maxlen=self.maxlen)
        self._lock = threading.Lock()

    def record(self, compute_ms: float, transport_ms: float) -> None:
        with self._lock:
            self._compute_history.append(float(compute_ms))
            self._transport_history.append(float(transport_ms))

    def reset(self) -> None:
        with self._lock:
            self._compute_history.clear()
            self._transport_history.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._compute_history)

    @property
    def mean_compute_ms(self) -> float:
        return self.compute_stats()["mean"]

    @property
    def mean_transport_ms(self) -> float:
        return self.transport_stats()["mean"]

    @staticmethod
    def _calc_stats(samples: Sequence[float], budget_ms: float) -> dict[str, float]:
        if not samples:
            return {
                "count": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "jitter": 0.0,
                "budget_ms": budget_ms,
                "within_budget": 1.0,
            }
        arr = np.asarray(samples, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        jitter_val = float(np.mean(np.abs(np.diff(arr)))) if arr.size > 1 else 0.0
        return {
            "count": float(arr.size),
            "mean": mean_val,
            "std": std_val,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "jitter": jitter_val,
            "budget_ms": budget_ms,
            "within_budget": 1.0 if mean_val < budget_ms else 0.0,
        }

    def compute_stats(self) -> dict[str, float]:
        with self._lock:
            comp_list = list(self._compute_history)
        return self._calc_stats(comp_list, self.compute_budget_ms)

    def transport_stats(self) -> dict[str, float]:
        with self._lock:
            trans_list = list(self._transport_history)
        return self._calc_stats(trans_list, self.transport_budget_ms)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            "compute": self.compute_stats(),
            "transport": self.transport_stats(),
        }


class StreamingInferenceWorker:
    """Real-time model-agnostic inference worker running at a fixed cadence (default 10 Hz).

    Pulls sEMG chunks from `EMGRingBuffer`, applies causal filtering via `Preprocess`,
    maintains a rolling window `(channels, window_size)`, invokes `model.predict_window(window)`
    (or `model(tensor)`), maps the 15 predicted joint angles to 20 Shadow Hand actuators,
    and transmits them via `UDPSender`.
    """

    def __init__(
        self,
        ring_buffer: EMGRingBuffer,
        model: torch.nn.Module | None = None,
        preprocessor: Preprocess | None = None,
        udp_sender: UDPSender | None = None,
        params: dict[str, Any] | None = None,
        window_size: int | None = None,
        sub_window: int | None = None,
        step: int | None = None,
        queue_len: int | None = None,
        fs: int | None = None,
        cadence_hz: float | None = None,
        pace_loop: bool = True,
        pop_timeout: float = 0.25,
        step_callback: Callable[[StepResult], None] | None = None,
        device: str | torch.device = "cpu",
        normalize_emg: bool | None = None,
        channel_scale: float | Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self.params = params if params is not None else _load_default_params()
        prep_cfg = self.params.get("preprocessing", {})
        win_cfg = self.params.get("windowing", {})
        model_cfg = self.params.get("model", {})

        self.fs = int(fs if fs is not None else prep_cfg.get("fs", 200))
        self.channels = int(prep_cfg.get("channels", 8))
        self.sub_window = int(sub_window if sub_window is not None else win_cfg.get("sub_window", 60))
        self.step_size = int(step if step is not None else win_cfg.get("step", 20))
        self.queue_len = int(queue_len if queue_len is not None else win_cfg.get("queue_len", 10))
        default_win = self.sub_window + (self.queue_len - 1) * self.step_size
        self.window_size = int(window_size if window_size is not None else win_cfg.get("window_size", default_win))

        if self.fs <= 0:
            raise ValueError(f"fs must be > 0, got {self.fs}")
        if self.channels <= 0:
            raise ValueError(f"channels must be > 0, got {self.channels}")
        if self.step_size <= 0:
            raise ValueError(f"step must be > 0, got {self.step_size}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")

        self.cadence_hz = float(cadence_hz if cadence_hz is not None else (self.fs / self.step_size))
        if self.cadence_hz <= 0:
            raise ValueError(f"cadence_hz must be > 0, got {self.cadence_hz}")
        self.tick_period_s = 1.0 / self.cadence_hz

        self.ring_buffer = ring_buffer
        self.pace_loop = bool(pace_loop)
        self.pop_timeout = float(pop_timeout)
        self.step_callback = step_callback
        self.device = torch.device(device)

        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = Preprocess(
                fs=self.fs,
                bandpass=prep_cfg.get("bandpass", [15, 95]),
                notch=prep_cfg.get("notch", 50),
                channels=self.channels,
            )

        if model is not None:
            self.model = model
        else:
            self.model = Model(
                channels=self.channels,
                targets=int(model_cfg.get("targets", 15)),
                window_size=self.window_size,
                length=int(model_cfg["length"]) if "length" in model_cfg else None,
                activation=str(model_cfg.get("activation", "relu")),
                hidden=int(model_cfg.get("hidden", 64)),
                downsample=int(model_cfg.get("downsample", 1)),
                ssm_speed=float(model_cfg.get("ssm_speed", 4.0)),
            )
        self.model.to(self.device)
        self.model.eval()

        self.udp_sender = udp_sender
        norm_cfg = self.params.get("normalization", {})
        if normalize_emg is not None:
            self.norm_enabled = bool(normalize_emg)
        elif getattr(self.model, "handles_normalization", False):
            self.norm_enabled = False
        else:
            self.norm_enabled = bool(norm_cfg.get("enabled", True))
        self.channel_offset: np.ndarray = np.zeros((self.channels, 1), dtype=np.float32)
        self._user_channel_scale: np.ndarray | None = None
        if channel_scale is not None:
            self.set_channel_scale(channel_scale)

        self._raw_buffer = np.zeros((self.channels, self.window_size), dtype=np.float32)
        self._filtered_buffer = np.zeros((self.channels, self.window_size), dtype=np.float32)
        self._raw_samples_accumulated = 0

        self.latency_monitor = LatencyMonitor(
            maxlen=1000,
            compute_budget_ms=10.0,
            transport_budget_ms=20.0,
        )
        self._steps_completed = 0
        self._samples_processed = 0
        self._latest_angles: np.ndarray | None = None
        self._latest_controls: np.ndarray | None = None
        self._latest_step_result: StepResult | None = None

        self.is_calibrating = False
        self.calib_phase = 0
        self.is_calibrated = False
        self.channel_perm = np.arange(self.channels, dtype=np.int64)
        self.channel_std = np.ones((self.channels, 1), dtype=np.float32)
        self._calib_start_time = 0.0
        self._calib_open_raw: list[np.ndarray] = []
        self._calib_fist_raw: list[np.ndarray] = []
        self._ctrl_sock: socket.socket | None = None
        self._ctrl_port: int | None = None

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def set_channel_scale(self, scale: float | Sequence[float] | np.ndarray | None) -> None:
        if scale is None:
            self._user_channel_scale = None
            return
        arr = np.asarray(scale, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((self.channels, 1), float(arr), dtype=np.float32)
        elif arr.shape == (self.channels,):
            arr = arr.reshape(self.channels, 1)
        elif arr.shape != (self.channels, 1):
            raise ValueError(f"channel_scale must be scalar or length {self.channels}, got {arr.shape}")
        self._user_channel_scale = np.maximum(arr, 1e-3)

    def reset_channel_scale(self) -> None:
        self._user_channel_scale = None

    def calibrate_guided(
        self,
        open_raw: Sequence[np.ndarray] | np.ndarray,
        fist_raw: Sequence[np.ndarray] | np.ndarray,
    ) -> dict[str, Any]:
        """Align pod rotation and per-channel scale from Open Hand and Tight Fist segments."""
        open_arr = np.concatenate(open_raw, axis=1) if isinstance(open_raw, list) else np.asarray(open_raw)
        fist_arr = np.concatenate(fist_raw, axis=1) if isinstance(fist_raw, list) else np.asarray(fist_raw)
        open_arr = open_arr.astype(np.float32)
        fist_arr = fist_arr.astype(np.float32)

        open_rms = np.sqrt(np.mean((open_arr - np.mean(open_arr, axis=1, keepdims=True)) ** 2, axis=1))
        fist_rms = np.sqrt(np.mean((fist_arr - np.mean(fist_arr, axis=1, keepdims=True)) ** 2, axis=1))
        ext_pod = int(np.argmax(open_rms))
        flex_pod = int(np.argmax(fist_rms))

        perm = np.roll(np.arange(self.channels, dtype=np.int64), -ext_pod)
        combined = np.concatenate([open_arr, fist_arr], axis=1)[perm]
        offset = np.mean(combined, axis=1, keepdims=True)
        if getattr(self.model, "handles_normalization", False):
            ch_std = np.ones((self.channels, 1), dtype=np.float32)
        else:
            ch_std = np.maximum(np.std(combined - offset, axis=1, keepdims=True), 1.0)

        with self._state_lock:
            self.channel_perm = perm
            self.channel_offset = offset.astype(np.float32)
            self.channel_std = ch_std.astype(np.float32)
            self.is_calibrated = True

        return {
            "extensor_pod": ext_pod + 1,
            "flexor_pod": flex_pod + 1,
            "channel_perm": perm.tolist(),
        }

    @property
    def raw_buffer(self) -> np.ndarray:
        with self._state_lock:
            return self._raw_buffer.copy()

    @property
    def filtered_buffer(self) -> np.ndarray:
        with self._state_lock:
            return self._filtered_buffer.copy()

    @property
    def steps_completed(self) -> int:
        with self._state_lock:
            return self._steps_completed

    @property
    def samples_processed(self) -> int:
        with self._state_lock:
            return self._samples_processed

    @property
    def is_warmed_up(self) -> bool:
        with self._state_lock:
            return self._raw_samples_accumulated >= self.window_size

    @property
    def latest_angles(self) -> np.ndarray | None:
        with self._state_lock:
            return self._latest_angles.copy() if self._latest_angles is not None else None

    @property
    def latest_controls(self) -> np.ndarray | None:
        with self._state_lock:
            return self._latest_controls.copy() if self._latest_controls is not None else None

    @property
    def latest_actuators(self) -> np.ndarray | None:
        return self.latest_controls

    @property
    def latest_step_result(self) -> StepResult | None:
        with self._state_lock:
            return self._latest_step_result

    @property
    def latest_compute_latency_ms(self) -> float:
        with self._state_lock:
            return self._latest_step_result.compute_latency_ms if self._latest_step_result is not None else 0.0

    @property
    def mean_compute_latency_ms(self) -> float:
        return self.latency_monitor.compute_stats()["mean"]

    @property
    def p95_compute_latency_ms(self) -> float:
        return self.latency_monitor.compute_stats()["p95"]

    @property
    def mean_transport_latency_ms(self) -> float:
        return self.latency_monitor.transport_stats()["mean"]

    def reset(self) -> None:
        with self._state_lock:
            self.preprocessor.reset_state()
            if hasattr(self.model, "reset") and callable(self.model.reset):
                self.model.reset()
            self._raw_buffer.fill(0.0)
            self._filtered_buffer.fill(0.0)
            self._raw_samples_accumulated = 0
            self._steps_completed = 0
            self._samples_processed = 0
            self._latest_angles = None
            self._latest_controls = None
            self._latest_step_result = None
            self.latency_monitor.reset()

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        out = chunk[self.channel_perm] - self.channel_offset
        if self._user_channel_scale is not None:
            out = out / self._user_channel_scale
        elif self.is_calibrated:
            out = out / self.channel_std
        elif self.norm_enabled:
            std = np.maximum(np.std(out, axis=-1, keepdims=True), 1.0)
            out = out / std
        return out.astype(np.float32, copy=False)

    def _predict_angles(self, window: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_window") and callable(self.model.predict_window):
            angles = self.model.predict_window(window)
            return np.clip(np.asarray(angles, dtype=np.float32).reshape(-1), 0.0, 180.0)

        with torch.no_grad():
            x = torch.from_numpy(window[np.newaxis, ...].astype(np.float32)).to(self.device)
            out = self.model(x).detach().cpu().numpy()
        return np.clip(np.asarray(out, dtype=np.float32).reshape(-1), 0.0, 180.0)

    def warmup(self, initial_emg: np.ndarray | None = None, pop_timeout: float = 1.0) -> None:
        if initial_emg is not None:
            arr = np.asarray(initial_emg, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != self.channels:
                raise ValueError(f"Expected initial_emg shape ({self.channels}, N), got {arr.shape}")
            for start in range(0, arr.shape[1] - self.step_size + 1, self.step_size):
                self.step(raw_chunk=arr[:, start : start + self.step_size], transmit=False)
        else:
            steps_needed = max(1, self.window_size // self.step_size)
            for _ in range(steps_needed):
                self.step(pop_timeout=pop_timeout, transmit=False)

    def step(
        self,
        raw_chunk: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        pop_timeout: float | None = None,
        transmit: bool = True,
    ) -> StepResult:
        if raw_chunk is None:
            wait_timeout = self.pop_timeout if pop_timeout is None else float(pop_timeout)
            raw_chunk, timestamps = self.ring_buffer.pop_chunk(self.step_size, timeout=wait_timeout)

        chunk = np.asarray(raw_chunk, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"Expected 2D chunk array, got {chunk.ndim}D with shape {chunk.shape}")
        if chunk.shape != (self.channels, self.step_size):
            raise ValueError(
                f"Channel count mismatch or sample size mismatch: "
                f"expected ({self.channels}, {self.step_size}), got {chunk.shape}"
            )

        t_compute_start = time.perf_counter()
        norm_chunk = self._normalize_chunk(chunk)
        filtered_chunk = self.preprocessor.filter_chunk(norm_chunk)

        with self._state_lock:
            self._raw_buffer[:, : -self.step_size] = self._raw_buffer[:, self.step_size :]
            self._raw_buffer[:, -self.step_size :] = chunk
            self._filtered_buffer[:, : -self.step_size] = self._filtered_buffer[:, self.step_size :]
            self._filtered_buffer[:, -self.step_size :] = filtered_chunk
            self._raw_samples_accumulated += self.step_size
            self._samples_processed += self.step_size
            window_snapshot = self._filtered_buffer.copy()

        angles_deg = self._predict_angles(window_snapshot)
        t_compute_end = time.perf_counter()
        compute_latency_ms = (t_compute_end - t_compute_start) * 1000.0

        controls = map_angles_to_controls(angles_deg, params=self.params)
        if transmit and self.udp_sender is not None:
            self.udp_sender.send(controls)

        t_done = time.perf_counter()
        if timestamps is not None and timestamps.size > 0:
            t_ref = float(timestamps[-1])
            transport_latency_ms = max(0.0, (t_done - t_ref) * 1000.0)
        else:
            transport_latency_ms = (t_done - t_compute_start) * 1000.0

        self.latency_monitor.record(compute_latency_ms, transport_latency_ms)

        with self._state_lock:
            self._steps_completed += 1
            step_idx = self._steps_completed
            self._latest_angles = angles_deg
            self._latest_controls = controls
            res = StepResult(
                step_index=step_idx,
                angles=angles_deg,
                controls=controls,
                compute_latency_ms=compute_latency_ms,
                transport_latency_ms=transport_latency_ms,
                timestamp=t_done,
            )
            self._latest_step_result = res

        if self.step_callback is not None:
            self.step_callback(res)

        return res

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="StreamingInferenceWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_latency_stats(self) -> dict[str, dict[str, float]]:
        return self.latency_monitor.summary()

    def _run_loop(self) -> None:
        next_deadline = time.perf_counter()
        while not self._stop_event.is_set():
            if self.pace_loop:
                now = time.perf_counter()
                sleep_dur = next_deadline - now
                if sleep_dur > 0:
                    if self._stop_event.wait(timeout=sleep_dur):
                        break
                next_deadline = max(next_deadline + self.tick_period_s, time.perf_counter())

            try:
                self.step(pop_timeout=self.pop_timeout, transmit=True)
            except TimeoutError:
                continue
            except Exception as exc:
                logger.error("Unexpected error in StreamingInferenceWorker loop: %s", exc, exc_info=True)
                if self._stop_event.wait(timeout=0.01):
                    break

    def __enter__(self) -> StreamingInferenceWorker:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
