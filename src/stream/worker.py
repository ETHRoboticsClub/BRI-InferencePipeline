"""Stateful streaming Temporal Riemannian Regressor (TRR) inference engine."""

from __future__ import annotations

import copy
import json
import logging
import pathlib
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from src.kinematics.mapping import map_angles_to_controls
from src.kinematics.udp_sender import UDPSender
from src.model.model import Model
from src.preprocess import Preprocess
from src.stream.ring_buffer import EMGRingBuffer

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "params.yaml"


def _fast_ledoit_wolf(x_centered: np.ndarray) -> np.ndarray:
    """Analytical Ledoit-Wolf shrinkage covariance for a zero-mean (p, n) matrix."""
    p, n = x_centered.shape
    if n <= 1:
        return np.eye(p, dtype=np.float32)
    xt = x_centered.T.astype(np.float64)
    emp_cov = (xt.T @ xt) / float(n)
    mu = float(np.trace(emp_cov)) / float(p)
    eye_p = np.eye(p, dtype=np.float64)
    delta = float(np.sum((emp_cov - mu * eye_p) ** 2)) / float(p)
    if delta <= 0.0:
        return emp_cov.astype(np.float32)
    x2 = xt**2
    beta_0 = (float(np.sum(x2.T @ x2)) / float(p * n * n)) - (float(np.sum(emp_cov**2)) / float(p * n))
    shrinkage = min(delta, max(0.0, beta_0)) / delta
    return ((1.0 - shrinkage) * emp_cov + shrinkage * mu * eye_p).astype(np.float32)


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


def _load_default_params(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load configuration from params dict or default params.yaml."""
    base: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.is_file():
        try:
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if loaded and isinstance(loaded, dict):
                base = loaded
        except Exception:
            pass

    if not base:
        base = {
            "preprocessing": {"fs": 200, "bandpass": [15, 95], "notch": 50, "channels": 8},
            "windowing": {"sub_window": 60, "step": 20, "queue_len": 10},
            "socket": {"host": "127.0.0.1", "port": 5005},
        }

    if params is not None:
        merged = copy.deepcopy(base)
        for k, v in params.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v
        return merged

    return base


@dataclass
class StepResult:
    """Result of a single streaming inference step."""

    angles: np.ndarray
    controls: np.ndarray
    compute_latency_ms: float
    transport_latency_ms: float
    tangent_vector: np.ndarray

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"Invalid StepResult key: {item}")


class LatencyMonitor:
    """Tracks per-step forward compute and end-to-end transport latency metrics."""

    def __init__(self, max_history: int = 10000) -> None:
        self.max_history = int(max_history)
        self._compute_latencies: list[float] = []
        self._transport_latencies: list[float] = []
        self._lock = threading.Lock()

    def record(self, compute_ms: float, transport_ms: float) -> None:
        with self._lock:
            self._compute_latencies.append(float(compute_ms))
            self._transport_latencies.append(float(transport_ms))
            if len(self._compute_latencies) > self.max_history:
                self._compute_latencies.pop(0)
                self._transport_latencies.pop(0)

    def reset(self) -> None:
        with self._lock:
            self._compute_latencies.clear()
            self._transport_latencies.clear()

    @staticmethod
    def _compute_stats(data: list[float]) -> dict[str, float]:
        if not data:
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
            }
        arr = np.asarray(data, dtype=np.float64)
        jitter = float(np.mean(np.abs(np.diff(arr)))) if len(arr) > 1 else 0.0
        return {
            "count": float(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "jitter": jitter,
        }

    def compute_stats(self) -> dict[str, float]:
        with self._lock:
            return self._compute_stats(self._compute_latencies)

    def transport_stats(self) -> dict[str, float]:
        with self._lock:
            return self._compute_stats(self._transport_latencies)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "compute": self._compute_stats(self._compute_latencies),
                "transport": self._compute_stats(self._transport_latencies),
            }

    @property
    def mean_compute_ms(self) -> float:
        return self.compute_stats()["mean"]

    @property
    def mean_transport_ms(self) -> float:
        return self.transport_stats()["mean"]

    @property
    def p95_compute_ms(self) -> float:
        return self.compute_stats()["p95"]

    @property
    def p95_transport_ms(self) -> float:
        return self.transport_stats()["p95"]

    @property
    def jitter_compute_ms(self) -> float:
        return self.compute_stats()["jitter"]

    @property
    def jitter_transport_ms(self) -> float:
        return self.transport_stats()["jitter"]

    @property
    def compute_latencies(self) -> list[float]:
        with self._lock:
            return list(self._compute_latencies)

    @property
    def transport_latencies(self) -> list[float]:
        with self._lock:
            return list(self._transport_latencies)


class StreamingInferenceWorker:
    """Stateful 10 Hz streaming TRR inference worker and latency monitor."""

    def __init__(
        self,
        ring_buffer: EMGRingBuffer,
        model: Model,
        preprocessor: Preprocess | None = None,
        udp_sender: UDPSender | None = None,
        params: dict[str, Any] | None = None,
        sub_window: int | None = None,
        step: int | None = None,
        queue_len: int | None = None,
        fs: int | None = None,
        cadence_hz: float | None = None,
        pace_loop: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.ring_buffer = ring_buffer
        self.model = model
        self.udp_sender = udp_sender
        self.params = _load_default_params(params)
        self.device = torch.device(device)
        self.pace_loop = bool(pace_loop)

        prep_cfg = self.params.get("preprocessing", {})
        win_cfg = self.params.get("windowing", {})

        self.fs = int(fs if fs is not None else prep_cfg.get("fs", 200))
        default_sub_win = win_cfg.get("sub_window", 60)
        self.sub_window = int(sub_window if sub_window is not None else default_sub_win)
        self.step_size = int(step if step is not None else win_cfg.get("step", 20))
        self.queue_len = int(queue_len if queue_len is not None else win_cfg.get("queue_len", 10))
        self.channels = int(prep_cfg.get("channels", getattr(ring_buffer, "channels", 8)))
        self.cadence_hz = float(cadence_hz) if cadence_hz is not None else float(self.fs / self.step_size)

        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = Preprocess(
                fs=self.fs,
                bandpass=prep_cfg.get("bandpass", [15, 95]),
                notch=float(prep_cfg.get("notch", 50.0)),
                channels=self.channels,
            )

        self.model.eval()
        self.model.to(self.device)
        self.latency_monitor = LatencyMonitor()

        self._raw_window_len = max(self.sub_window, 300)
        self._bands_sub_window = self.sub_window
        self._raw_buffer = np.zeros((self.channels, self._raw_window_len), dtype=np.float32)
        self._filtered_buffer = np.zeros((self.channels, self.sub_window), dtype=np.float32)
        self._filtered_bands_buffer = np.zeros((3, self.channels, self._bands_sub_window), dtype=np.float32)

        self.channel_std = np.full((3, self.channels, 1), 6.0, dtype=np.float32)
        self.inv_sqrt_mean_cov = np.tile(np.eye(self.channels, dtype=np.float32), (3, 1, 1))
        self.is_calibrated = False
        self._full_manifold_calibrated = False
        self._intrasubject_direct = False
        self.warmup_active_subwindows: list[np.ndarray] = []
        self.warmup_target_count = 50
        self._smoothed_angles: np.ndarray | None = None
        self._prev_log_rms: np.ndarray | None = None
        self._row, self._col = np.triu_indices(self.channels)
        self._diag_idx = np.where(self._row == self._col)[0]
        self._triu_weights = np.where(self._row == self._col, 1.0, np.sqrt(2.0)).astype(np.float32)

        if getattr(self.model, "intrasubject_calibrated", False):
            c_std = getattr(self.model, "calibrated_channel_std", None)
            c_inv = getattr(self.model, "calibrated_inv_sqrt_mean_cov", None)
            if c_std is not None and c_inv is not None:
                self.channel_std = np.asarray(c_std, dtype=np.float32).copy()
                self.inv_sqrt_mean_cov = np.asarray(c_inv, dtype=np.float32).copy()
                self.is_calibrated = True
                self._full_manifold_calibrated = True
                self._intrasubject_direct = not getattr(self.model, "calibrated_manifold", False)
            intra_bands = getattr(self.model, "intrasubject_bands", None)
            if intra_bands is not None and hasattr(self.preprocessor, "set_bands"):
                self.preprocessor.set_bands(intra_bands)
            intra_sub_len = getattr(self.model, "intrasubject_sub_len", None)
            if intra_sub_len is not None and int(intra_sub_len) >= 40:
                self._bands_sub_window = int(intra_sub_len)
                self._filtered_bands_buffer = np.zeros((3, self.channels, self._bands_sub_window), dtype=np.float32)
            intra_seq_len = getattr(self.model, "intrasubject_seq_len", None)
            if queue_len is None and intra_seq_len is not None and int(intra_seq_len) >= 5:
                self.queue_len = int(intra_seq_len)

        self.channel_perm = np.arange(self.channels, dtype=np.int64)
        self.extensor_channels: list[int] = [0, 1]
        self.flexor_channels: list[int] = [2, 3, 4, 5, 6, 7]

        norm_cfg = self.params.get("normalization", {})
        self.norm_enabled = bool(norm_cfg.get("enabled", True))
        self.norm_mode = str(norm_cfg.get("mode", "adaptive")).lower()
        self.norm_fixed_scale = float(norm_cfg.get("fixed_scale", 35.0))
        self.min_std = float(norm_cfg.get("min_std", 1.0))
        buf_sec = float(norm_cfg.get("buffer_seconds", 5.0))
        self.norm_buf_size = max(int(buf_sec * self.fs), 200)
        self._norm_buffer = np.zeros((self.channels, self.norm_buf_size), dtype=np.float32)
        self._norm_count = 0
        self._user_channel_scale: np.ndarray | None = None

        dense_mod = getattr(self.model, "gru", None)
        gru_in_dim = dense_mod.dense[0].in_features if (dense_mod is not None and hasattr(dense_mod, "dense")) else 108
        if prep_cfg.get("multi_band", False) or len(prep_cfg.get("bands", [])) > 1:
            tangent_dim = gru_in_dim
        else:
            tangent_dim = self.model.tangent_space.out_dim

        self.tangent_dim = tangent_dim
        self._fifo_tangent_seq = torch.zeros((1, self.queue_len, tangent_dim), dtype=torch.float32, device=self.device)
        self._fifo_tangent_108 = torch.zeros(
            (1, self.queue_len, max(108, gru_in_dim)), dtype=torch.float32, device=self.device
        )

        self._steps_completed = 0
        self._samples_processed = 0
        self._latest_angles: np.ndarray | None = None
        self._latest_controls: np.ndarray | None = None
        self._smoothed_controls: np.ndarray | None = None
        self._latest_step_result: StepResult | None = None
        self._latest_compute_latency_ms: float = 0.0
        self._latest_transport_latency_ms: float = 0.0
        self._last_error: Exception | None = None

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()

        self.is_calibrating: bool = False
        self.calib_phase: int = 0
        self.calib_status_text: str = "Ready"
        self._calib_start_time: float = 0.0
        self._calib_open_raw: list[np.ndarray] = []
        self._calib_fist_raw: list[np.ndarray] = []
        self._calib_open_angles: list[np.ndarray] = []
        self._calib_fist_angles: list[np.ndarray] = []
        self._calib_rms_samples: list[np.ndarray] = []
        self._ctrl_sock: socket.socket | None = None
        self._ctrl_port: int | None = None

    @property
    def raw_buffer(self) -> np.ndarray:
        with self._state_lock:
            return self._raw_buffer[:, -self.sub_window :].copy()

    @property
    def filtered_buffer(self) -> np.ndarray:
        with self._state_lock:
            return self._filtered_buffer.copy()

    @property
    def tangent_fifo(self) -> torch.Tensor:
        with self._state_lock:
            return self._fifo_tangent_seq.detach().clone()

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
            return self._steps_completed >= self.queue_len

    @property
    def latest_angles(self) -> np.ndarray | None:
        with self._state_lock:
            return self._latest_angles.copy() if self._latest_angles is not None else None

    @property
    def latest_controls(self) -> np.ndarray | None:
        with self._state_lock:
            return self._latest_controls.copy() if self._latest_controls is not None else None

    @property
    def latest_step_result(self) -> StepResult | None:
        with self._state_lock:
            return self._latest_step_result

    @property
    def mean_compute_latency_ms(self) -> float:
        return self.latency_monitor.mean_compute_ms

    @property
    def mean_transport_latency_ms(self) -> float:
        return self.latency_monitor.mean_transport_ms

    @property
    def mean_latency_ms(self) -> float:
        return self.latency_monitor.mean_compute_ms

    @property
    def p95_compute_latency_ms(self) -> float:
        return self.latency_monitor.p95_compute_ms

    @property
    def p95_transport_latency_ms(self) -> float:
        return self.latency_monitor.p95_transport_ms

    @property
    def latest_compute_latency_ms(self) -> float:
        with self._state_lock:
            return self._latest_compute_latency_ms

    @property
    def latest_transport_latency_ms(self) -> float:
        with self._state_lock:
            return self._latest_transport_latency_ms

    def get_latency_stats(self) -> dict[str, Any]:
        return self.latency_monitor.summary()

    def set_channel_scale(self, scale: np.ndarray | Sequence[float] | float) -> None:
        with self._state_lock:
            arr = np.asarray(scale, dtype=np.float32)
            if arr.ndim == 0:
                self._user_channel_scale = np.full((self.channels, 1), float(arr), dtype=np.float32)
            elif arr.ndim == 1 and arr.size == self.channels:
                self._user_channel_scale = arr.reshape(self.channels, 1)
            elif arr.shape == (self.channels, 1):
                self._user_channel_scale = arr
            else:
                raise ValueError(f"Invalid scale shape: {arr.shape}")
            band_scale = np.maximum(self._user_channel_scale * 0.55, 3.0).astype(np.float32)
            for b_idx in range(3):
                self.channel_std[b_idx] = band_scale.copy()

    def reset_channel_scale(self) -> None:
        with self._state_lock:
            self._user_channel_scale = None
            self.channel_std.fill(6.0)

    def reset(self) -> None:
        with self._state_lock:
            self._raw_buffer.fill(0.0)
            self._filtered_buffer.fill(0.0)
            self._filtered_bands_buffer.fill(0.0)
            self._norm_buffer.fill(0.0)
            self._norm_count = 0
            self._fifo_tangent_seq.zero_()
            self._fifo_tangent_108.zero_()
            self.preprocessor.reset_state(channels=self.channels)
            self.latency_monitor.reset()
            self._steps_completed = 0
            self._samples_processed = 0
            self._latest_angles = None
            self._latest_controls = None
            self._smoothed_controls = None
            self._smoothed_angles = None
            self._prev_log_rms = None
            self._latest_step_result = None
            self._latest_compute_latency_ms = 0.0
            self._latest_transport_latency_ms = 0.0
            self._last_error = None

    def calibrate(self, active_subwindows: Sequence[np.ndarray] | None = None) -> None:
        """Fit active channel std and Riemannian mean whitening from active subwindows."""
        with self._state_lock:
            if active_subwindows is not None:
                self.warmup_active_subwindows = [np.asarray(w, dtype=np.float32) for w in active_subwindows]
            if len(self.warmup_active_subwindows) < 2:
                return

            try:
                from pyriemann.estimation import Covariances
                from pyriemann.utils.mean import mean_covariance

                cov_estimator = Covariances(estimator="lwf")
                for b_idx in range(3):
                    active_b = np.concatenate([w[b_idx] for w in self.warmup_active_subwindows], axis=-1)
                    std_b = np.std(active_b, axis=-1, keepdims=True)
                    self.channel_std[b_idx] = np.maximum(std_b, 3.0).astype(np.float32)

                    scaled_subs = np.stack(
                        [
                            (w[b_idx] - np.mean(w[b_idx], axis=-1, keepdims=True)) / self.channel_std[b_idx]
                            for w in self.warmup_active_subwindows
                        ],
                        axis=0,
                    )
                    covs_b = cov_estimator.fit_transform(scaled_subs).astype(np.float32)
                    covs_b += 1e-6 * np.eye(self.channels, dtype=np.float32)
                    mean_cov_b = mean_covariance(covs_b, metric="riemann").astype(np.float32)

                    eigvals, eigvecs = np.linalg.eigh(mean_cov_b)
                    eigvals = np.maximum(eigvals, 1e-4)
                    self.inv_sqrt_mean_cov[b_idx] = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype(
                        np.float32
                    )

                self.is_calibrated = True
                self._full_manifold_calibrated = active_subwindows is not None
            except Exception as exc:
                logger.error("Active calibration failed: %s", exc)

    def calibrate_guided(
        self,
        open_raw_samples: np.ndarray | Sequence[np.ndarray],
        fist_raw_samples: np.ndarray | Sequence[np.ndarray],
    ) -> dict[str, Any]:
        """Calibrate pod rotation alignment and per-band channel std from Open and Fist phases."""
        with self._state_lock:
            open_arr = (
                np.concatenate(open_raw_samples, axis=-1)
                if isinstance(open_raw_samples, (list, tuple))
                else np.asarray(open_raw_samples, dtype=np.float32)
            )
            fist_arr = (
                np.concatenate(fist_raw_samples, axis=-1)
                if isinstance(fist_raw_samples, (list, tuple))
                else np.asarray(fist_raw_samples, dtype=np.float32)
            )

            if open_arr.ndim == 2 and open_arr.shape[0] != self.channels and open_arr.shape[1] == self.channels:
                open_arr = open_arr.T
            if fist_arr.ndim == 2 and fist_arr.shape[0] != self.channels and fist_arr.shape[1] == self.channels:
                fist_arr = fist_arr.T

            open_dc = open_arr - np.mean(open_arr, axis=-1, keepdims=True)
            fist_dc = fist_arr - np.mean(fist_arr, axis=-1, keepdims=True)

            rms_ext = np.sqrt(np.mean(open_dc**2, axis=-1))
            rms_flex = np.sqrt(np.mean(fist_dc**2, axis=-1))

            ext_diff = rms_ext - 0.45 * rms_flex
            best_ext_pod = int(np.argmax(ext_diff))

            flex_diff = rms_flex - 0.35 * rms_ext
            flex_diff[best_ext_pod] = -1e9
            best_flex_pod = int(np.argmax(flex_diff))

            perm = np.roll(np.arange(self.channels, dtype=np.int64), -best_ext_pod)
            flex_pos = int(np.where(perm == best_flex_pod)[0][0])
            if flex_pos != 4:
                perm[4], perm[flex_pos] = perm[flex_pos], perm[4]

            if self._intrasubject_direct:
                self.channel_perm = np.arange(self.channels, dtype=np.int64)
            else:
                self.channel_perm = perm

            combined_raw = np.concatenate([open_arr, fist_arr], axis=-1)[self.channel_perm]
            temp_prep = Preprocess(
                fs=self.fs,
                bandpass=self.preprocessor.bandpass_cutoffs,
                notch=self.preprocessor.notch_freq,
                channels=self.channels,
                bands=getattr(self.preprocessor, "bands", None),
            )
            bands_combined = temp_prep.filter_bands(combined_raw)

            for b_idx in range(3):
                std_b = np.std(bands_combined[b_idx], axis=-1, keepdims=True)
                self.channel_std[b_idx] = np.maximum(std_b, 1.5).astype(np.float32)
                if self._intrasubject_direct:
                    b_norm = bands_combined[b_idx] / self.channel_std[b_idx]
                    n_total = b_norm.shape[-1]
                    cov_list = []
                    w_len = self._bands_sub_window
                    for s_i in range(0, max(1, n_total - w_len + 1), max(1, w_len // 2)):
                        sub_seg = b_norm[:, s_i : s_i + w_len]
                        if sub_seg.shape[-1] >= 20:
                            sub_seg = sub_seg - np.mean(sub_seg, axis=-1, keepdims=True)
                            c_seg = _fast_ledoit_wolf(sub_seg)
                            cov_list.append(c_seg + 1e-6 * np.eye(self.channels, dtype=np.float32))
                    if cov_list:
                        mean_c = np.mean(cov_list, axis=0)
                        evals, evecs = np.linalg.eigh(mean_c)
                        evals = np.maximum(evals, 1e-6)
                        new_w = (evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T).astype(np.float32)
                        self.inv_sqrt_mean_cov[b_idx] = (0.50 * self.inv_sqrt_mean_cov[b_idx] + 0.50 * new_w).astype(
                            np.float32
                        )

            self.is_calibrated = True
            if self._intrasubject_direct:
                self._full_manifold_calibrated = True

            return {
                "best_extensor_pod": best_ext_pod + 1,
                "best_flexor_pod": best_flex_pod + 1,
                "extensor_pod": best_ext_pod + 1,
                "flexor_pod": best_flex_pod + 1,
                "channel_perm": self.channel_perm.tolist(),
                "channel_std_mean": float(np.mean(self.channel_std)),
            }

    def _start_control_socket(self) -> None:
        if self._ctrl_sock is not None:
            return
        for cand_port in (5006, 5007, 5008, 5009):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind(("127.0.0.1", cand_port))
                sock.setblocking(False)
                self._ctrl_sock = sock
                self._ctrl_port = cand_port
                return
            except OSError:
                sock.close()

    def _stop_control_socket(self) -> None:
        if self._ctrl_sock is not None:
            try:
                self._ctrl_sock.close()
            except Exception:
                pass
            self._ctrl_sock = None
            self._ctrl_port = None

    def _notify_mujoco_status(self, phase: int, text: str, detail: str = "") -> None:
        payload = json.dumps({"type": "calib_status", "phase": phase, "text": text, "detail": detail}).encode("utf-8")
        sock = self._ctrl_sock
        close_after = sock is None
        if sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(payload, ("127.0.0.1", 5005))
        except Exception:
            pass
        finally:
            if close_after:
                sock.close()

    def export_calibration_state(self) -> dict[str, Any]:
        with self._state_lock:
            kin_cfg = self.params.get("kinematics", {})
            return {
                "channel_perm": self.channel_perm.tolist(),
                "channel_std": self.channel_std.tolist(),
                "user_channel_scale": (
                    self._user_channel_scale.tolist() if self._user_channel_scale is not None else None
                ),
                "empirical_bounds": kin_cfg.get("empirical_bounds"),
                "mode": kin_cfg.get("mode", "empirical"),
                "calib_status_text": self.calib_status_text,
            }

    def import_calibration_state(self, state: dict[str, Any]) -> None:
        with self._state_lock:
            if state.get("channel_perm") is not None:
                self.channel_perm = np.asarray(state["channel_perm"], dtype=np.int64)
            if state.get("channel_std") is not None:
                self.channel_std = np.asarray(state["channel_std"], dtype=np.float32)
            if state.get("user_channel_scale") is not None:
                self._user_channel_scale = np.asarray(state["user_channel_scale"], dtype=np.float32)
            if state.get("empirical_bounds") is not None:
                kin_cfg = self.params.setdefault("kinematics", {})
                kin_cfg["mode"] = state.get("mode", "empirical")
                kin_cfg["empirical_bounds"] = state["empirical_bounds"]
            self.is_calibrated = True
            self.is_calibrating = False
            self.calib_phase = 3
            if state.get("calib_status_text"):
                self.calib_status_text = str(state["calib_status_text"])

    def broadcast_calibration_sync(self) -> None:
        msg = json.dumps({"cmd": "calib_sync", "state": self.export_calibration_state()}).encode("utf-8")
        sock = self._ctrl_sock
        close_after = sock is None
        if sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for p in (5006, 5007, 5008, 5009):
                if p != self._ctrl_port:
                    try:
                        sock.sendto(msg, ("127.0.0.1", p))
                    except Exception:
                        pass
        finally:
            if close_after:
                sock.close()

    def start_interactive_calibration(self, broadcast: bool = True) -> None:
        with self._state_lock:
            if self.is_calibrating:
                return
            self.is_calibrating = True
            self.calib_phase = 1
            self._calib_start_time = time.perf_counter()
            self._calib_open_raw.clear()
            self._calib_fist_raw.clear()
            self._calib_open_angles.clear()
            self._calib_fist_angles.clear()
            self._calib_rms_samples.clear()
            self.calib_status_text = "CALIB (1/2): EXTEND HAND OPEN FLAT (2.0s)"
            self._notify_mujoco_status(1, self.calib_status_text, "Extend your fingers wide open")

            if broadcast:
                msg = json.dumps({"cmd": "calibrate_peer"}).encode("utf-8")
                sock = self._ctrl_sock
                close_after = sock is None
                if sock is None:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    for p in (5006, 5007, 5008, 5009):
                        if p != self._ctrl_port:
                            try:
                                sock.sendto(msg, ("127.0.0.1", p))
                            except Exception:
                                pass
                finally:
                    if close_after:
                        sock.close()

    def _poll_control_socket(self) -> None:
        if self._ctrl_sock is None:
            return
        while True:
            try:
                data, _ = self._ctrl_sock.recvfrom(8192)
            except (BlockingIOError, Exception):
                break
            try:
                msg = json.loads(data.decode("utf-8"))
                cmd = msg.get("cmd")
                if cmd == "calibrate":
                    self.start_interactive_calibration(broadcast=True)
                elif cmd == "calibrate_peer":
                    self.start_interactive_calibration(broadcast=False)
                elif cmd == "calib_sync" and isinstance(msg.get("state"), dict):
                    self.import_calibration_state(msg["state"])
            except Exception:
                pass

    def _step_interactive_calibration(
        self, raw_sub: np.ndarray, rms_per_ch: np.ndarray, angles_deg: np.ndarray
    ) -> None:
        if not self.is_calibrating:
            return
        elapsed = time.perf_counter() - self._calib_start_time
        if elapsed < 2.0:
            self.calib_phase = 1
            rem = max(0.0, 2.0 - elapsed)
            self.calib_status_text = f"CALIB (1/2): EXTEND HAND OPEN FLAT ({rem:3.1f}s)"
            self._calib_open_raw.append(raw_sub[:, -self.step_size :].copy())
            self._calib_open_angles.append(angles_deg.copy())
            self._calib_rms_samples.append(rms_per_ch.copy())
            self._notify_mujoco_status(1, self.calib_status_text, "Hold fingers extended flat")
        elif elapsed < 4.0:
            self.calib_phase = 2
            rem = max(0.0, 4.0 - elapsed)
            self.calib_status_text = f"CALIB (2/2): MAKE A TIGHT FIST ({rem:3.1f}s)"
            self._calib_fist_raw.append(raw_sub[:, -self.step_size :].copy())
            self._calib_fist_angles.append(angles_deg.copy())
            self._calib_rms_samples.append(rms_per_ch.copy())
            self._notify_mujoco_status(2, self.calib_status_text, "Close hand into a firm fist")
        else:
            self.is_calibrating = False
            self.calib_phase = 3
            info = {}
            if self._calib_open_raw and self._calib_fist_raw:
                try:
                    info = self.calibrate_guided(self._calib_open_raw, self._calib_fist_raw)
                except Exception:
                    pass
            if self._calib_open_angles and self._calib_fist_angles:
                open_mean = np.mean(self._calib_open_angles, axis=0)
                fist_mean = np.mean(self._calib_fist_angles, axis=0)
                ext_b = np.maximum(open_mean, fist_mean)
                flex_b = np.minimum(open_mean, fist_mean)
                too_small = (ext_b - flex_b) < 4.0
                if np.any(too_small):
                    ext_b[too_small] += 2.0
                    flex_b[too_small] -= 2.0
                kin_cfg = self.params.setdefault("kinematics", {})
                kin_cfg["mode"] = "empirical"
                kin_cfg["empirical_bounds"] = {
                    "extension": ext_b.tolist(),
                    "flexion": flex_b.tolist(),
                }
            if self._calib_rms_samples:
                active_rms = np.percentile(np.array(self._calib_rms_samples), 85, axis=0)
                self.set_channel_scale(np.maximum(active_rms, 5.0))

            ext_p = info.get("extensor_pod", 1)
            flex_p = info.get("flexor_pod", 5)
            self.calib_status_text = f"CALIBRATION COMPLETE! (Ext: P{ext_p} | Flex: P{flex_p})"
            self._notify_mujoco_status(3, self.calib_status_text, "Pod rotation & gains aligned")
            self.broadcast_calibration_sync()

    def warmup(self, initial_emg: np.ndarray | None = None, n_samples: int = 240) -> None:
        """Warm up rolling buffers and tangent FIFO with initial samples."""
        if initial_emg is None and len(self.ring_buffer) >= self.step_size:
            avail = (min(len(self.ring_buffer), n_samples) // self.step_size) * self.step_size
            if avail > 0:
                initial_emg, _ = self.ring_buffer.get_latest(avail)

        if initial_emg is None:
            return

        arr = np.asarray(initial_emg, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"initial_emg must be 2D array, got shape {arr.shape}")
        if arr.shape[0] != self.channels and arr.shape[1] == self.channels:
            arr = arr.T

        total = arr.shape[1]
        for start_idx in range(0, total - self.step_size + 1, self.step_size):
            self.step(raw_chunk=arr[:, start_idx : start_idx + self.step_size], transmit=False)

    def step(
        self,
        raw_chunk: np.ndarray | None = None,
        timestamps: np.ndarray | Sequence[float] | None = None,
        transmit: bool = True,
        pop_timeout: float = 0.5,
    ) -> StepResult:
        """Execute one complete streaming inference step."""
        if raw_chunk is None:
            chunk_data, ts_data = self.ring_buffer.pop_chunk(self.step_size, timeout=pop_timeout)
            raw_chunk = chunk_data
            if timestamps is None:
                timestamps = ts_data

        arr = np.asarray(raw_chunk, dtype=np.float32)
        if arr.ndim == 1:
            if arr.size != self.channels:
                raise ValueError(f"Sample size {arr.size} does not match channel count {self.channels}")
            arr = arr[:, np.newaxis]
        elif arr.ndim == 2:
            if arr.shape[0] != self.channels and arr.shape[1] == self.channels:
                arr = arr.T
            if arr.shape[0] != self.channels:
                raise ValueError(f"Channel count mismatch: expected {self.channels}, got shape {arr.shape}")
        else:
            raise ValueError(f"Expected 2D chunk array, got shape {arr.shape}")

        n_samples = arr.shape[1]
        t_start = time.perf_counter()

        if timestamps is None:
            dt = 1.0 / self.fs
            ts = t_start - (n_samples - 1 - np.arange(n_samples, dtype=np.float64)) * dt
        else:
            ts = np.asarray(timestamps, dtype=np.float64)
        t_acq = float(ts[-1]) if ts.size > 0 else t_start

        with self._state_lock:
            if n_samples >= self._raw_window_len:
                self._raw_buffer[:, :] = arr[:, -self._raw_window_len :]
            else:
                self._raw_buffer = np.roll(self._raw_buffer, -n_samples, axis=-1)
                self._raw_buffer[:, -n_samples:] = arr

            raw_sub = self._raw_buffer[self.channel_perm, -self.sub_window :]
            raw_dc = raw_sub - np.mean(raw_sub, axis=-1, keepdims=True)
            rms_raw = float(np.sqrt(np.mean(raw_dc**2)))
            rms_per_ch = np.sqrt(np.mean(raw_dc**2, axis=-1))

            filtered_chunk = self.preprocessor.filter_chunk(arr)
            if n_samples >= self.sub_window:
                self._filtered_buffer[:, :] = filtered_chunk[:, -self.sub_window :]
            else:
                self._filtered_buffer = np.roll(self._filtered_buffer, -n_samples, axis=-1)
                self._filtered_buffer[:, -n_samples:] = filtered_chunk

            dense_mod = getattr(self.model, "gru", None)
            expected_dim = (
                dense_mod.dense[0].in_features
                if (dense_mod is not None and hasattr(dense_mod, "dense"))
                else self.model.tangent_space.out_dim
            )

            if expected_dim in (108, 124):
                bands_chunk = self.preprocessor.filter_bands(arr)
                if n_samples >= self._bands_sub_window:
                    self._filtered_bands_buffer[:, :, :] = bands_chunk[:, :, -self._bands_sub_window :]
                else:
                    self._filtered_bands_buffer = np.roll(self._filtered_bands_buffer, -n_samples, axis=-1)
                    self._filtered_bands_buffer[:, :, -n_samples:] = bands_chunk

            if self.norm_enabled:
                if n_samples >= self.norm_buf_size:
                    self._norm_buffer[:, :] = filtered_chunk[:, -self.norm_buf_size :]
                else:
                    self._norm_buffer = np.roll(self._norm_buffer, -n_samples, axis=-1)
                    self._norm_buffer[:, -n_samples:] = filtered_chunk
                self._norm_count += n_samples

            sub_window_np = self._filtered_buffer[:, -self.sub_window :].copy()
            if self.norm_enabled:
                if self._user_channel_scale is not None:
                    scale = np.maximum(self._user_channel_scale, self.min_std)
                elif self.norm_mode == "fixed":
                    scale = np.full((self.channels, 1), self.norm_fixed_scale, dtype=np.float32)
                else:
                    valid_samples = min(self._norm_count, self.norm_buf_size)
                    if valid_samples >= self.sub_window:
                        chan_std = np.std(self._norm_buffer[:, -valid_samples:], axis=-1, keepdims=True)
                        scale = np.maximum(chan_std, self.min_std)
                    else:
                        scale = np.full((self.channels, 1), self.norm_fixed_scale, dtype=np.float32)
                sub_window_np = sub_window_np / scale

            sub_window_t = torch.from_numpy(sub_window_np).to(device=self.device, dtype=torch.float32)

            with torch.no_grad():
                if expected_dim in (108, 124):
                    bands_sub = self._filtered_bands_buffer[:, self.channel_perm, -self._bands_sub_window :]

                    if self._intrasubject_direct and rms_raw > 5.0:
                        for b_idx in range(3):
                            cur_std_b = np.maximum(np.std(bands_sub[b_idx], axis=-1, keepdims=True), 1.5).astype(
                                np.float32
                            )
                            self.channel_std[b_idx] = (0.998 * self.channel_std[b_idx] + 0.002 * cur_std_b).astype(
                                np.float32
                            )

                    if not self.is_calibrated and rms_raw > 5.0:
                        self.warmup_active_subwindows.append(bands_sub.copy())
                        if len(self.warmup_active_subwindows) >= self.warmup_target_count:
                            for b_idx in range(3):
                                active_b = np.concatenate([w[b_idx] for w in self.warmup_active_subwindows], axis=-1)
                                std_b = np.std(active_b, axis=-1, keepdims=True)
                                self.channel_std[b_idx] = np.maximum(std_b * 0.85, 3.0).astype(np.float32)
                            self.is_calibrated = True

                    has_f_stats = (
                        getattr(self.model, "feature_mean", None) is not None
                        and getattr(self.model, "feature_std", None) is not None
                    )
                    if has_f_stats:
                        f_mean = self.model.feature_mean.cpu().numpy().reshape(-1)
                        f_std = self.model.feature_std.cpu().numpy().reshape(-1)
                        safe_std = np.maximum(f_std, 0.05)
                    else:
                        f_mean = np.zeros(expected_dim, dtype=np.float32)
                        safe_std = np.ones(expected_dim, dtype=np.float32)

                    tangent_parts = []
                    eye_c = np.eye(self.channels, dtype=np.float32)

                    for b_idx in range(3):
                        b_sub = bands_sub[b_idx]
                        b_scaled = (b_sub - np.mean(b_sub, axis=-1, keepdims=True)) / self.channel_std[b_idx]
                        cov = _fast_ledoit_wolf(b_scaled) + 1e-6 * eye_c
                        wb = self.inv_sqrt_mean_cov[b_idx]
                        cov_w = wb @ cov @ wb.T

                        w_evals, w_evecs = np.linalg.eigh(cov_w)
                        w_evals = np.maximum(w_evals, 1e-8)
                        log_cov = w_evecs @ (np.log(w_evals)[:, None] * w_evecs.T)
                        tan_b = (log_cov[self._row, self._col] * self._triu_weights).astype(np.float32)
                        tangent_parts.append(tan_b)

                    tan_vec = np.concatenate(tangent_parts, axis=-1)

                    if expected_dim == 124:
                        b0_sub = bands_sub[0]
                        b0_scaled = (b0_sub - np.mean(b0_sub, axis=-1, keepdims=True)) / self.channel_std[0]
                        log_rms = np.log(np.sqrt(np.mean(b0_scaled**2, axis=-1)) + 1e-4).astype(np.float32)
                        if self._prev_log_rms is None:
                            move_cmd = np.zeros_like(log_rms)
                        else:
                            move_cmd = np.maximum(0.0, log_rms - self._prev_log_rms).astype(np.float32)
                        self._prev_log_rms = log_rms
                        tan_vec = np.concatenate([tan_vec, log_rms, move_cmd], axis=-1)

                    f_norm = np.clip((tan_vec - f_mean) / safe_std, -4.0, 4.0) if has_f_stats else tan_vec

                    tangent_vec_108 = torch.from_numpy(f_norm).to(device=self.device, dtype=torch.float32)
                    self._fifo_tangent_108 = torch.roll(self._fifo_tangent_108, -1, dims=1)
                    self._fifo_tangent_108[0, -1] = tangent_vec_108

                    tangent_vec = tangent_vec_108[: self.tangent_dim]
                    self._fifo_tangent_seq = torch.roll(self._fifo_tangent_seq, -1, dims=1)
                    self._fifo_tangent_seq[0, -1] = tangent_vec
                    angles_tensor = self.model.forward_from_tangent(self._fifo_tangent_108)
                else:
                    tangent_vec = self.model.tangent_space.compute_single_subwindow(sub_window_t)
                    self._fifo_tangent_seq = torch.roll(self._fifo_tangent_seq, -1, dims=1)
                    self._fifo_tangent_seq[0, -1] = tangent_vec
                    angles_tensor = self.model.forward_from_tangent(self._fifo_tangent_seq)

                raw_angles_deg = angles_tensor.squeeze().cpu().numpy().astype(np.float32)
                raw_angles_deg = np.clip(raw_angles_deg, 0.0, 180.0)

                if getattr(self.model, "biomechanical_dip_coupling", False) and raw_angles_deg.size == 15:
                    for pip_i, dip_i in ((4, 5), (7, 8), (10, 11), (13, 14)):
                        coupled_dip = 180.0 - 0.68 * (180.0 - raw_angles_deg[pip_i])
                        raw_angles_deg[dip_i] = float(
                            np.clip(0.75 * raw_angles_deg[dip_i] + 0.25 * coupled_dip, 0.0, 180.0)
                        )

                if self._full_manifold_calibrated and not self._intrasubject_direct:
                    self._smoothed_angles = raw_angles_deg.copy()
                elif self._smoothed_angles is None:
                    self._smoothed_angles = raw_angles_deg.copy()
                else:
                    self._smoothed_angles = (0.35 * raw_angles_deg + 0.65 * self._smoothed_angles).astype(np.float32)

                angles_deg = np.clip(self._smoothed_angles, 0.0, 180.0).astype(np.float32)

                self._poll_control_socket()
                self._step_interactive_calibration(raw_sub, rms_per_ch, angles_deg)

            t_compute_end = time.perf_counter()
            compute_latency_ms = (t_compute_end - t_start) * 1000.0

            kin_cfg = self.params.get("kinematics", {})
            st_mode = kin_cfg.get("streaming_mode", kin_cfg.get("mode", "empirical"))
            kin_params = {**self.params, "kinematics": {**kin_cfg, "mode": st_mode}}
            raw_controls = map_angles_to_controls(angles_deg, params=kin_params)

            smoothing_alpha = float(np.clip(kin_cfg.get("smoothing_alpha", 0.35), 0.01, 1.0))
            if self._smoothed_controls is None:
                self._smoothed_controls = raw_controls.copy()
            else:
                self._smoothed_controls = (
                    smoothing_alpha * raw_controls + (1.0 - smoothing_alpha) * self._smoothed_controls
                )
            controls = np.clip(self._smoothed_controls, 0.0, 1.0).astype(np.float32)

            if transmit and self.udp_sender is not None:
                self.udp_sender.send(controls)

            t_udp_sent = time.perf_counter()
            if abs(t_start - t_acq) <= 60.0 and (t_udp_sent >= t_acq):
                transport_latency_ms = (t_udp_sent - t_acq) * 1000.0
            else:
                transport_latency_ms = (t_udp_sent - t_start) * 1000.0

            self.latency_monitor.record(compute_ms=compute_latency_ms, transport_ms=transport_latency_ms)
            self._steps_completed += 1
            self._samples_processed += n_samples
            self._latest_angles = angles_deg
            self._latest_controls = controls
            self._latest_compute_latency_ms = compute_latency_ms
            self._latest_transport_latency_ms = transport_latency_ms

            tangent_np = tangent_vec.detach().cpu().numpy().astype(np.float32)
            result = StepResult(
                angles=angles_deg,
                controls=controls,
                compute_latency_ms=compute_latency_ms,
                transport_latency_ms=transport_latency_ms,
                tangent_vector=tangent_np,
            )
            self._latest_step_result = result
            return result

    def _run_loop(self) -> None:
        _set_windows_timer_resolution(1)
        target_dt = 1.0 / self.cadence_hz
        t_start = time.perf_counter()
        step_idx = 0

        try:
            while not self._stop_event.is_set():
                self._poll_control_socket()
                try:
                    raw_chunk, timestamps = self.ring_buffer.pop_chunk(self.step_size, timeout=0.1)
                except TimeoutError:
                    if self._stop_event.is_set():
                        break
                    continue
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    logger.error("Error popping from ring buffer in worker loop: %s", exc)
                    self._last_error = exc
                    continue

                if self._stop_event.is_set():
                    break

                try:
                    self.step(raw_chunk=raw_chunk, timestamps=timestamps, transmit=True)
                except Exception as exc:
                    logger.error("Error executing inference step in worker loop: %s", exc)
                    self._last_error = exc
                    if self._stop_event.is_set():
                        break

                step_idx += 1

                if self.pace_loop:
                    target_time = t_start + step_idx * target_dt
                    now = time.perf_counter()
                    if now - target_time > 0.5:
                        t_start = now - step_idx * target_dt
                        target_time = now

                    rem = target_time - now
                    if rem > 0.0015 and self._stop_event.wait(timeout=rem - 0.001):
                        break
                    while time.perf_counter() < target_time and not self._stop_event.is_set():
                        pass
        finally:
            _reset_windows_timer_resolution(1)

    def start(self) -> None:
        if self.is_alive():
            return
        self._stop_event.clear()
        self._start_control_socket()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="StreamingInferenceWorker")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._stop_control_socket()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        self.stop()
        if self.udp_sender is not None:
            try:
                self.udp_sender.close()
            except Exception:
                pass

    def __enter__(self) -> StreamingInferenceWorker:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
