"""Interactive Intra-Subject Myo sEMG + MediaPipe Hand Recording Studio & Dataset Manager.

Features:
- Open and inspect any existing `.npz` recording dataset with per-session quality metrics
  (duration, sample count, EMG RMS activity, hand motion coverage, joint range of motion).
- Record new synchronized 200 Hz Myo sEMG + 3D MediaPipe hand sessions with a get-ready
  countdown, live pause/resume (`SPACE`), prompt navigation (`N` / `R`), and sinusoidal pace guide.
- Choose to Append to or Overwrite an existing recording, or Save As a new `.npz` file.
- Post-session Review Dialog after every recording with session quality diagnostics and a
  protected Discard confirmation ("Are you sure you want to discard the recording?").
- Session management tools: visually plot/inspect individual sessions, import/merge `.npz`
  batches, remove specific sessions (with automatic backup), and trigger model training directly.

Usage (Launch Interactive Recording & Dataset Manager Studio):
    .venv\\Scripts\\python.exe scripts/record_intrasubject_myo.py

Usage (Direct CLI Recording with Post-Session Save/Discard Review):
    .venv\\Scripts\\python.exe scripts/record_intrasubject_myo.py --direct --duration 600

Usage (Start fresh and overwrite existing dataset):
    .venv\\Scripts\\python.exe scripts/record_intrasubject_myo.py --direct --duration 600 --overwrite

Usage (Automated Synthetic Verification Test):
    .venv\\Scripts\\python.exe scripts/record_intrasubject_myo.py --synthetic-test
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
import tkinter as tk
import urllib.request
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stream.ble_myo import BleMyoClient  # noqa: E402

# 15 joint angle triplets (parent, joint, child) in MediaPipe 21-landmark hand graph:
# 0..2: Thumb (CMC, MCP, IP), 3..5: Index (MCP, PIP, DIP),
# 6..8: Middle (MCP, PIP, DIP), 9..11: Ring (MCP, PIP, DIP), 12..14: Pinky (MCP, PIP, DIP)
JOINT_TRIPLETS: list[tuple[int, int, int]] = [
    (0, 1, 2),  # 0: Thumb CMC
    (1, 2, 3),  # 1: Thumb MCP
    (2, 3, 4),  # 2: Thumb IP
    (0, 5, 6),  # 3: Index MCP
    (5, 6, 7),  # 4: Index PIP
    (6, 7, 8),  # 5: Index DIP
    (0, 9, 10),  # 6: Middle MCP
    (9, 10, 11),  # 7: Middle PIP
    (10, 11, 12),  # 8: Middle DIP
    (0, 13, 14),  # 9: Ring MCP
    (13, 14, 15),  # 10: Ring PIP
    (14, 15, 16),  # 11: Ring DIP
    (0, 17, 18),  # 12: Pinky MCP
    (17, 18, 19),  # 13: Pinky PIP
    (18, 19, 20),  # 14: Pinky DIP
]

HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),  # Pinky & Palm
]

GUIDED_PROMPTS: list[tuple[str, float]] = [
    ("OPEN HAND", 8.0),
    ("SPREAD FINGERS", 8.0),
    ("THUMB ONLY", 14.0),
    ("INDEX FINGER", 14.0),
    ("MIDDLE FINGER", 14.0),
    ("RING FINGER", 14.0),
    ("PINKY FINGER", 14.0),
    ("THUMB + INDEX PINCH", 12.0),
    ("INDEX + MIDDLE", 12.0),
    ("RING + PINKY", 12.0),
    ("OPEN & CLOSE HAND", 14.0),
    ("FINGER WAVE", 24.0),
]

GUIDED_COACHING_TIPS: dict[str, tuple[str, float]] = {
    "OPEN HAND": ("Relax hand facing camera", 0.0),
    "SPREAD FINGERS": ("Spread wide, then relax", 0.35),
    "THUMB ONLY": ("Bend & straighten thumb", 0.40),
    "INDEX FINGER": ("Bend & straighten index", 0.40),
    "MIDDLE FINGER": ("Bend & straighten middle", 0.40),
    "RING FINGER": ("Bend & straighten ring", 0.40),
    "PINKY FINGER": ("Bend & straighten pinky", 0.40),
    "THUMB + INDEX PINCH": ("Touch tips gently & open", 0.40),
    "INDEX + MIDDLE": ("Bend both together", 0.40),
    "RING + PINKY": ("Bend both together", 0.40),
    "OPEN & CLOSE HAND": ("Gentle full hand curl", 0.35),
    "FINGER WAVE": ("Roll fingers smoothly", 0.45),
}

# Landmarks to highlight on the webcam skeleton for each prompt
GUIDED_ACTIVE_LANDMARKS: dict[str, set[int]] = {
    "OPEN HAND": set(),
    "SPREAD FINGERS": set(range(1, 21)),
    "THUMB ONLY": {1, 2, 3, 4},
    "INDEX FINGER": {5, 6, 7, 8},
    "MIDDLE FINGER": {9, 10, 11, 12},
    "RING FINGER": {13, 14, 15, 16},
    "PINKY FINGER": {17, 18, 19, 20},
    "THUMB + INDEX PINCH": {1, 2, 3, 4, 5, 6, 7, 8},
    "INDEX + MIDDLE": {5, 6, 7, 8, 9, 10, 11, 12},
    "RING + PINKY": {13, 14, 15, 16, 17, 18, 19, 20},
    "OPEN & CLOSE HAND": set(range(1, 21)),
    "FINGER WAVE": set(range(1, 21)),
}

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_hand_landmarker_model() -> Path:
    """Ensure MediaPipe HandLandmarker .task model exists locally, downloading if necessary."""
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    task_path = models_dir / "hand_landmarker.task"
    if not task_path.is_file():
        print(f"Downloading hand_landmarker.task model from {HAND_LANDMARKER_MODEL_URL}...")
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, task_path)
        print(f"Downloaded model to {task_path} ({task_path.stat().st_size} bytes).")
    return task_path


def landmarks_to_15_angles(pts: np.ndarray) -> np.ndarray:
    """Compute 15 interior joint angles in degrees [0, 180] from (21, 3) landmarks.

    Applies monocular Z-depth damping (0.55x) and soft biomechanical DIP-PIP coupling
    to suppress single-camera depth jitter and fingertip self-occlusion noise.
    """
    pts_damped = np.asarray(pts, dtype=np.float32).copy()
    if pts_damped.ndim == 2 and pts_damped.shape[1] >= 3:
        pts_damped[:, 2] *= 0.55

    angles = np.zeros(15, dtype=np.float32)
    for idx, (a, b, c) in enumerate(JOINT_TRIPLETS):
        v1 = pts_damped[a] - pts_damped[b]
        v2 = pts_damped[c] - pts_damped[b]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            angles[idx] = 170.0
            continue
        cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angles[idx] = float(np.degrees(np.arccos(cos_theta)))

    # Softly regularize distal interphalangeal (DIP) angles with proximal (PIP) flexion
    # to reduce monocular fingertip occlusion spikes during full finger flexion
    for pip_i, dip_i in ((4, 5), (7, 8), (10, 11), (13, 14)):
        coupled_dip = 180.0 - 0.68 * (180.0 - angles[pip_i])
        angles[dip_i] = 0.75 * angles[dip_i] + 0.25 * coupled_dip

    return np.clip(angles, 30.0, 180.0)


def inspect_existing_dataset(out_path: Path) -> tuple[int, int, float]:
    """Return (n_samples, n_sessions, duration_minutes) for an existing .npz dataset."""
    if not out_path.is_file():
        return 0, 0, 0.0
    with np.load(out_path) as data:
        n_samples = int(data["emg"].shape[0])
        fs = int(data["fs"]) if "fs" in data else 200
        if "session_lengths" in data:
            n_sessions = int(len(data["session_lengths"]))
        else:
            n_sessions = 1 if n_samples > 0 else 0
    return n_samples, n_sessions, n_samples / float(max(fs, 1)) / 60.0


def compute_session_diagnostics(emg_seg: np.ndarray, targets_seg: np.ndarray, fs: int = 200) -> dict[str, float | str]:
    """Compute quality metrics for a recorded (emg, targets) segment."""
    n_samples = int(len(emg_seg))
    dur_sec = n_samples / float(max(fs, 1))
    if n_samples < 10:
        return {
            "duration_sec": dur_sec,
            "emg_rms": 0.0,
            "active_pct": 0.0,
            "rom_deg": 0.0,
            "moving_pct": 0.0,
            "status": "Empty",
        }

    emg_dc = emg_seg - np.mean(emg_seg, axis=0, keepdims=True)
    emg_rms = float(np.sqrt(np.mean(emg_dc**2)))

    # Compute active contraction percentage across 100-sample (0.5s) windows
    win = min(100, n_samples)
    step = max(1, win // 2)
    win_rms = [float(np.sqrt(np.mean(emg_dc[i : i + win] ** 2))) for i in range(0, max(1, n_samples - win + 1), step)]
    active_pct = 100.0 * float(np.mean([r > 2.5 for r in win_rms])) if win_rms else 0.0

    # Compute kinematic range of motion (5th to 95th percentile span across 15 joints)
    p95 = np.percentile(targets_seg, 95, axis=0)
    p05 = np.percentile(targets_seg, 5, axis=0)
    rom_deg = float(np.mean(p95 - p05))

    # Compute fraction of timeline where hand kinematics are actively changing (not frozen)
    if n_samples > fs:
        diff_1s = np.mean(np.abs(targets_seg[fs:] - targets_seg[:-fs]), axis=-1)
        moving_pct = 100.0 * float(np.mean(diff_1s > 0.25))
    else:
        moving_pct = 100.0 if rom_deg > 5.0 else 0.0

    if dur_sec < 30.0:
        status = "Short (<30s)"
    elif emg_rms < 1.2:
        status = "Weak EMG"
    elif moving_pct < 35.0 or rom_deg < 12.0:
        status = "Low Hand Motion"
    else:
        status = "Healthy"

    return {
        "duration_sec": dur_sec,
        "emg_rms": emg_rms,
        "active_pct": active_pct,
        "rom_deg": rom_deg,
        "moving_pct": moving_pct,
        "status": status,
    }


def inspect_dataset_sessions(npz_path: Path) -> dict[str, Any]:
    """Load an existing `.npz` recording and return full per-session breakdown and diagnostics."""
    if not npz_path.is_file():
        return {
            "exists": False,
            "path": npz_path,
            "total_samples": 0,
            "total_minutes": 0.0,
            "fs": 200,
            "sessions": [],
        }

    with np.load(npz_path) as data:
        emg = data["emg"].astype(np.float32)
        targets = data["targets"].astype(np.float32)
        fs = int(data["fs"]) if "fs" in data else 200
        if "session_lengths" in data:
            raw_lengths = [int(x) for x in data["session_lengths"] if int(x) > 0]
        else:
            raw_lengths = [int(len(emg))] if len(emg) > 0 else []

    total_samples = int(len(emg))
    if sum(raw_lengths) != total_samples and total_samples > 0:
        raw_lengths = [total_samples]

    sessions_info: list[dict[str, Any]] = []
    offset = 0
    for idx, s_len in enumerate(raw_lengths):
        e_seg = emg[offset : offset + s_len]
        t_seg = targets[offset : offset + s_len]
        diag = compute_session_diagnostics(e_seg, t_seg, fs=fs)
        sessions_info.append(
            {
                "index": idx + 1,
                "start_sample": offset,
                "end_sample": offset + s_len,
                "n_samples": s_len,
                "duration_sec": float(diag["duration_sec"]),
                "duration_min": float(diag["duration_sec"]) / 60.0,
                "emg_rms": float(diag["emg_rms"]),
                "active_pct": float(diag["active_pct"]),
                "rom_deg": float(diag["rom_deg"]),
                "moving_pct": float(diag["moving_pct"]),
                "status": str(diag["status"]),
            }
        )
        offset += s_len

    return {
        "exists": True,
        "path": npz_path,
        "total_samples": total_samples,
        "total_minutes": total_samples / float(max(fs, 1)) / 60.0,
        "fs": fs,
        "sessions": sessions_info,
    }


def remove_session_from_dataset(npz_path: Path, session_index_0based: int) -> dict[str, Any]:
    """Remove a specific session from `npz_path` (with backup) and return updated stats."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")

    with np.load(npz_path) as data:
        emg = data["emg"].astype(np.float32)
        targets = data["targets"].astype(np.float32)
        fs = int(data["fs"]) if "fs" in data else 200
        if "session_lengths" in data:
            lengths = [int(x) for x in data["session_lengths"] if int(x) > 0]
        else:
            lengths = [int(len(emg))]

    if sum(lengths) != len(emg):
        lengths = [int(len(emg))]

    if not (0 <= session_index_0based < len(lengths)):
        raise IndexError(f"Session index {session_index_0based} out of range (0..{len(lengths) - 1})")

    # Backup removed session before modifying dataset
    sessions_dir = npz_path.parent / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0, *lengths])
    s0, s1 = int(offsets[session_index_0based]), int(offsets[session_index_0based + 1])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    removed_backup = sessions_dir / f"removed_session{session_index_0based + 1}_{stamp}_{(s1 - s0) // max(fs, 1)}s.npz"
    np.savez_compressed(
        removed_backup,
        emg=emg[s0:s1],
        targets=targets[s0:s1],
        fs=np.int32(fs),
        session_lengths=np.array([s1 - s0], dtype=np.int64),
    )

    if len(lengths) == 1:
        npz_path.unlink()
        return {"total_samples": 0, "n_sessions": 0, "total_minutes": 0.0, "backup": str(removed_backup)}

    emg_new = np.concatenate([emg[:s0], emg[s1:]], axis=0)
    targets_new = np.concatenate([targets[:s0], targets[s1:]], axis=0)
    new_lengths = [s_len for i, s_len in enumerate(lengths) if i != session_index_0based]

    tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        emg=emg_new,
        targets=targets_new,
        fs=np.int32(fs),
        session_lengths=np.asarray(new_lengths, dtype=np.int64),
    )
    tmp_path.replace(npz_path)

    return {
        "total_samples": int(len(emg_new)),
        "n_sessions": len(new_lengths),
        "total_minutes": len(emg_new) / float(max(fs, 1)) / 60.0,
        "backup": str(removed_backup),
    }


def save_or_append_dataset(
    out_path: Path,
    new_emg: np.ndarray,
    new_targets: np.ndarray,
    fs: int = 200,
    append: bool = True,
    save_session_backup: bool = True,
) -> dict[str, int | float]:
    """Save or append a newly recorded (emg, targets) batch to `out_path` safely."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sessions_dir = out_path.parent / "sessions"
    new_len = int(len(new_emg))

    if save_session_backup:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_backup = sessions_dir / f"session_{stamp}_{new_len // max(fs, 1)}s.npz"
        np.savez_compressed(
            batch_backup,
            emg=new_emg.astype(np.float32),
            targets=new_targets.astype(np.float32),
            fs=np.int32(fs),
            session_lengths=np.array([new_len], dtype=np.int64),
        )
        print(f"Saved individual session backup to {batch_backup}")

    if append and out_path.is_file():
        with np.load(out_path) as prev:
            prev_emg = prev["emg"].astype(np.float32)
            prev_targets = prev["targets"].astype(np.float32)
            has_session_lengths = "session_lengths" in prev
            if has_session_lengths:
                prev_lengths = prev["session_lengths"].astype(np.int64).tolist()
            else:
                prev_lengths = [int(len(prev_emg))]

        # If upgrading a single-session file without session_lengths, preserve a backup of it first
        if save_session_backup and not has_session_lengths and len(prev_emg) > 0:
            initial_backup = sessions_dir / "session_000_initial_backup.npz"
            if not initial_backup.is_file():
                shutil.copy2(out_path, initial_backup)
                print(f"Backed up initial pre-append dataset to {initial_backup}")

        emg_combined = np.concatenate([prev_emg, new_emg.astype(np.float32)], axis=0)
        targets_combined = np.concatenate([prev_targets, new_targets.astype(np.float32)], axis=0)
        session_lengths = np.array([*prev_lengths, new_len], dtype=np.int64)
        prev_total = int(len(prev_emg))
    else:
        # If overwriting an existing dataset, back up the previous file first so no work is ever lost
        if save_session_backup and not append and out_path.is_file():
            sessions_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            overwrite_bak = sessions_dir / f"pre_overwrite_backup_{stamp}.npz"
            shutil.copy2(out_path, overwrite_bak)
            print(f"Backed up previous dataset before overwrite to {overwrite_bak}")

        emg_combined = new_emg.astype(np.float32)
        targets_combined = new_targets.astype(np.float32)
        session_lengths = np.array([new_len], dtype=np.int64)
        prev_total = 0

    # Write atomically via a temporary file so an interrupted write never corrupts existing data
    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        emg=emg_combined,
        targets=targets_combined,
        fs=np.int32(fs),
        session_lengths=session_lengths,
    )
    tmp_path.replace(out_path)

    total_samples = int(len(emg_combined))
    total_min = total_samples / float(max(fs, 1)) / 60.0
    added_min = new_len / float(max(fs, 1)) / 60.0
    n_sessions = int(len(session_lengths))

    if prev_total > 0:
        prev_min = prev_total / float(max(fs, 1)) / 60.0
        print(
            f"Expanded dataset at {out_path}:\n"
            f"  Previous : {prev_total:,} samples ({prev_min:.1f} min)\n"
            f"  Added    : +{new_len:,} samples (+{added_min:.1f} min)\n"
            f"  Total    : {total_samples:,} samples ({total_min:.1f} min across {n_sessions} sessions)"
        )
    else:
        print(
            f"Saved dataset to {out_path}: {total_samples:,} samples ({total_min:.1f} min across {n_sessions} session)"
        )

    return {
        "prev_samples": prev_total,
        "added_samples": new_len,
        "total_samples": total_samples,
        "n_sessions": n_sessions,
        "total_minutes": total_min,
    }


def confirm_discard_recording(parent: tk.Misc | None = None, duration_sec: float = 0.0) -> bool:
    """Ask the user 'Are you sure you want to discard the recording?' before discarding."""
    own_root = False
    if parent is None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        parent = root
        own_root = True

    detail = (
        f"\n\nThis will permanently discard the newly recorded {duration_sec:.1f}s session."
        if duration_sec > 0.0
        else ""
    )
    confirmed = bool(
        messagebox.askyesno(
            title="Discard Recording?",
            message=f"Are you sure you want to discard the recording?{detail}",
            icon=messagebox.WARNING,
            parent=parent,
        )
    )
    if own_root and isinstance(parent, tk.Tk):
        parent.destroy()
    return confirmed


def prompt_post_session_review(
    emg_all: np.ndarray,
    targets_200: np.ndarray,
    out_path: Path,
    default_append: bool = True,
    hand_detect_pct: float = 100.0,
    fs: int = 200,
    parent: tk.Misc | None = None,
) -> dict[str, Any]:
    """Display a Post-Session Review dialog allowing the user to Save (Append/Overwrite/Save As) or Discard.

    When 'Discard Recording' is clicked, prompts 'Are you sure you want to discard the recording?'.
    """
    diag = compute_session_diagnostics(emg_all, targets_200, fs=fs)
    prev_samples, prev_sessions, prev_min = inspect_existing_dataset(out_path)

    own_root = False
    if parent is None:
        dlg = tk.Tk()
        dlg.title("Review Recorded Session — Save or Discard")
        own_root = True
    else:
        dlg = tk.Toplevel(parent)
        dlg.title("Review Recorded Session — Save or Discard")
        dlg.transient(parent)
        dlg.grab_set()

    dlg.configure(bg="#141418")
    dlg.geometry("620x500")
    dlg.resizable(False, False)
    dlg.attributes("-topmost", True)

    decision: dict[str, Any] = {
        "action": "discarded",
        "out_path": out_path,
        "append": default_append,
        "result": None,
    }

    # Header
    hdr = tk.Frame(dlg, bg="#1c1c24", padx=20, pady=14)
    hdr.pack(fill=tk.X)
    tk.Label(
        hdr,
        text="Recording Complete — Review Session Quality",
        bg="#1c1c24",
        fg="#00e676",
        font=("Segoe UI", 13, "bold"),
    ).pack(anchor=tk.W)
    tk.Label(
        hdr,
        text="Inspect the captured session metrics below and choose whether to save or discard.",
        bg="#1c1c24",
        fg="#a0a0b4",
        font=("Segoe UI", 9),
    ).pack(anchor=tk.W, pady=(2, 0))

    # Metrics summary card
    card = tk.Frame(dlg, bg="#1e1e28", padx=18, pady=12, highlightbackground="#2e2e3e", highlightthickness=1)
    card.pack(fill=tk.X, padx=20, pady=14)

    dur_s = float(diag["duration_sec"])
    n_samp = int(len(emg_all))
    status_str = str(diag["status"])
    status_color = "#00e676" if status_str == "Healthy" else "#ffab00"

    metrics_rows = [
        ("Recorded Duration:", f"{dur_s:.1f} sec ({dur_s / 60.0:.2f} min)  |  {n_samp:,} EMG samples @ {fs} Hz"),
        ("Webcam Hand Tracking:", f"{hand_detect_pct:.1f}% of video frames detected"),
        (
            "Forearm EMG Activity:",
            f"RMS = {float(diag['emg_rms']):.2f}  |  Active windows = {float(diag['active_pct']):.1f}%",
        ),
        (
            "Finger Range of Motion:",
            f"Mean ROM = {float(diag['rom_deg']):.1f}°  |  Dynamic motion = {float(diag['moving_pct']):.1f}%",
        ),
        ("Session Quality Check:", status_str),
    ]
    for r_idx, (k_txt, v_txt) in enumerate(metrics_rows):
        tk.Label(card, text=k_txt, bg="#1e1e28", fg="#9a9ab0", font=("Segoe UI", 9, "bold")).grid(
            row=r_idx, column=0, sticky=tk.W, pady=3
        )
        tk.Label(
            card,
            text=v_txt,
            bg="#1e1e28",
            fg=status_color if r_idx == len(metrics_rows) - 1 else "#f0f0f8",
            font=("Segoe UI", 9, "bold" if r_idx == len(metrics_rows) - 1 else "normal"),
        ).grid(row=r_idx, column=1, sticky=tk.W, padx=(14, 0), pady=3)

    # Save Mode Selection
    mode_box = tk.Frame(dlg, bg="#141418", padx=20)
    mode_box.pack(fill=tk.X, pady=(2, 8))

    tk.Label(
        mode_box,
        text="Destination & Save Mode:",
        bg="#141418",
        fg="#00e5ff",
        font=("Segoe UI", 10, "bold"),
    ).pack(anchor=tk.W, pady=(0, 6))

    path_var = tk.StringVar(value=str(out_path))
    path_row = tk.Frame(mode_box, bg="#141418")
    path_row.pack(fill=tk.X, pady=(0, 8))

    path_entry = tk.Entry(
        path_row,
        textvariable=path_var,
        bg="#1e1e28",
        fg="#f0f0f8",
        insertbackground="#ffffff",
        relief=tk.FLAT,
        font=("Consolas", 9),
    )
    path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 8))

    def _browse_target() -> None:
        chosen = filedialog.asksaveasfilename(
            parent=dlg,
            title="Select Target Dataset (.npz)",
            initialdir=str(out_path.parent),
            initialfile=out_path.name,
            defaultextension=".npz",
            filetypes=[("NumPy Compressed Dataset", "*.npz")],
        )
        if chosen:
            path_var.set(chosen)

    tk.Button(
        path_row,
        text="Browse...",
        command=_browse_target,
        bg="#2a2a38",
        fg="#ffffff",
        relief=tk.FLAT,
        font=("Segoe UI", 9),
        padx=10,
        pady=3,
    ).pack(side=tk.RIGHT)

    save_mode_var = tk.StringVar(value="append" if default_append else "overwrite")
    existing_note = (
        f"(Current file has {prev_min:.1f} min across {prev_sessions} session(s))"
        if prev_samples > 0
        else "(Target file does not exist yet — will create new)"
    )

    tk.Radiobutton(
        mode_box,
        text=f"Append to existing recording dataset {existing_note}",
        variable=save_mode_var,
        value="append",
        bg="#141418",
        fg="#f0f0f8",
        selectcolor="#2a2a38",
        activebackground="#141418",
        activeforeground="#00e676",
        font=("Segoe UI", 9),
    ).pack(anchor=tk.W, pady=2)

    tk.Radiobutton(
        mode_box,
        text="Overwrite existing recording dataset (replaces existing sessions; backup kept in sessions/)",
        variable=save_mode_var,
        value="overwrite",
        bg="#141418",
        fg="#f0f0f8",
        selectcolor="#2a2a38",
        activebackground="#141418",
        activeforeground="#ffab00",
        font=("Segoe UI", 9),
    ).pack(anchor=tk.W, pady=2)

    # Action Buttons
    btn_row = tk.Frame(dlg, bg="#141418", padx=20, pady=14)
    btn_row.pack(fill=tk.X, side=tk.BOTTOM)

    def _on_save() -> None:
        target_p = Path(path_var.get().strip())
        do_append = save_mode_var.get() == "append"
        if not do_append and target_p.is_file():
            ex_s, ex_n, ex_m = inspect_existing_dataset(target_p)
            if ex_s > 0:
                ok = messagebox.askyesno(
                    title="Confirm Overwrite",
                    message=(
                        f"'{target_p.name}' currently contains {ex_m:.1f} min ({ex_n} session(s)).\n\n"
                        f"Are you sure you want to OVERWRITE it with this new {dur_s:.1f}s session?\n"
                        f"(A backup of the previous file will be saved in sessions/.)"
                    ),
                    icon=messagebox.WARNING,
                    parent=dlg,
                )
                if not ok:
                    return
        res = save_or_append_dataset(
            target_p,
            emg_all,
            targets_200,
            fs=fs,
            append=do_append,
            save_session_backup=True,
        )
        decision["action"] = "saved"
        decision["out_path"] = target_p
        decision["append"] = do_append
        decision["result"] = res
        dlg.destroy()

    def _on_discard() -> None:
        if confirm_discard_recording(parent=dlg, duration_sec=dur_s):
            print("Recording discarded by user.")
            decision["action"] = "discarded"
            dlg.destroy()

    tk.Button(
        btn_row,
        text="🗑 Discard Recording",
        command=_on_discard,
        bg="#5a1e28",
        fg="#ffffff",
        activebackground="#7a2634",
        activeforeground="#ffffff",
        relief=tk.FLAT,
        font=("Segoe UI", 10, "bold"),
        padx=16,
        pady=8,
    ).pack(side=tk.LEFT)

    tk.Button(
        btn_row,
        text="💾 Save Recording",
        command=_on_save,
        bg="#00a859",
        fg="#ffffff",
        activebackground="#00c86b",
        activeforeground="#ffffff",
        relief=tk.FLAT,
        font=("Segoe UI", 10, "bold"),
        padx=22,
        pady=8,
    ).pack(side=tk.RIGHT)

    dlg.protocol("WM_DELETE_WINDOW", _on_discard)
    if own_root:
        dlg.mainloop()
    else:
        dlg.wait_window()
    return decision


def generate_synthetic_recording(out_path: Path, duration_sec: float = 24.0, append: bool = False) -> None:
    """Generate a realistic synthetic 200 Hz 8-channel sEMG + 15-DOF kinematics dataset."""
    fs = 200
    total_samples = int(duration_sec * fs)
    rng = np.random.default_rng(42)
    t = np.arange(total_samples, dtype=np.float32) / float(fs)

    emg = rng.normal(0.0, 0.8, size=(total_samples, 8)).astype(np.float32)
    open_pose = np.array(
        [168.0, 172.0, 172.0, 168.0, 172.0, 174.0, 168.0, 172.0, 174.0, 166.0, 170.0, 172.0, 160.0, 168.0, 172.0],
        dtype=np.float32,
    )
    targets = np.tile(open_pose, (total_samples, 1))

    # Cycle through 6 distinct regimes (Rest, Thumb, Index, Middle, Ring, Pinky, Full Fist)
    carriers = (
        np.sin(2.0 * np.pi * 28.0 * t)[:, None]
        + 0.8 * np.sin(2.0 * np.pi * 52.0 * t + 0.5)[:, None]
        + 0.6 * np.sin(2.0 * np.pi * 78.0 * t + 1.1)[:, None]
    ).astype(np.float32)

    seg_len = 400  # 2.0 seconds per segment
    n_segs = total_samples // seg_len
    for s_idx in range(n_segs):
        s0 = s_idx * seg_len
        s1 = s0 + seg_len
        mode = s_idx % 7
        env = np.sin(np.linspace(0.0, np.pi, seg_len, dtype=np.float32))[:, None]
        if mode == 0:
            pass
        elif mode == 1:
            emg[s0:s1, 1:2] += 24.0 * env * carriers[s0:s1]
            emg[s0:s1, 6:7] += 10.0 * env * carriers[s0:s1]
            targets[s0:s1, 0:3] -= env * np.array([45.0, 65.0, 80.0], dtype=np.float32)
        elif mode == 2:
            emg[s0:s1, 6:7] += 28.0 * env * carriers[s0:s1]
            emg[s0:s1, 2:3] += 11.0 * env * carriers[s0:s1]
            targets[s0:s1, 3:6] -= env * np.array([70.0, 110.0, 75.0], dtype=np.float32)
        elif mode == 3:
            emg[s0:s1, 2:3] += 28.0 * env * carriers[s0:s1]
            targets[s0:s1, 6:9] -= env * np.array([75.0, 115.0, 75.0], dtype=np.float32)
        elif mode == 4:
            emg[s0:s1, 3:4] += 28.0 * env * carriers[s0:s1]
            targets[s0:s1, 9:12] -= env * np.array([75.0, 115.0, 75.0], dtype=np.float32)
        elif mode == 5:
            emg[s0:s1, 7:8] += 28.0 * env * carriers[s0:s1]
            targets[s0:s1, 12:15] -= env * np.array([75.0, 115.0, 75.0], dtype=np.float32)
        elif mode == 6:
            for c in (3, 4, 5, 6):
                emg[s0:s1, c : c + 1] += 25.0 * env * carriers[s0:s1]
            targets[s0:s1, :] -= env * np.array(
                [45.0, 65.0, 80.0, 70.0, 110.0, 75.0, 75.0, 115.0, 75.0, 75.0, 115.0, 75.0, 75.0, 115.0, 75.0],
                dtype=np.float32,
            )

    save_or_append_dataset(
        out_path,
        emg,
        targets,
        fs=fs,
        append=append,
        save_session_backup=False,
    )


def record_live_session(
    out_path: Path,
    duration_sec: float,
    camera_id: int = 0,
    append: bool = True,
    countdown_sec: float = 3.0,
    interactive_review: bool = True,
    parent: tk.Misc | None = None,
) -> dict[str, Any]:
    """Record live 200 Hz Myo sEMG and MediaPipe webcam hand kinematics.

    Supports live Pause/Resume (`SPACE`), Next Prompt (`N`), Stop & Review (`Q`/`ESC`),
    and a Post-Session Save/Discard dialog with discard confirmation.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    prev_samples, prev_sessions, prev_minutes = inspect_existing_dataset(out_path) if append else (0, 0, 0.0)
    if prev_samples > 0:
        print(
            f"Found existing dataset at {out_path}: {prev_samples:,} samples "
            f"({prev_minutes:.1f} min across {prev_sessions} session(s)).\n"
            f"Default save mode: {'APPEND' if append else 'OVERWRITE'}."
        )
    else:
        mode_label = "Starting new dataset" if append else "Overwriting dataset (--overwrite)"
        print(f"{mode_label} at {out_path}.")

    model_asset_path = ensure_hand_landmarker_model()
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_asset_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    client = BleMyoClient(force_mock=False, mock_fallback=False, share_connection=True)
    print("Connecting to Myo armband (or shared BLE hub)...")
    client.start(wait_ready=True, timeout=30.0)

    print("Waiting for active 200 Hz EMG packets from Myo armband...")
    t_wait = time.perf_counter()
    while len(client.ring_buffer) == 0:
        if (time.perf_counter() - t_wait) > 60.0:
            client.stop()
            landmarker.close()
            raise RuntimeError("Timed out (60s) waiting for Myo EMG stream to begin.")
        time.sleep(0.02)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        client.stop()
        landmarker.close()
        raise RuntimeError(f"Could not open webcam camera_id={camera_id}")

    last_ts_ms = -1

    # Optional Get-Ready Countdown so user can position hand before t_start = 0
    if countdown_sec > 0.0:
        t_cd_start = time.perf_counter()
        while (time.perf_counter() - t_cd_start) < countdown_sec:
            rem_cd = max(0.0, countdown_sec - (time.perf_counter() - t_cd_start))
            if len(client.ring_buffer) > 0:
                client.ring_buffer.clear()
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = max(int((time.perf_counter() - t_cd_start) * 1000.0), last_ts_ms + 1)
            last_ts_ms = ts_ms
            res = landmarker.detect_for_video(mp_img, ts_ms)
            if res.hand_landmarks:
                lms = res.hand_landmarks[0]
                h, w = frame.shape[:2]
                pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                for s_i, e_i in HAND_CONNECTIONS:
                    if s_i < len(pixel_pts) and e_i < len(pixel_pts):
                        cv2.line(frame, pixel_pts[s_i], pixel_pts[e_i], (0, 255, 120), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 76), (20, 20, 24), -1)
            cv2.putText(
                frame,
                f"GET READY — SHOW PALM ({rem_cd:3.1f}s)",
                (18, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 230, 255),
                2,
            )
            cv2.imshow("BRI Intra-Subject Myo + MediaPipe Recorder", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    # Clear any stale EMG samples buffered before t_start
    client.ring_buffer.clear()

    emg_chunks: list[np.ndarray] = []
    emg_sample_times: list[np.ndarray] = []
    kin_frames: list[np.ndarray] = []
    kin_times: list[float] = []

    last_angles = np.full(15, 170.0, dtype=np.float32)
    t_start = time.perf_counter()
    last_emg_t = 0.0
    cycle_len = sum(d for _, d in GUIDED_PROMPTS)
    prompt_time_offset = 0.0
    is_paused = False
    pause_started_at = 0.0
    detected_frames = 0
    total_frames = 0

    print(
        f"Recording started for {duration_sec:.0f}s. "
        f"Controls: [SPACE]=Pause/Resume | [N]=Next Prompt | [Q/ESC]=Finish & Review."
    )
    try:
        while True:
            wall_now = time.perf_counter()
            if is_paused:
                # Drain ring buffer while paused so stale samples never accumulate
                if len(client.ring_buffer) > 0:
                    client.ring_buffer.clear()
                ret, frame = cap.read()
                if ret:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 76), (25, 25, 35), -1)
                    cv2.putText(
                        frame,
                        "PAUSED — [SPACE] Resume   [Q] Finish",
                        (18, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 215, 255),
                        2,
                    )
                    cv2.imshow("BRI Intra-Subject Myo + MediaPipe Recorder", frame)
                key = cv2.waitKey(15) & 0xFF
                if key == ord(" "):
                    paused_dt = time.perf_counter() - pause_started_at
                    t_start += paused_dt
                    client.ring_buffer.clear()
                    is_paused = False
                elif key in (ord("q"), 27):
                    break
                continue

            now = wall_now - t_start
            if now >= duration_sec:
                break

            # Drain all available Myo samples and stamp each sample with its wall-clock arrival window
            n_avail = len(client.ring_buffer)
            if n_avail > 0:
                chunk, _ = client.ring_buffer.pop_chunk(n_avail, timeout=0.005)
                n_got = int(chunk.shape[1])
                if n_got > 0:
                    t0_chunk = max(last_emg_t, now - (n_got / 200.0))
                    chunk_t = np.linspace(t0_chunk, now, n_got, dtype=np.float64)
                    last_emg_t = now
                    chunk_T = chunk.T.astype(np.float32)
                    emg_chunks.append(chunk_T)
                    emg_sample_times.append(chunk_t)

            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Determine active prompt first so we can highlight the active finger(s) on the hand skeleton
            c_pos = (now + prompt_time_offset) % cycle_len
            acc = 0.0
            active_idx = len(GUIDED_PROMPTS) - 1
            active_label = GUIDED_PROMPTS[-1][0]
            rem_sec = 0.0
            prompt_local_t = 0.0
            for idx_p, (p_label, p_dur) in enumerate(GUIDED_PROMPTS):
                if c_pos <= acc + p_dur:
                    active_idx = idx_p
                    active_label = p_label
                    rem_sec = acc + p_dur - c_pos
                    prompt_local_t = c_pos - acc
                    break
                acc += p_dur

            next_label = GUIDED_PROMPTS[(active_idx + 1) % len(GUIDED_PROMPTS)][0]
            active_lms = GUIDED_ACTIVE_LANDMARKS.get(active_label, set())

            # Strictly monotonically increasing millisecond timestamp for detect_for_video
            ts_ms = max(int(time.perf_counter() * 1000.0), last_ts_ms + 1)
            last_ts_ms = ts_ms

            res = landmarker.detect_for_video(mp_img, ts_ms)
            hand_detected = bool(res.hand_landmarks)
            total_frames += 1
            if hand_detected:
                detected_frames += 1
                lms = res.hand_landmarks[0]
                h, w = frame.shape[:2]
                pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

                for s_i, e_i in HAND_CONNECTIONS:
                    if s_i < len(pixel_pts) and e_i < len(pixel_pts):
                        is_active_bone = (e_i in active_lms) or (s_i in active_lms and s_i != 0)
                        color = (0, 230, 255) if is_active_bone else (70, 150, 90)
                        thickness = 3 if is_active_bone else 2
                        cv2.line(frame, pixel_pts[s_i], pixel_pts[e_i], color, thickness)

                for lm_i, (px, py) in enumerate(pixel_pts):
                    is_active_joint = lm_i in active_lms
                    j_color = (0, 255, 255) if is_active_joint else (90, 160, 110)
                    j_radius = 5 if is_active_joint else 3
                    cv2.circle(frame, (px, py), j_radius, j_color, -1)

                pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                raw_frame_angles = landmarks_to_15_angles(pts)
                last_angles = (0.70 * raw_frame_angles + 0.30 * last_angles).astype(np.float32)

            kin_frames.append(last_angles.copy())
            kin_times.append(time.perf_counter() - t_start)

            tip_text, pace_hz = GUIDED_COACHING_TIPS.get(
                active_label,
                ("Move smoothly", 0.40),
            )

            # Top Guided Banner: 1 large action title + rhythm bar with 2-word action cue
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 84), (18, 18, 24), -1)
            cv2.putText(
                frame,
                f"{active_label}   ({int(np.ceil(rem_sec))}s)",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.88,
                (80, 255, 130),
                2,
            )

            bar_x0, bar_y0, bar_w, bar_h = 18, 52, 220, 18
            cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (80, 80, 95), 1)
            if pace_hz > 0.0:
                phase = 0.5 * (1.0 - float(np.cos(2.0 * np.pi * pace_hz * prompt_local_t)))
                fill_w = int(bar_w * phase)
                cv2.rectangle(
                    frame,
                    (bar_x0 + 1, bar_y0 + 1),
                    (bar_x0 + max(1, fill_w), bar_y0 + bar_h - 1),
                    (0, 215, 255),
                    -1,
                )
                pace_cue = "BEND IN" if np.sin(2.0 * np.pi * pace_hz * prompt_local_t) >= 0 else "STRAIGHTEN"
            else:
                pace_cue = "RELAX HAND"

            if rem_sec <= 2.5:
                sub_text = f"NEXT: {next_label}"
                sub_color = (0, 230, 255)
            else:
                sub_text = f"{pace_cue}  -  {tip_text}"
                sub_color = (225, 235, 245)

            cv2.putText(
                frame,
                sub_text,
                (bar_x0 + bar_w + 14, bar_y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                sub_color,
                2 if rem_sec <= 2.5 else 1,
            )

            # Bottom Status Bar: keeps timer & shortcuts out of the way of the main prompt
            fh, fw = frame.shape[:2]
            if not hand_detected:
                cv2.rectangle(frame, (0, fh - 36), (fw, fh), (0, 0, 180), -1)
                cv2.putText(
                    frame,
                    "SHOW HAND TO CAMERA",
                    (18, fh - 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
            else:
                cv2.rectangle(frame, (0, fh - 30), (fw, fh), (18, 18, 24), -1)
                el_m, el_s = divmod(int(now), 60)
                tot_m, tot_s = divmod(int(duration_sec), 60)
                cv2.putText(
                    frame,
                    f"{el_m:02d}:{el_s:02d} / {tot_m:02d}:{tot_s:02d}   |   [SPACE] Pause   [N] Skip   [Q] Finish",
                    (18, fh - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (185, 190, 205),
                    1,
                )
            cv2.imshow("BRI Intra-Subject Myo + MediaPipe Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                is_paused = True
                pause_started_at = time.perf_counter()
            elif key in (ord("n"), ord("N")):
                prompt_time_offset += rem_sec + 0.05
            elif key in (ord("r"), ord("R")):
                prompt_time_offset -= prompt_local_t
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        client.stop()

    if not emg_chunks or len(kin_frames) < 10:
        raise RuntimeError("Insufficient EMG or webcam frames captured.")

    emg_all = np.concatenate(emg_chunks, axis=0)  # (T_emg, 8)
    T_emg = len(emg_all)
    if T_emg < 1000:
        raise RuntimeError(
            f"Recording too short ({T_emg} EMG samples = {T_emg / 200.0:.1f}s < 5s minimum); "
            f"skipping save to avoid polluting dataset."
        )
    kin_arr = np.stack(kin_frames, axis=0)  # (T_kin, 15)

    # Interpolate kinematics onto 200 Hz EMG timeline using per-sample EMG arrival timestamps
    t_kin = np.asarray(kin_times, dtype=np.float64)
    t_emg = np.concatenate(emg_sample_times, axis=0)
    if len(t_emg) != T_emg:
        t_emg = np.linspace(t_kin[0], t_kin[-1], T_emg, dtype=np.float64)
    targets_200 = np.stack([np.interp(t_emg, t_kin, kin_arr[:, j]) for j in range(15)], axis=-1).astype(np.float32)

    hand_detect_pct = 100.0 * float(detected_frames) / float(max(1, total_frames))

    if interactive_review:
        return prompt_post_session_review(
            emg_all=emg_all,
            targets_200=targets_200,
            out_path=out_path,
            default_append=append,
            hand_detect_pct=hand_detect_pct,
            fs=200,
            parent=parent,
        )

    res = save_or_append_dataset(
        out_path,
        emg_all,
        targets_200,
        fs=200,
        append=append,
        save_session_backup=True,
    )
    return {"action": "saved", "out_path": out_path, "append": append, "result": res}


class RecordingManagerApp:
    """Graphical Studio for Opening, Inspecting, Recording, Appending/Overwriting, and Training Datasets."""

    def __init__(
        self,
        initial_path: Path,
        default_duration: float = 600.0,
        default_camera: int = 0,
        default_append: bool = True,
        root: tk.Tk | None = None,
    ) -> None:
        self.current_path = initial_path.resolve()
        self.owns_root = root is None
        self.root = tk.Tk() if root is None else root
        self.root.title("BRI Intra-Subject EMG + Kinematics Recording Studio")
        self.root.geometry("1040x680")
        self.root.minsize(920, 600)
        self.root.configure(bg="#141418")

        self.path_var = tk.StringVar(value=str(self.current_path))
        self.summary_var = tk.StringVar(value="Loading dataset...")
        self.status_var = tk.StringVar(value="Ready. Open an existing .npz recording or start a new recording session.")
        self.save_mode_var = tk.StringVar(value="append" if default_append else "overwrite")
        self.duration_var = tk.StringVar(value=f"{int(default_duration)}")
        self.camera_var = tk.IntVar(value=default_camera)
        self.countdown_var = tk.IntVar(value=3)

        self._build_ui()
        self.refresh_dataset_view()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Treeview",
            background="#1c1c24",
            foreground="#f0f0f8",
            fieldbackground="#1c1c24",
            rowheight=28,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background="#282836",
            foreground="#00e5ff",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Treeview", background=[("selected", "#2d4a68")])

        # 1. Top Header Bar — File Selector & Dataset Overview
        top_bar = tk.Frame(self.root, bg="#1c1c24", padx=18, pady=12)
        top_bar.pack(fill=tk.X)

        title_row = tk.Frame(top_bar, bg="#1c1c24")
        title_row.pack(fill=tk.X)
        tk.Label(
            title_row,
            text="⚡ BRI INTRA-SUBJECT RECORDING & DATASET STUDIO",
            bg="#1c1c24",
            fg="#f0f0f8",
            font=("Segoe UI", 13, "bold"),
        ).pack(side=tk.LEFT)

        tk.Label(
            title_row,
            textvariable=self.summary_var,
            bg="#262636",
            fg="#00e676",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
        ).pack(side=tk.RIGHT)

        file_row = tk.Frame(top_bar, bg="#1c1c24")
        file_row.pack(fill=tk.X, pady=(10, 0))

        tk.Label(
            file_row,
            text="Active Recording File (.npz):",
            bg="#1c1c24",
            fg="#a0a0b4",
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8))

        entry = tk.Entry(
            file_row,
            textvariable=self.path_var,
            bg="#121216",
            fg="#00e5ff",
            insertbackground="#ffffff",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 8))
        entry.bind("<Return>", lambda _e: self._on_path_entered())

        tk.Button(
            file_row,
            text="📂 Open Existing .npz...",
            command=self.open_dataset_dialog,
            bg="#2a2a38",
            fg="#ffffff",
            activebackground="#3a3a4e",
            activeforeground="#00e5ff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            file_row,
            text="📄 New / Save As .npz...",
            command=self.new_dataset_dialog,
            bg="#2a2a38",
            fg="#ffffff",
            activebackground="#3a3a4e",
            activeforeground="#00e5ff",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=12,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            file_row,
            text="➕ Import & Append .npz...",
            command=self.import_append_dialog,
            bg="#2a2a38",
            fg="#00e676",
            activebackground="#3a3a4e",
            activeforeground="#00e676",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        # 2. Main Split Body: Left = Sessions Table & Inspector, Right = Recording & Training Controls
        body = tk.Frame(self.root, bg="#141418", padx=18, pady=14)
        body.pack(fill=tk.BOTH, expand=True)

        left_pane = tk.Frame(body, bg="#1c1c24", padx=14, pady=12, highlightbackground="#2a2a38", highlightthickness=1)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        tk.Label(
            left_pane,
            text="Recorded Sessions in Opened Dataset",
            bg="#1c1c24",
            fg="#00e5ff",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))

        cols = ("session", "duration", "samples", "rms", "motion", "rom", "status")
        self.tree = ttk.Treeview(left_pane, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("session", text="Session")
        self.tree.heading("duration", text="Duration")
        self.tree.heading("samples", text="Samples (200Hz)")
        self.tree.heading("rms", text="EMG RMS (Active %)")
        self.tree.heading("motion", text="Hand Motion")
        self.tree.heading("rom", text="Joint ROM")
        self.tree.heading("status", text="Health")

        self.tree.column("session", width=75, anchor=tk.CENTER)
        self.tree.column("duration", width=95, anchor=tk.CENTER)
        self.tree.column("samples", width=105, anchor=tk.CENTER)
        self.tree.column("rms", width=130, anchor=tk.CENTER)
        self.tree.column("motion", width=95, anchor=tk.CENTER)
        self.tree.column("rom", width=85, anchor=tk.CENTER)
        self.tree.column("status", width=105, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        table_btns = tk.Frame(left_pane, bg="#1c1c24")
        table_btns.pack(fill=tk.X, pady=(10, 0))

        tk.Button(
            table_btns,
            text="📈 Inspect / Plot Selected Session",
            command=self.plot_selected_session,
            bg="#28344a",
            fg="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            table_btns,
            text="🗑 Remove Selected Session",
            command=self.remove_selected_session,
            bg="#4a2028",
            fg="#ff8a80",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=12,
            pady=6,
        ).pack(side=tk.LEFT)

        tk.Button(
            table_btns,
            text="🔄 Refresh",
            command=self.refresh_dataset_view,
            bg="#2a2a38",
            fg="#d0d0dc",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=10,
            pady=6,
        ).pack(side=tk.RIGHT)

        # Right Pane — Recording Settings & Actions
        right_pane = tk.Frame(
            body, bg="#1c1c24", width=330, padx=16, pady=12, highlightbackground="#2a2a38", highlightthickness=1
        )
        right_pane.pack(side=tk.RIGHT, fill=tk.Y)
        right_pane.pack_propagate(False)

        tk.Label(
            right_pane,
            text="New Recording Session",
            bg="#1c1c24",
            fg="#00e676",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))

        tk.Label(
            right_pane,
            text="Recording Mode:",
            bg="#1c1c24",
            fg="#a0a0b4",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor=tk.W, pady=(4, 2))

        tk.Radiobutton(
            right_pane,
            text="Append to opened dataset",
            variable=self.save_mode_var,
            value="append",
            bg="#1c1c24",
            fg="#f0f0f8",
            selectcolor="#2a2a38",
            activebackground="#1c1c24",
            activeforeground="#00e676",
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W)

        tk.Radiobutton(
            right_pane,
            text="Overwrite opened dataset",
            variable=self.save_mode_var,
            value="overwrite",
            bg="#1c1c24",
            fg="#f0f0f8",
            selectcolor="#2a2a38",
            activebackground="#1c1c24",
            activeforeground="#ffab00",
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(0, 8))

        # Duration preset / custom input
        dur_frame = tk.Frame(right_pane, bg="#1c1c24")
        dur_frame.pack(fill=tk.X, pady=4)
        tk.Label(
            dur_frame,
            text="Target Duration (sec):",
            bg="#1c1c24",
            fg="#a0a0b4",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor=tk.W)

        dur_sub = tk.Frame(dur_frame, bg="#1c1c24")
        dur_sub.pack(fill=tk.X, pady=(4, 0))
        self.dur_combo = ttk.Combobox(
            dur_sub,
            textvariable=self.duration_var,
            values=["60", "120", "180", "300", "600"],
            width=10,
            font=("Segoe UI", 9),
        )
        self.dur_combo.pack(side=tk.LEFT)
        tk.Label(
            dur_sub,
            text="(600s = 10 min batch)",
            bg="#1c1c24",
            fg="#8a8a9a",
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Camera & Countdown settings
        cam_frame = tk.Frame(right_pane, bg="#1c1c24")
        cam_frame.pack(fill=tk.X, pady=8)
        tk.Label(cam_frame, text="Webcam ID:", bg="#1c1c24", fg="#a0a0b4", font=("Segoe UI", 9)).grid(
            row=0, column=0, sticky=tk.W, pady=3
        )
        tk.Spinbox(
            cam_frame,
            from_=0,
            to=5,
            textvariable=self.camera_var,
            width=6,
            bg="#121216",
            fg="#f0f0f8",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
        ).grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=3)

        tk.Label(cam_frame, text="Countdown (s):", bg="#1c1c24", fg="#a0a0b4", font=("Segoe UI", 9)).grid(
            row=1, column=0, sticky=tk.W, pady=3
        )
        tk.Spinbox(
            cam_frame,
            from_=0,
            to=15,
            textvariable=self.countdown_var,
            width=6,
            bg="#121216",
            fg="#f0f0f8",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
        ).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=3)

        tk.Button(
            right_pane,
            text="⏺ Start Live Recording",
            command=self.start_recording_clicked,
            bg="#00a859",
            fg="#ffffff",
            activebackground="#00c86b",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 11, "bold"),
            pady=10,
        ).pack(fill=tk.X, pady=(10, 6))

        tk.Label(
            right_pane,
            text=(
                "After recording finishes (or when you press 'Q'),\n"
                "you will review the session and choose to\n"
                "Save or Discard (with confirmation)."
            ),
            bg="#1c1c24",
            fg="#8a8a9e",
            justify=tk.LEFT,
            font=("Segoe UI", 8),
        ).pack(anchor=tk.W, pady=(0, 14))

        ttk.Separator(right_pane, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        tk.Label(
            right_pane,
            text="Model Training",
            bg="#1c1c24",
            fg="#00e5ff",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W, pady=(4, 6))

        self.train_btn = tk.Button(
            right_pane,
            text="🧠 Train Model on Opened Dataset",
            command=self.train_model_clicked,
            bg="#3b3888",
            fg="#ffffff",
            activebackground="#4e4ab0",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            pady=8,
        )
        self.train_btn.pack(fill=tk.X, pady=(2, 6))

        # Status Footer
        footer = tk.Frame(self.root, bg="#101014", padx=18, pady=8)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(
            footer,
            textvariable=self.status_var,
            bg="#101014",
            fg="#d0d0dc",
            font=("Segoe UI", 9),
            anchor=tk.W,
        ).pack(fill=tk.X)

    def _on_path_entered(self) -> None:
        self.current_path = Path(self.path_var.get().strip()).resolve()
        self.refresh_dataset_view()

    def open_dataset_dialog(self) -> None:
        initial_dir = self.current_path.parent if self.current_path.parent.is_dir() else REPO_ROOT
        chosen = filedialog.askopenfilename(
            parent=self.root,
            title="Open Recording Dataset (.npz)",
            initialdir=str(initial_dir),
            filetypes=[("NumPy Compressed Dataset", "*.npz"), ("All Files", "*.*")],
        )
        if chosen:
            self.current_path = Path(chosen).resolve()
            self.path_var.set(str(self.current_path))
            self.refresh_dataset_view()
            self.status_var.set(f"Opened recording dataset: {self.current_path}")

    def new_dataset_dialog(self) -> None:
        initial_dir = self.current_path.parent if self.current_path.parent.is_dir() else REPO_ROOT
        chosen = filedialog.asksaveasfilename(
            parent=self.root,
            title="Select or Create Target Recording File (.npz)",
            initialdir=str(initial_dir),
            initialfile="myo_intrasubject.npz",
            defaultextension=".npz",
            filetypes=[("NumPy Compressed Dataset", "*.npz")],
        )
        if chosen:
            self.current_path = Path(chosen).resolve()
            self.path_var.set(str(self.current_path))
            self.refresh_dataset_view()
            self.status_var.set(f"Target recording path set to: {self.current_path}")

    def import_append_dialog(self) -> None:
        initial_dir = self.current_path.parent / "sessions"
        if not initial_dir.is_dir():
            initial_dir = self.current_path.parent if self.current_path.parent.is_dir() else REPO_ROOT
        chosen = filedialog.askopenfilename(
            parent=self.root,
            title="Select Session .npz File to Import & Append",
            initialdir=str(initial_dir),
            filetypes=[("NumPy Compressed Dataset", "*.npz")],
        )
        if not chosen:
            return
        src_path = Path(chosen).resolve()
        if src_path == self.current_path:
            messagebox.showwarning(
                "Same File",
                "Cannot import a file onto itself. Select a different session .npz file.",
                parent=self.root,
            )
            return
        try:
            with np.load(src_path) as data:
                imp_emg = data["emg"].astype(np.float32)
                imp_tgt = data["targets"].astype(np.float32)
                imp_fs = int(data["fs"]) if "fs" in data else 200
            res = save_or_append_dataset(
                self.current_path,
                imp_emg,
                imp_tgt,
                fs=imp_fs,
                append=True,
                save_session_backup=False,
            )
            self.refresh_dataset_view()
            self.status_var.set(
                f"Appended '{src_path.name}' (+{res['added_samples']:,} samples). "
                f"Total: {res['total_minutes']:.1f} min across {res['n_sessions']} session(s)."
            )
        except Exception as exc:
            messagebox.showerror("Import Failed", f"Could not import {src_path}:\n{exc}", parent=self.root)

    def refresh_dataset_view(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.current_path = Path(self.path_var.get().strip()).resolve()
        info = inspect_dataset_sessions(self.current_path)
        if not info["exists"] or info["total_samples"] == 0:
            self.summary_var.set("No existing data (0.0 min | 0 sessions)")
            return

        n_sess = len(info["sessions"])
        self.summary_var.set(
            f"{info['total_minutes']:.1f} min Total  |  {info['total_samples']:,} samples  |  {n_sess} Session(s)"
        )
        for s in info["sessions"]:
            dur_s = s["duration_sec"]
            mm, ss = int(dur_s // 60), int(dur_s % 60)
            self.tree.insert(
                "",
                tk.END,
                iid=str(s["index"] - 1),
                values=(
                    f"Session #{s['index']}",
                    f"{mm:02d}:{ss:02d} ({s['duration_min']:.1f}m)",
                    f"{s['n_samples']:,}",
                    f"{s['emg_rms']:.1f} ({s['active_pct']:.0f}%)",
                    f"{s['moving_pct']:.0f}%",
                    f"{s['rom_deg']:.1f}°",
                    s["status"],
                ),
            )

    def remove_selected_session(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select Session", "Please select a session in the table first.", parent=self.root)
            return
        idx_0 = int(sel[0])
        vals = self.tree.item(sel[0], "values")
        sess_label = vals[0] if vals else f"Session #{idx_0 + 1}"
        dur_label = vals[1] if len(vals) > 1 else ""
        if not messagebox.askyesno(
            "Remove Session?",
            f"Are you sure you want to remove {sess_label} ({dur_label}) from '{self.current_path.name}'?\n\n"
            f"(A backup of the removed session will be saved in sessions/.)",
            icon=messagebox.WARNING,
            parent=self.root,
        ):
            return
        try:
            res = remove_session_from_dataset(self.current_path, idx_0)
            self.refresh_dataset_view()
            self.status_var.set(
                f"Removed {sess_label}. Backup saved to {Path(str(res['backup'])).name}. "
                f"Remaining: {res['total_minutes']:.1f} min ({res['n_sessions']} session(s))."
            )
        except Exception as exc:
            messagebox.showerror("Remove Failed", str(exc), parent=self.root)

    def plot_selected_session(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select Session", "Please select a session in the table to inspect.", parent=self.root)
            return
        idx_0 = int(sel[0])
        info = inspect_dataset_sessions(self.current_path)
        if not info["exists"] or idx_0 >= len(info["sessions"]):
            return
        s_meta = info["sessions"][idx_0]
        s0, s1 = int(s_meta["start_sample"]), int(s_meta["end_sample"])
        fs = int(info["fs"])

        with np.load(self.current_path) as data:
            e_seg = data["emg"][s0:s1].astype(np.float32)
            t_seg = data["targets"][s0:s1].astype(np.float32)

        import matplotlib.pyplot as plt

        t_axis = np.arange(len(e_seg), dtype=np.float32) / float(max(fs, 1))
        step = max(1, len(e_seg) // 2500)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
        fig.canvas.manager.set_window_title(f"Session #{idx_0 + 1} Inspector — {self.current_path.name}")

        for ch in range(min(8, e_seg.shape[1])):
            ax1.plot(t_axis[::step], e_seg[::step, ch] + ch * 80.0, linewidth=0.7, label=f"Pod {ch + 1}")
        ax1.set_title(
            f"Session #{idx_0 + 1}: 8-Channel Myo sEMG ({s_meta['duration_min']:.1f} min | RMS={s_meta['emg_rms']:.1f})"
        )
        ax1.set_ylabel("sEMG + Offset")
        ax1.legend(loc="upper right", ncol=4, fontsize=7)

        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for f_i, f_name in enumerate(finger_names):
            mean_f = np.mean(t_seg[::step, f_i * 3 : f_i * 3 + 3], axis=-1)
            ax2.plot(t_axis[::step], mean_f, linewidth=1.1, label=f_name)
        ax2.set_title(
            f"Synchronized 15-DOF MediaPipe Finger Angles "
            f"(ROM={s_meta['rom_deg']:.1f}°, Motion={s_meta['moving_pct']:.0f}%)"
        )
        ax2.set_ylabel("Mean Joint Angle (°)")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylim(30.0, 185.0)
        ax2.legend(loc="lower right", ncol=5, fontsize=8)
        fig.tight_layout()
        plt.show()

    def start_recording_clicked(self) -> None:
        try:
            dur = float(self.duration_var.get().strip())
            if dur < 10.0:
                messagebox.showwarning("Duration Too Short", "Please specify at least 10 seconds.", parent=self.root)
                return
        except ValueError:
            messagebox.showwarning("Invalid Duration", "Please enter a valid number of seconds.", parent=self.root)
            return

        self.current_path = Path(self.path_var.get().strip()).resolve()
        do_append = self.save_mode_var.get() == "append"
        cam_id = int(self.camera_var.get())
        cd_sec = float(self.countdown_var.get())

        self.status_var.set("Connecting to Myo armband & Webcam... Follow prompts in the OpenCV window.")
        self.root.update_idletasks()

        try:
            decision = record_live_session(
                out_path=self.current_path,
                duration_sec=dur,
                camera_id=cam_id,
                append=do_append,
                countdown_sec=cd_sec,
                interactive_review=True,
                parent=self.root,
            )
            if decision.get("action") == "saved":
                saved_path = Path(decision["out_path"]).resolve()
                self.current_path = saved_path
                self.path_var.set(str(saved_path))
                res = decision.get("result") or {}
                self.status_var.set(
                    f"Saved recording to {saved_path.name}! "
                    f"Dataset now has {res.get('total_minutes', 0.0):.1f} min "
                    f"across {res.get('n_sessions', 1)} session(s)."
                )
            else:
                self.status_var.set("Recording session was discarded. Dataset unchanged.")
            self.refresh_dataset_view()
        except Exception as exc:
            self.status_var.set(f"Recording aborted: {exc}")
            messagebox.showerror("Recording Error", str(exc), parent=self.root)

    def train_model_clicked(self) -> None:
        self.current_path = Path(self.path_var.get().strip()).resolve()
        if not self.current_path.is_file():
            messagebox.showwarning(
                "No Dataset",
                f"Dataset file not found at:\n{self.current_path}\n\nPlease open or record a dataset first.",
                parent=self.root,
            )
            return

        self.train_btn.configure(state=tk.DISABLED, text="⏳ Training Model...")
        self.status_var.set(f"Training intra-subject TRR model on {self.current_path.name}... Please wait.")

        def _worker() -> None:
            try:
                from scripts.train_intrasubject_myo import train_intrasubject

                out_ckpt = REPO_ROOT / "models" / "best_model.pt"
                metrics = train_intrasubject(self.current_path, out_ckpt)
                mae = metrics.get("val_mae", 0.0)
                nmse = metrics.get("val_nmse", 0.0)
                self.root.after(
                    0,
                    lambda: (
                        self.train_btn.configure(state=tk.NORMAL, text="🧠 Train Model on Opened Dataset"),
                        self.status_var.set(
                            f"Training complete! Held-out MAE = {mae:.2f}°, NMSE = {nmse:.3f} "
                            f"-> Saved to models/best_model.pt"
                        ),
                        messagebox.showinfo(
                            "Training Complete",
                            f"Intra-Subject TRR model trained and saved to:\n{out_ckpt}\n\n"
                            f"Held-out Validation MAE: {mae:.2f}°\n"
                            f"Held-out Validation NMSE: {nmse:.3f}",
                            parent=self.root,
                        ),
                    ),
                )
            except Exception as exc:
                err_msg = str(exc)
                self.root.after(
                    0,
                    lambda: (
                        self.train_btn.configure(state=tk.NORMAL, text="🧠 Train Model on Opened Dataset"),
                        self.status_var.set(f"Training failed: {err_msg}"),
                        messagebox.showerror("Training Error", err_msg, parent=self.root),
                    ),
                )

        threading.Thread(target=_worker, daemon=True).start()

    def run(self) -> None:
        if self.owns_root:
            self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Intra-Subject Myo + MediaPipe Recording Studio & Dataset Manager."
    )
    parser.add_argument(
        "--out",
        "--open",
        dest="out",
        type=Path,
        default=REPO_ROOT / "data" / "recordings" / "myo_intrasubject.npz",
        help="Target or existing .npz recording path to open (default: data/recordings/myo_intrasubject.npz)",
    )
    parser.add_argument("--duration", type=float, default=600.0, help="Recording duration in seconds")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument(
        "--countdown",
        type=float,
        default=3.0,
        help="Get-ready countdown in seconds before recording starts (default: 3.0)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Append new recording to existing .npz dataset if present (default: True)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz dataset instead of appending",
    )
    parser.add_argument(
        "--direct",
        "--no-gui",
        dest="direct",
        action="store_true",
        help="Start live recording directly without launching the Dataset Manager studio window first",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Save immediately after recording without showing the interactive Save/Discard review dialog",
    )
    parser.add_argument(
        "--synthetic-test",
        action="store_true",
        help="Generate synthetic 200 Hz multi-finger dataset for automated verification",
    )
    args = parser.parse_args()
    should_append = bool(args.append and not args.overwrite)

    if args.synthetic_test:
        generate_synthetic_recording(args.out, duration_sec=28.0, append=should_append)
    elif args.direct:
        decision = record_live_session(
            args.out,
            duration_sec=args.duration,
            camera_id=args.camera,
            append=should_append,
            countdown_sec=args.countdown,
            interactive_review=not args.no_review,
        )
        if decision.get("action") == "discarded":
            sys.exit(1)
    else:
        app = RecordingManagerApp(
            initial_path=args.out,
            default_duration=args.duration,
            default_camera=args.camera,
            default_append=should_append,
        )
        app.run()


if __name__ == "__main__":
    main()
