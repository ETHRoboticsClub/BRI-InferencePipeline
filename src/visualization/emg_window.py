"""Real-time 8-channel sEMG signal monitor and inferred finger joint angles visualizer.

Provides a desktop visualization window showing:
1. Live 8-channel sEMG waveforms from the Thalmic Labs Myo armband (or mock fallback).
2. Live inferred 15-DOF continuous finger joint kinematics decoded by the TRR neural network.
3. 5-digit flexion trajectories over time (Thumb, Index, Middle, Ring, Pinky).
4. Real-time per-channel RMS muscle activation meters and angle degree gauges.
5. Live MuJoCo Shadow Hand normalized [0, 1] actuator control output streaming over UDP.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

# Ensure Windows TCL/TK libraries can always be located (especially under WindowsApps Python)
if "TCL_LIBRARY" not in os.environ or "TK_LIBRARY" not in os.environ:
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    tcl_cand = Path(base_prefix) / "tcl"
    if (tcl_cand / "tcl8.6").exists():
        os.environ.setdefault("TCL_LIBRARY", str(tcl_cand / "tcl8.6"))
    if (tcl_cand / "tk8.6").exists():
        os.environ.setdefault("TK_LIBRARY", str(tcl_cand / "tk8.6"))

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.kinematics.udp_sender import UDPSender
from src.model.model import Model
from src.preprocess import Preprocess
from src.stream.ble_myo import BleMyoClient
from src.stream.worker import StreamingInferenceWorker

logger = logging.getLogger(__name__)

# Color palettes
EMG_COLORS = (
    "#00e5ff",  # CH 1 - Cyan (Pod 1)
    "#76ff03",  # CH 2 - Bright Lime (Pod 2)
    "#ffd600",  # CH 3 - Electric Yellow (Pod 3)
    "#ff9100",  # CH 4 - Warm Amber (Pod 4 / USB pod)
    "#ff1744",  # CH 5 - Coral Red (Pod 5)
    "#e040fb",  # CH 6 - Vivid Purple (Pod 6)
    "#2979ff",  # CH 7 - Azure Blue (Pod 7)
    "#1de9b6",  # CH 8 - Spring Teal (Pod 8)
)

FINGER_COLORS = {
    "Thumb": "#ff9100",  # Amber / Orange
    "Index": "#00e5ff",  # Cyan
    "Middle": "#76ff03",  # Bright Lime
    "Ring": "#e040fb",  # Magenta / Purple
    "Pinky": "#ff1744",  # Coral / Red
}

FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")


class EMGVisualizerApp:
    """Tkinter-based live oscilloscope, TRR inferred kinematics, and MuJoCo bridge monitor.

    Parameters
    ----------
    client : BleMyoClient
        Active acquisition client providing streaming samples through its ring buffer.
    model_path : str or Path, optional
        Path to pretrained TRR model checkpoint. Defaults to 'models/best_model.pt'.
    enable_udp : bool, default=True
        Whether to stream decoded actuator controls to MuJoCo via UDP (:5005).
    fs : int, default=200
        Sampling frequency in Hertz.
    default_time_window : float, default=0.1
        Initial duration of the visible rolling oscilloscope window in seconds.
    """

    def __init__(
        self,
        client: BleMyoClient,
        model_path: str | Path | None = None,
        enable_udp: bool = True,
        fs: int = 200,
        default_time_window: float = 0.1,
        root: tk.Tk | None = None,
    ) -> None:
        self.client = client
        self.fs = fs
        self.time_window = default_time_window
        self.channels = 8
        self.enable_udp = enable_udp

        # 1. Setup Signal Preprocessor
        self.filter_engine = Preprocess(
            fs=self.fs,
            bandpass=(15.0, 95.0),
            notch=50.0,
            notch_q=30.0,
            channels=self.channels,
        )

        # 2. Setup TRR Neural Network Model & Worker
        self._init_inference_engine(model_path)

        # State flags
        self.is_paused = False
        self.filter_active = True
        self.view_mode = "split"  # 'split', 'emg', 'angles'
        self.scale_mode = "±50"
        self.gain_spacing = 75.0

        # Calibration state
        self.is_calibrating = False
        self.calib_phase = 0
        self.calib_start_time = 0.0
        self.calib_open_angles: list[np.ndarray] = []
        self.calib_fist_angles: list[np.ndarray] = []
        self.calib_rms_samples: list[np.ndarray] = []
        self.calib_open_raw: list[np.ndarray] = []
        self.calib_fist_raw: list[np.ndarray] = []

        # Rolling history buffers for plotting
        n_samples = int(self.time_window * self.fs)
        self.time_axis = np.linspace(-self.time_window, 0.0, n_samples)
        # 15 continuous angles history (initialized to 180° = fully extended)
        self.angles_history = np.full((15, n_samples), 180.0, dtype=np.float32)

        # Diagnostics & FPS counters
        self.last_update_time = time.perf_counter()
        self.frame_count = 0
        self.fps = 0.0
        self.last_sample_count = 0
        self.sample_rate_est = 0.0
        self.last_rate_time = time.perf_counter()

        # GUI Root
        self.owns_root = root is None
        self.root = tk.Tk() if root is None else root
        self._configure_root_window()
        self._build_ui()

        # Bind keyboard shortcuts
        self.root.bind("<space>", lambda e: self.toggle_pause())
        self.root.bind("<f>", lambda e: self.toggle_filter())
        self.root.bind("<F>", lambda e: self.toggle_filter())
        self.root.bind("<c>", lambda e: self.start_calibration())
        self.root.bind("<C>", lambda e: self.start_calibration())
        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start inference worker thread
        if self.worker is not None:
            self.worker.start()

        # Schedule update loop (~40 Hz = 25 ms)
        self.is_running = True
        self.root.after(50, self._update_loop)

    def _init_inference_engine(self, model_path: str | Path | None) -> None:
        """Load pretrained TRR model and initialize streaming worker."""
        self.model = Model(channels=8, targets=15)
        if model_path:
            path = Path(model_path)
        else:
            cand1 = Path("models/best_model.pt")
            cand2 = Path(__file__).resolve().parent.parent.parent / "models" / "best_model.pt"
            path = cand1 if cand1.is_file() else cand2

        if path.is_file():
            try:
                self.model.load_pretrained_weights(path, map_location="cpu")
                logger.info("Loaded TRR weights from %s", path)
            except Exception as exc:
                logger.warning("Failed to load weights from %s: %s", path, exc)
        else:
            logger.warning("No checkpoint found at %s. Using initialized weights.", path)

        self.model.eval()

        # Setup UDP transmitter for MuJoCo
        self.udp_sender: UDPSender | None = None
        if self.enable_udp:
            try:
                self.udp_sender = UDPSender(host="127.0.0.1", port=5005)
            except Exception as exc:
                logger.warning("UDPSender initialization skipped: %s", exc)

        # Initialize StreamingInferenceWorker
        try:
            self.worker: StreamingInferenceWorker | None = StreamingInferenceWorker(
                ring_buffer=self.client.ring_buffer,
                model=self.model,
                udp_sender=self.udp_sender,
                pace_loop=True,
            )
        except Exception as exc:
            logger.warning("StreamingInferenceWorker creation failed: %s", exc)
            self.worker = None

    def _configure_root_window(self) -> None:
        """Set window title, geometry, and dark styling."""
        self.root.title("Myo sEMG Monitor & Inferred Finger Kinematics")
        self.root.geometry("1380x880")
        self.root.minsize(1080, 700)
        self.root.configure(bg="#121215")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#121215")
        style.configure("Header.TFrame", background="#1a1a20")
        style.configure(
            "Header.TLabel",
            background="#1a1a20",
            foreground="#f0f0f5",
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "SubHeader.TLabel",
            background="#1a1a20",
            foreground="#9a9aa8",
            font=("Segoe UI", 9),
        )

    def _build_ui(self) -> None:
        """Construct the top control toolbar, split oscilloscope plot, and activity sidebar."""
        # 1. Top Header Bar
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(16, 10))
        header.pack(side=tk.TOP, fill=tk.X)

        title_box = ttk.Frame(header, style="Header.TFrame")
        title_box.pack(side=tk.LEFT)

        title_lbl = ttk.Label(
            title_box,
            text="⚡ MYO sEMG MONITOR & INFERRED KINEMATICS",
            style="Header.TLabel",
        )
        title_lbl.pack(anchor=tk.W)

        sub_lbl = ttk.Label(
            title_box,
            text="8-Channel sEMG • TRR 15-DOF Kinematics • MuJoCo UDP Bridge",
            style="SubHeader.TLabel",
        )
        sub_lbl.pack(anchor=tk.W)

        # Status Pill Badge
        self.status_var = tk.StringVar(value="CONNECTING...")
        self.status_lbl = tk.Label(
            header,
            textvariable=self.status_var,
            bg="#252535",
            fg="#ffd600",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
            relief=tk.FLAT,
        )
        self.status_lbl.pack(side=tk.LEFT, padx=(20, 0))

        # Right Controls: Filter Toggle, View Selector, Time Span, Gain Scale, Pause
        controls_box = ttk.Frame(header, style="Header.TFrame")
        controls_box.pack(side=tk.RIGHT)

        # View mode selector
        ttk.Label(
            controls_box,
            text="View:",
            style="SubHeader.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.view_combo = ttk.Combobox(
            controls_box,
            values=["Split (EMG + Angles)", "EMG Only", "Angles Only"],
            state="readonly",
            width=18,
            font=("Segoe UI", 9),
        )
        self.view_combo.set("Split (EMG + Angles)")
        self.view_combo.bind("<<ComboboxSelected>>", self._on_view_changed)
        self.view_combo.pack(side=tk.LEFT, padx=(0, 12))

        # Filter mode toggle
        self.filter_var = tk.BooleanVar(value=True)
        filter_btn = tk.Checkbutton(
            controls_box,
            text="Filter EMG",
            variable=self.filter_var,
            command=self._on_filter_changed,
            bg="#1a1a20",
            fg="#f0f0f5",
            selectcolor="#2a2a38",
            activebackground="#1a1a20",
            activeforeground="#00e5ff",
            font=("Segoe UI", 9),
        )
        filter_btn.pack(side=tk.LEFT, padx=(0, 10))

        # UDP Stream Toggle
        self.udp_var = tk.BooleanVar(value=self.enable_udp)
        udp_btn = tk.Checkbutton(
            controls_box,
            text="Send UDP (:5005)",
            variable=self.udp_var,
            command=self._on_udp_toggled,
            bg="#1a1a20",
            fg="#00e676",
            selectcolor="#2a2a38",
            activebackground="#1a1a20",
            activeforeground="#00e676",
            font=("Segoe UI", 9, "bold"),
        )
        udp_btn.pack(side=tk.LEFT, padx=(0, 12))

        # Time Window Selector
        ttk.Label(
            controls_box,
            text="Window:",
            style="SubHeader.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.window_combo = ttk.Combobox(
            controls_box,
            values=["0.1s", "0.2s", "0.5s", "1.0s", "2.0s"],
            state="readonly",
            width=6,
            font=("Segoe UI", 9),
        )
        self.window_combo.set(f"{self.time_window:.1f}s")
        self.window_combo.bind("<<ComboboxSelected>>", self._on_window_changed)
        self.window_combo.pack(side=tk.LEFT, padx=(0, 12))

        # Pause / Resume Button
        self.pause_btn = tk.Button(
            controls_box,
            text="⏸ Pause",
            command=self.toggle_pause,
            bg="#2a2a38",
            fg="#ffffff",
            activebackground="#3a3a4e",
            activeforeground="#00e5ff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=3,
        )
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 6))

        # Calibration Button
        self.calib_btn = tk.Button(
            controls_box,
            text="🎯 Calibrate (C)",
            command=self.start_calibration,
            bg="#2a2a38",
            fg="#ffd600",
            activebackground="#3a3a4e",
            activeforeground="#ffd600",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=3,
        )
        self.calib_btn.pack(side=tk.LEFT, padx=(0, 6))

        # 2. Main Center Body (Plots + Sidebar)
        body = ttk.Frame(self.root, padding=(12, 10))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Matplotlib Embedded Plot Canvas (Left)
        plot_frame = ttk.Frame(body)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._init_matplotlib_figures(plot_frame)

        # Right-hand Inferred Kinematics & Muscle Activation Sidebar
        sidebar = tk.Frame(body, bg="#1a1a22", width=340, padx=12, pady=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        sidebar.pack_propagate(False)

        self._build_sidebar(sidebar)

        # 3. Bottom Status Bar
        bottom_bar = tk.Frame(self.root, bg="#15151b", height=28)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.footer_lbl = tk.Label(
            bottom_bar,
            text="Shortcuts: [Space] Pause/Resume  •  [F] Toggle Filter  •  [Esc] Exit",
            bg="#15151b",
            fg="#707080",
            font=("Segoe UI", 8),
            anchor=tk.W,
            padx=14,
            pady=4,
        )
        self.footer_lbl.pack(side=tk.LEFT)

        self.perf_lbl = tk.Label(
            bottom_bar,
            text="Rate: -- Hz  •  Inference: -- ms  •  FPS: --",
            bg="#15151b",
            fg="#00e5ff",
            font=("Segoe UI", 8, "bold"),
            anchor=tk.E,
            padx=14,
            pady=4,
        )
        self.perf_lbl.pack(side=tk.RIGHT)

    def _init_matplotlib_figures(self, parent: ttk.Frame) -> None:
        """Create split Matplotlib subplots for 8-channel sEMG and 5-finger inferred kinematics."""
        n_samples = int(self.time_window * self.fs)
        self.time_axis = np.linspace(-self.time_window, 0.0, n_samples)

        self.fig = Figure(figsize=(8, 7), dpi=100, facecolor="#141418")
        self.gs = self.fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.28)

        # Subplot 1: 8-Channel sEMG Oscilloscope
        self.ax_emg = self.fig.add_subplot(self.gs[0, 0])
        self.ax_emg.set_facecolor("#101014")
        self.ax_emg.set_title(
            "8 Forearm sEMG Pods (Raw / SOS Filtered)",
            color="#00e5ff",
            fontsize=9,
            fontweight="bold",
            loc="left",
        )

        self.baselines = np.array([(7 - i) * self.gain_spacing for i in range(8)], dtype=np.float32)
        self.emg_lines = []
        for i in range(8):
            base_y = self.baselines[i]
            self.ax_emg.axhline(y=base_y, color="#252532", linestyle="--", linewidth=0.6, zorder=1)
            zero_data = np.full(n_samples, base_y, dtype=np.float32)
            (line,) = self.ax_emg.plot(self.time_axis, zero_data, color=EMG_COLORS[i], linewidth=1.1, zorder=2)
            self.emg_lines.append(line)

        self._update_emg_axis_limits()
        self.ax_emg.set_xlim(-self.time_window, 0.0)
        self.ax_emg.set_yticks(self.baselines)
        self.ax_emg.set_yticklabels([f"P{i + 1}" for i in range(8)], color="#d0d0dc", fontsize=8, fontweight="bold")
        self.ax_emg.tick_params(axis="x", colors="#8a8a9a", labelsize=8)
        self.ax_emg.tick_params(axis="y", colors="#8a8a9a", length=0)
        for spine in self.ax_emg.spines.values():
            spine.set_color("#2a2a36")

        # Subplot 2: Inferred 5-Finger Joint Kinematics (0° = flexed, 180° = extended)
        self.ax_angles = self.fig.add_subplot(self.gs[1, 0], sharex=self.ax_emg)
        self.ax_angles.set_facecolor("#101014")
        self.ax_angles.set_title(
            "TRR Inferred Finger Kinematics (Mean Angle: 180°=Open, <90°=Flexed)",
            color="#ffd600",
            fontsize=9,
            fontweight="bold",
            loc="left",
        )

        # Reference lines for extension and 90° flexion
        self.ax_angles.axhline(y=180.0, color="#3a3a4e", linestyle=":", linewidth=0.8, label="180° (Open)")
        self.ax_angles.axhline(y=90.0, color="#4a2525", linestyle=":", linewidth=0.8, label="90° (Flexed)")

        self.finger_lines = {}
        for finger, color in FINGER_COLORS.items():
            init_data = np.full(n_samples, 180.0, dtype=np.float32)
            (line,) = self.ax_angles.plot(self.time_axis, init_data, color=color, linewidth=1.4, label=finger)
            self.finger_lines[finger] = line

        self.ax_angles.set_ylim(20.0, 195.0)
        self.ax_angles.set_ylabel("Angle (°)", color="#8a8a9a", fontsize=8)
        self.ax_angles.set_xlabel("Time (seconds relative to now)", color="#8a8a9a", fontsize=8)
        self.ax_angles.legend(
            loc="lower left",
            ncol=5,
            fontsize=7,
            facecolor="#1a1a22",
            edgecolor="#2a2a36",
            labelcolor="#d0d0dc",
        )
        self.ax_angles.tick_params(axis="x", colors="#8a8a9a", labelsize=8)
        self.ax_angles.tick_params(axis="y", colors="#8a8a9a", labelsize=8)
        for spine in self.ax_angles.spines.values():
            spine.set_color("#2a2a36")

        self.fig.tight_layout(pad=1.4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        """Construct the sidebar displaying live inferred angles, controls, and RMS meters."""
        # Section A: Inferred 15-DOF Kinematics
        tk.Label(
            parent,
            text="INFERRED FINGER ANGLES (15-DOF)",
            bg="#1a1a22",
            fg="#ffd600",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 2))

        tk.Label(
            parent,
            text="Decoded from 8-ch sEMG via TRR GRU",
            bg="#1a1a22",
            fg="#7a7a8a",
            font=("Segoe UI", 8),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 6))

        # Finger angle cards
        self.finger_deg_labels = {}
        self.finger_progress_canvases = {}

        finger_box = tk.Frame(parent, bg="#1a1a22")
        finger_box.pack(fill=tk.X, pady=(0, 10))

        finger_joint_indices = {
            "Thumb": (0, 1, 2),  # CMC, MCP, IP
            "Index": (3, 4, 5),  # MCP, PIP, DIP
            "Middle": (6, 7, 8),  # MCP, PIP, DIP
            "Ring": (9, 10, 11),  # MCP, PIP, DIP
            "Pinky": (12, 13, 14),  # MCP, PIP, DIP
        }
        self.finger_joint_indices = finger_joint_indices

        for finger, color in FINGER_COLORS.items():
            card = tk.Frame(
                finger_box,
                bg="#131318",
                padx=6,
                pady=4,
                highlightthickness=1,
                highlightbackground="#252530",
            )
            card.pack(fill=tk.X, pady=2)

            top_row = tk.Frame(card, bg="#131318")
            top_row.pack(fill=tk.X)

            tk.Label(
                top_row,
                text=finger.upper(),
                bg="#131318",
                fg=color,
                font=("Segoe UI", 8, "bold"),
            ).pack(side=tk.LEFT)

            deg_lbl = tk.Label(
                top_row,
                text="180.0°",
                bg="#131318",
                fg="#ffffff",
                font=("Segoe UI", 8, "bold"),
            )
            deg_lbl.pack(side=tk.RIGHT)
            self.finger_deg_labels[finger] = deg_lbl

            # Sub-joint breakdown text: MCP / PIP / DIP
            sub_lbl = tk.Label(
                card,
                text="MCP: 180°  PIP: 180°  DIP: 180°",
                bg="#131318",
                fg="#8a8a9a",
                font=("Segoe UI", 7),
                anchor=tk.W,
            )
            sub_lbl.pack(fill=tk.X, pady=(1, 2))
            self.finger_deg_labels[f"{finger}_sub"] = sub_lbl

            # Flexion bar: [0, 180] -> [0, 290] width
            c = tk.Canvas(card, bg="#0d0d10", height=8, width=290, highlightthickness=0)
            c.pack(fill=tk.X)
            rect = c.create_rectangle(0, 0, 290, 8, fill=color, width=0)
            self.finger_progress_canvases[finger] = (c, rect)

        # Section B: MuJoCo Shadow Hand Actuators [0, 1]
        tk.Frame(parent, bg="#2a2a36", height=1).pack(fill=tk.X, pady=8)

        tk.Label(
            parent,
            text="SHADOW HAND UDP CONTROLS [0, 1]",
            bg="#1a1a22",
            fg="#00e676",
            font=("Segoe UI", 8, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X)

        self.udp_status_lbl = tk.Label(
            parent,
            text="UDP: 127.0.0.1:5005 (Active)",
            bg="#1a1a22",
            fg="#a0a0b0",
            font=("Segoe UI", 7),
            anchor=tk.W,
        )
        self.udp_status_lbl.pack(fill=tk.X, pady=(1, 4))

        self.ctrl_labels = {}
        ctrl_frame = tk.Frame(parent, bg="#1a1a22")
        ctrl_frame.pack(fill=tk.X)

        key_actuators = (
            ("Thumb THJ1", 6),
            ("Index FFJ0", 9),
            ("Middle MFJ0", 12),
            ("Ring RFJ0", 15),
            ("Little LFJ0", 19),
        )
        self.key_actuators = key_actuators

        for act_name, idx in key_actuators:
            row = tk.Frame(ctrl_frame, bg="#1a1a22")
            row.pack(fill=tk.X, pady=1)

            tk.Label(
                row,
                text=act_name,
                bg="#1a1a22",
                fg="#9090a0",
                font=("Segoe UI", 7),
                width=12,
                anchor=tk.W,
            ).pack(side=tk.LEFT)

            c = tk.Canvas(row, bg="#101015", height=10, width=140, highlightthickness=0)
            c.pack(side=tk.LEFT, padx=3)
            rect = c.create_rectangle(0, 0, 0, 10, fill="#00e676", width=0)

            val_lbl = tk.Label(
                row,
                text="0.00",
                bg="#1a1a22",
                fg="#00e676",
                font=("Segoe UI", 7),
                width=5,
                anchor=tk.E,
            )
            val_lbl.pack(side=tk.RIGHT)

            self.ctrl_labels[idx] = (c, rect, val_lbl)

        # Section C: 8 Pod sEMG RMS Meters
        tk.Frame(parent, bg="#2a2a36", height=1).pack(fill=tk.X, pady=8)

        tk.Label(
            parent,
            text="sEMG POD CONTRACTION (RMS)",
            bg="#1a1a22",
            fg="#00e5ff",
            font=("Segoe UI", 8, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 3))

        self.rms_canvases = []
        self.rms_value_labels = []

        meter_frame = tk.Frame(parent, bg="#1a1a22")
        meter_frame.pack(fill=tk.X)

        for i in range(8):
            row = tk.Frame(meter_frame, bg="#1a1a22")
            row.pack(fill=tk.X, pady=1)

            lbl = tk.Label(
                row,
                text=f"P{i + 1}",
                bg="#1a1a22",
                fg=EMG_COLORS[i],
                font=("Segoe UI", 7, "bold"),
                width=3,
                anchor=tk.W,
            )
            lbl.pack(side=tk.LEFT)

            c = tk.Canvas(row, bg="#101015", height=9, width=170, highlightthickness=0)
            c.pack(side=tk.LEFT, padx=3)
            rect = c.create_rectangle(0, 0, 0, 9, fill=EMG_COLORS[i], width=0)
            self.rms_canvases.append((c, rect))

            val_lbl = tk.Label(
                row,
                text="0.0",
                bg="#1a1a22",
                fg="#d0d0dc",
                font=("Segoe UI", 7),
                width=4,
                anchor=tk.E,
            )
            val_lbl.pack(side=tk.RIGHT)
            self.rms_value_labels.append(val_lbl)

    def _update_emg_axis_limits(self) -> None:
        """Recalculate EMG Y-axis bounds based on gain and baselines."""
        top = self.baselines[0] + self.gain_spacing * 0.95
        bottom = self.baselines[7] - self.gain_spacing * 0.95
        self.ax_emg.set_ylim(bottom, top)

    def _on_filter_changed(self) -> None:
        """Handle toggle of digital signal filtering."""
        self.filter_active = self.filter_var.get()

    def _on_udp_toggled(self) -> None:
        """Enable or disable UDP streaming to MuJoCo."""
        self.enable_udp = self.udp_var.get()
        if self.worker is not None:
            if self.enable_udp:
                if self.worker.udp_sender is None:
                    self.worker.udp_sender = UDPSender(host="127.0.0.1", port=5005)
                self.udp_status_lbl.configure(text="UDP: 127.0.0.1:5005 (Active)", fg="#a0a0b0")
            else:
                self.worker.udp_sender = None
                self.udp_status_lbl.configure(text="UDP: Disabled (Paused)", fg="#ff5252")

    def _on_view_changed(self, event=None) -> None:
        """Switch between Split, EMG Only, and Angles Only views."""
        choice = self.view_combo.get()
        if "Split" in choice:
            self.ax_emg.set_visible(True)
            self.ax_angles.set_visible(True)
            self.gs.set_height_ratios([1.2, 1.0])
        elif "EMG" in choice:
            self.ax_emg.set_visible(True)
            self.ax_angles.set_visible(False)
            self.gs.set_height_ratios([1.0, 0.001])
        else:
            self.ax_emg.set_visible(False)
            self.ax_angles.set_visible(True)
            self.gs.set_height_ratios([0.001, 1.0])
        self.fig.tight_layout(pad=1.4)
        self.canvas.draw_idle()

    def _on_window_changed(self, event=None) -> None:
        """Handle change in time window duration."""
        val = self.window_combo.get().replace("s", "")
        try:
            self.time_window = float(val)
            n_samples = max(int(self.time_window * self.fs), 2)
            self.time_axis = np.linspace(-self.time_window, 0.0, n_samples)
            self.ax_emg.set_xlim(-self.time_window, 0.0)
            self.angles_history = np.full((15, n_samples), 180.0, dtype=np.float32)
            for line in self.emg_lines:
                line.set_xdata(self.time_axis)
                curr_y = line.get_ydata()
                init_val = curr_y[0] if len(curr_y) > 0 else 0.0
                line.set_ydata(np.full(n_samples, init_val, dtype=np.float32))
            for line in self.finger_lines.values():
                line.set_xdata(self.time_axis)
                line.set_ydata(np.full(n_samples, 180.0, dtype=np.float32))
            self.canvas.draw_idle()
        except ValueError:
            pass

    def toggle_pause(self) -> None:
        """Toggle freeze/resume of visualizer traces."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.configure(text="▶ Resume", bg="#3a3a4e")
        else:
            self.pause_btn.configure(text="⏸ Pause", bg="#2a2a38")

    def toggle_filter(self) -> None:
        """Toggle filtering via shortcut."""
        self.filter_var.set(not self.filter_var.get())
        self._on_filter_changed()

    def start_calibration(self) -> None:
        """Initiate 4-second interactive kinematic and gain calibration."""
        if self.is_calibrating:
            return
        self.is_calibrating = True
        self.calib_phase = 1
        self.calib_start_time = time.perf_counter()
        self.calib_open_angles.clear()
        self.calib_fist_angles.clear()
        self.calib_rms_samples.clear()
        self.calib_open_raw.clear()
        self.calib_fist_raw.clear()
        self.calib_btn.configure(text="⏳ Calibrating...", bg="#4a3a00")
        if self.worker is not None and hasattr(self.worker, "_notify_mujoco_status"):
            self.worker._notify_mujoco_status(
                1, "CALIB (1/2): EXTEND HAND OPEN FLAT (2.0s)", "Extend fingers wide open"
            )
        logger.info("Interactive calibration initiated.")

    def _finish_calibration(self) -> None:
        """Compute personal dynamic range bounds, pod alignment, and 3-band channel scale."""
        self.is_calibrating = False
        self.calib_btn.configure(text="🎯 Calibrate (C)", bg="#2a2a38")

        if self.calib_open_angles and self.calib_fist_angles and self.worker is not None:
            # 1. Calibrate 3-band pod rotation (extensor vs flexor alignment) and channel_std
            info = {}
            if self.calib_open_raw and self.calib_fist_raw and hasattr(self.worker, "calibrate_guided"):
                try:
                    info = self.worker.calibrate_guided(self.calib_open_raw, self.calib_fist_raw)
                except Exception as exc:
                    logger.warning("Guided 3-band calibration skipped: %s", exc)

            open_mean = np.mean(self.calib_open_angles, axis=0)
            fist_mean = np.mean(self.calib_fist_angles, axis=0)
            ext_b = np.maximum(open_mean, fist_mean)
            flex_b = np.minimum(open_mean, fist_mean)
            # Ensure minimum dynamic span of 4.0 degrees per joint
            span = ext_b - flex_b
            too_small = span < 4.0
            if np.any(too_small):
                ext_b[too_small] += 2.0
                flex_b[too_small] -= 2.0

            kin_cfg = self.worker.params.setdefault("kinematics", {})
            kin_cfg["mode"] = "empirical"
            kin_cfg["empirical_bounds"] = {
                "extension": ext_b.tolist(),
                "flexion": flex_b.tolist(),
            }

            if self.calib_rms_samples:
                # 85th percentile of RMS represents typical active contraction
                active_rms = np.percentile(np.array(self.calib_rms_samples), 85, axis=0)
                active_scale = np.maximum(active_rms, 5.0)
                self.worker.set_channel_scale(active_scale)

            ext_p = info.get("extensor_pod", 1)
            flex_p = info.get("flexor_pod", 5)
            done_msg = f"CALIBRATION COMPLETE! (Ext: P{ext_p} | Flex: P{flex_p})"
            logger.info("Calibration complete. Custom empirical bounds and pod alignment set.")
            self.status_var.set(done_msg)
            self.status_lbl.configure(bg="#003820", fg="#00e676")
            if hasattr(self.worker, "_notify_mujoco_status"):
                self.worker._notify_mujoco_status(3, done_msg, "Pod rotation & gains aligned")
            if hasattr(self.worker, "broadcast_calibration_sync"):
                self.worker.broadcast_calibration_sync()
        else:
            self.status_var.set("CALIBRATION SKIPPED")
            self.status_lbl.configure(bg="#252535", fg="#ffd600")

    def _update_loop(self) -> None:
        """Periodic GUI refresh callback (~40 FPS)."""
        if not self.is_running:
            return

        now = time.perf_counter()

        # Update FPS & Sample rate estimates every 0.5s
        self.frame_count += 1
        elapsed_rate = now - self.last_rate_time
        if elapsed_rate >= 0.5:
            self.fps = self.frame_count / elapsed_rate
            self.frame_count = 0

            cur_samples = len(self.client.ring_buffer)
            if hasattr(self.client, "is_mock") and self.client.is_mock:
                self.sample_rate_est = float(self.fs)
            else:
                delta_s = cur_samples - self.last_sample_count
                if delta_s >= 0:
                    self.sample_rate_est = delta_s / elapsed_rate

            self.last_sample_count = cur_samples
            self.last_rate_time = now

            self._refresh_status_badges()

        # Render signals & kinematics if not paused
        if not self.is_paused:
            self._render_signals_and_kinematics()

        self.root.after(25, self._update_loop)

    def _refresh_status_badges(self) -> None:
        """Update connection badge and diagnostic values."""
        is_shared = getattr(self.client, "is_shared_follower", False)
        if self.worker is not None and self.enable_udp:
            if is_shared and self.worker.udp_sender is not None:
                self.worker.udp_sender = None
                self.udp_status_lbl.configure(text="UDP: 127.0.0.1:5005 (Shared Primary)", fg="#00e5ff")
            elif not is_shared and self.worker.udp_sender is None and self.udp_sender is not None:
                self.worker.udp_sender = self.udp_sender
                self.udp_status_lbl.configure(text="UDP: 127.0.0.1:5005 (Active)", fg="#a0a0b0")

        if self.is_calibrating:
            return

        is_mock = getattr(self.client, "is_mock", True)
        is_conn = getattr(self.client, "is_connected", False)

        if is_conn:
            if is_shared:
                self.status_var.set("CONNECTED: MYO BLE (SHARED)")
            else:
                self.status_var.set("CONNECTED: MYO BLE")
            self.status_lbl.configure(bg="#003820", fg="#00e676")
        elif is_shared:
            self.status_var.set("SHARED HUB: SCANNING MYO...")
            self.status_lbl.configure(bg="#3a2a00", fg="#ffab00")
        elif is_mock:
            self.status_var.set("SIMULATION: MOCK REPLAY")
            self.status_lbl.configure(bg="#002b3d", fg="#00d4ff")
        else:
            self.status_var.set("SCANNING FOR MYO...")
            self.status_lbl.configure(bg="#3a2a00", fg="#ffab00")

        inf_ms = 0.0
        if self.worker is not None:
            inf_ms = getattr(self.worker, "latest_compute_latency_ms", 0.0)

        self.perf_lbl.configure(text=f"{self.sample_rate_est:.1f}Hz | TRR:{inf_ms:.1f}ms | GUI:{self.fps:.0f}FPS")

    def _render_signals_and_kinematics(self) -> None:
        """Pull latest samples, update sEMG traces, TRR inferred angles, and control meters."""
        target_n = int(self.time_window * self.fs)
        buf_len = len(self.client.ring_buffer)

        if buf_len < 2:
            return

        fetch_n = min(target_n, buf_len)
        try:
            samples, _ = self.client.ring_buffer.get_latest(fetch_n)
        except Exception as e:
            logger.debug("get_latest peek skipped: %s", e)
            return

        # 1. Update sEMG waveform traces
        if self.filter_active and fetch_n >= 8:
            try:
                emg_disp = self.filter_engine(samples)
            except Exception:
                emg_disp = samples
        else:
            emg_disp = samples

        if fetch_n < target_n:
            pad_len = target_n - fetch_n
            padded_emg = np.pad(emg_disp, ((0, 0), (pad_len, 0)), mode="edge")
        else:
            padded_emg = emg_disp[:, -target_n:]

        if self.ax_emg.get_visible():
            for i in range(8):
                y_vals = padded_emg[i] + self.baselines[i]
                self.emg_lines[i].set_ydata(y_vals)

        # 2. Update RMS meters
        rms_window = min(fetch_n, 40)
        recent_chunk = emg_disp[:, -rms_window:]
        rms_vals = np.sqrt(np.mean(recent_chunk**2, axis=-1))

        max_rms_scale = 50.0
        for i in range(8):
            rms = float(rms_vals[i])
            bar_w = int(min(170.0, (rms / max_rms_scale) * 170.0))
            canvas, rect = self.rms_canvases[i]
            canvas.coords(rect, 0, 0, bar_w, 9)
            self.rms_value_labels[i].configure(text=f"{rms:4.1f}")

        # 3. Read Inferred 15-DOF Finger Angles & MuJoCo Controls
        latest_angles = None
        latest_controls = None

        if self.worker is not None:
            latest_angles = self.worker.latest_angles
            latest_controls = self.worker.latest_controls
            if getattr(self.worker, "is_calibrating", False) and not self.is_calibrating:
                self.start_calibration()

        # Handle interactive calibration phase progression
        if self.is_calibrating:
            elapsed = time.perf_counter() - self.calib_start_time
            raw_win = samples[:, -rms_window:]
            if elapsed < 2.0:
                self.calib_phase = 1
                rem = max(0.0, 2.0 - elapsed)
                msg = f"CALIB (1/2): EXTEND HAND OPEN FLAT ({rem:3.1f}s)"
                self.status_var.set(msg)
                self.status_lbl.configure(bg="#3a2a00", fg="#ffd600")
                if latest_angles is not None:
                    self.calib_open_angles.append(latest_angles.copy())
                self.calib_rms_samples.append(rms_vals.copy())
                self.calib_open_raw.append(raw_win.copy())
                if self.worker is not None and hasattr(self.worker, "_notify_mujoco_status"):
                    self.worker._notify_mujoco_status(1, msg, "Extend fingers wide open")
            elif elapsed < 4.0:
                self.calib_phase = 2
                rem = max(0.0, 4.0 - elapsed)
                msg = f"CALIB (2/2): MAKE A TIGHT FIST ({rem:3.1f}s)"
                self.status_var.set(msg)
                self.status_lbl.configure(bg="#4a1a00", fg="#ff9100")
                if latest_angles is not None:
                    self.calib_fist_angles.append(latest_angles.copy())
                self.calib_rms_samples.append(rms_vals.copy())
                self.calib_fist_raw.append(raw_win.copy())
                if self.worker is not None and hasattr(self.worker, "_notify_mujoco_status"):
                    self.worker._notify_mujoco_status(2, msg, "Close hand into a firm fist")
            else:
                self._finish_calibration()

        if latest_angles is not None and len(latest_angles) == 15:
            # Shift angle history rolling buffer
            self.angles_history = np.roll(self.angles_history, -1, axis=-1)
            self.angles_history[:, -1] = np.clip(latest_angles, 0.0, 180.0)

            # Update 5 Finger Kinematics Trajectories on ax_angles
            if self.ax_angles.get_visible():
                for finger, indices in self.finger_joint_indices.items():
                    mean_traj = np.mean(self.angles_history[list(indices), :], axis=0)
                    self.finger_lines[finger].set_ydata(mean_traj)

            # Update Sidebar Finger Cards
            for finger, indices in self.finger_joint_indices.items():
                mcp, pip, dip = (
                    float(latest_angles[indices[0]]),
                    float(latest_angles[indices[1]]),
                    float(latest_angles[indices[2]]),
                )
                mean_deg = (mcp + pip + dip) / 3.0

                self.finger_deg_labels[finger].configure(text=f"{mean_deg:5.1f}°")
                self.finger_deg_labels[f"{finger}_sub"].configure(
                    text=f"MCP:{mcp:3.0f}° PIP:{pip:3.0f}° DIP:{dip:3.0f}°"
                )

                # Extension fraction [0.0 = flexed, 1.0 = extended] -> bar width
                frac = float(np.clip(mean_deg / 180.0, 0.0, 1.0))
                bar_w = int(frac * 290.0)
                canvas, rect = self.finger_progress_canvases[finger]
                canvas.coords(rect, 0, 0, bar_w, 8)

        # 4. Update MuJoCo Controls Display
        if latest_controls is not None and len(latest_controls) == 20:
            for act_name, idx in self.key_actuators:
                val = float(latest_controls[idx])
                bar_w = int(min(140.0, val * 140.0))
                canvas, rect, lbl = self.ctrl_labels[idx]
                canvas.coords(rect, 0, 0, bar_w, 10)
                lbl.configure(text=f"{val:4.2f}")

        self.canvas.draw_idle()

    def on_close(self) -> None:
        """Cleanly terminate update loop, stop worker, stop client, and close window."""
        self.is_running = False
        if self.worker is not None:
            try:
                self.worker.stop()
            except Exception as e:
                logger.debug("Error stopping worker: %s", e)

        try:
            self.client.stop()
        except Exception as e:
            logger.debug("Error stopping client: %s", e)

        if self.owns_root:
            self.root.destroy()
        logger.info("EMG & Kinematics visualizer window closed.")


def run_visualizer(
    mode: str = "auto",
    data_path: str | None = None,
    device_address: str | None = None,
    model_path: str | None = None,
    enable_udp: bool = True,
    default_time_window: float = 0.1,
) -> None:
    """Launch the live sEMG and inferred kinematics monitor application.

    Parameters
    ----------
    mode : str, default="auto"
        Acquisition mode: "auto", "ble", or "mock".
    data_path : str, optional
        Path to custom `.npz` file for mock replay.
    device_address : str, optional
        Target Myo BLE MAC address.
    model_path : str, optional
        Path to pretrained TRR checkpoint.
    enable_udp : bool, default=True
        Whether to transmit controls to MuJoCo (:5005).
    default_time_window : float, default=0.1
        Initial duration of the visible rolling oscilloscope window in seconds.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting Myo sEMG & Inferred Kinematics Monitor (mode=%s)...", mode)

    force_mock = mode.lower() == "mock"
    mock_fallback = False

    client = BleMyoClient(
        device_address=device_address,
        raw_mode=True,
        mock_fallback=mock_fallback,
        force_mock=force_mock,
        data_path=data_path,
        discovery_timeout=3.0 if mode.lower() == "auto" else 8.0,
    )
    client.start()

    root = tk.Tk()
    _app = EMGVisualizerApp(
        client=client,
        model_path=model_path,
        enable_udp=enable_udp,
        default_time_window=default_time_window,
        root=root,
    )
    _ = _app
    root.mainloop()


if __name__ == "__main__":
    run_visualizer()
