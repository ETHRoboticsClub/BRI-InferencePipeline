"""Unit and regression tests for the real-time EMG visualizer and kinematics monitor GUI window."""

import os
import sys
import time
from pathlib import Path

# Ensure Windows TCL/TK libraries can always be located
if "TCL_LIBRARY" not in os.environ or "TK_LIBRARY" not in os.environ:
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    tcl_cand = Path(base_prefix) / "tcl"
    if (tcl_cand / "tcl8.6").exists():
        os.environ.setdefault("TCL_LIBRARY", str(tcl_cand / "tcl8.6"))
    if (tcl_cand / "tk8.6").exists():
        os.environ.setdefault("TK_LIBRARY", str(tcl_cand / "tk8.6"))

import tkinter as tk

import numpy as np
import pytest

from src.stream.ble_myo import BleMyoClient
from src.visualization.emg_window import EMGVisualizerApp


@pytest.fixture(scope="module")
def headless_root():
    """Create a hidden Tkinter root window for headless testing."""
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def mock_client():
    """Create and start a Mock BleMyoClient."""
    client = BleMyoClient(force_mock=True)
    client.start()
    time.sleep(0.15)  # Allow initial buffer fill
    yield client
    client.stop()


def test_visualizer_initialization(headless_root, mock_client):
    """Verify that EMGVisualizerApp constructs all widgets, lines, and TRR worker properly."""
    app = EMGVisualizerApp(
        client=mock_client,
        enable_udp=False,
        fs=200,
        default_time_window=2.0,
        root=headless_root,
    )

    assert len(app.emg_lines) == 8
    assert len(app.finger_lines) == 5
    assert len(app.finger_deg_labels) >= 5
    assert len(app.rms_canvases) == 8
    assert len(app.rms_value_labels) == 8
    assert app.filter_active is True
    assert app.is_paused is False
    assert app.time_window == 2.0
    assert app.worker is not None

    app.on_close()


def test_visualizer_render_and_metrics(headless_root, mock_client):
    """Verify signal rendering, line updates, inferred angles, and RMS meter calculations."""
    app = EMGVisualizerApp(
        client=mock_client,
        enable_udp=False,
        fs=200,
        default_time_window=1.0,
        root=headless_root,
    )

    # Let buffer accumulate
    time.sleep(0.2)

    # Trigger render
    app._render_signals_and_kinematics()
    app._refresh_status_badges()

    # Verify EMG lines received valid non-empty data
    for line in app.emg_lines:
        y = line.get_ydata()
        assert len(y) == 200  # 1.0s @ 200 Hz
        assert np.all(np.isfinite(y))

    # Verify Finger angle lines received valid non-empty data
    for finger, line in app.finger_lines.items():
        y = line.get_ydata()
        assert len(y) == 200
        assert np.all(np.isfinite(y))
        assert np.all((y >= 0.0) & (y <= 200.0))

    # Verify toggle pause
    app.toggle_pause()
    assert app.is_paused is True
    app.toggle_pause()
    assert app.is_paused is False

    # Verify toggle filter
    app.toggle_filter()
    assert app.filter_active is False
    app._render_signals_and_kinematics()
    app.toggle_filter()
    assert app.filter_active is True

    # Verify view mode changes
    app.view_combo.set("Angles Only")
    app._on_view_changed()
    assert not app.ax_emg.get_visible()
    assert app.ax_angles.get_visible()

    app.view_combo.set("Split (EMG + Angles)")
    app._on_view_changed()
    assert app.ax_emg.get_visible()
    assert app.ax_angles.get_visible()

    # Verify UDP toggle
    app.udp_var.set(False)
    app._on_udp_toggled()
    assert app.enable_udp is False

    app.on_close()
