#!/usr/bin/env python3
"""CLI script to launch the live 8-channel Myo sEMG and inferred kinematics monitor window."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.emg_window import run_visualizer  # noqa: E402


def _resolve_default_mac() -> str | None:
    cfg_file = PROJECT_ROOT / "src" / "params.yaml"
    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            mac = cfg.get("ble", {}).get("device_address")
            if mac and str(mac).strip():
                return str(mac).strip()
        except Exception:
            pass
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the live Myo 8-channel sEMG and inferred kinematics monitor window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "ble", "mock"],
        default="auto",
        help="Acquisition source: 'ble'/'auto' (Myo BLE) or 'mock' (replay/synthetic).",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to an EMG .npz file for mock replay.")
    parser.add_argument("--mac", type=str, default=None, help="Target Myo MAC address.")
    parser.add_argument("--weights", type=str, default="models/best_model.pt", help="Path to model checkpoint.")
    parser.add_argument("--no-udp", action="store_true", help="Disable UDP control transmission (:5005).")
    parser.add_argument("--window", type=float, default=0.1, help="Oscilloscope time window in seconds.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_visualizer(
        mode=args.mode,
        data_path=args.data,
        device_address=args.mac or _resolve_default_mac(),
        model_path=args.weights,
        enable_udp=not args.no_udp,
        default_time_window=args.window,
    )


if __name__ == "__main__":
    main()
