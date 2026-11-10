"""Kinematic mapping from decoded 15-DOF finger joint angles to MuJoCo Shadow Hand controls."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "params.yaml"

CANONICAL_DEFAULTS: dict[str, Any] = {
    "mode": "linear",
    "smoothing_alpha": 0.35,
    "headroom": 1.35,
    "curve": "smooth",
    "rest_angle_deg": 180.0,
    "neutral_wrist": {"wrj2": 0.750, "wrj1": 0.588},
    "neutral_abduction": 0.500,
    "flex_ranges": {
        "thumb_cmc": 60.0,
        "thumb_mcp": 80.0,
        "thumb_ip": 80.0,
        "mcp": 90.0,
        "pip": 110.0,
        "dip": 90.0,
    },
    "tendon_weights": {
        "pip": 0.6,
        "dip": 0.4,
    },
    "actuator_scales": {
        "thumb_thj5_base": 0.500,
        "thumb_thj5_gain": 0.350,
        "thumb_thj4_gain": 0.850,
        "thumb_thj2_base": 0.500,
        "thumb_thj2_gain": 0.450,
        "thumb_thj1_base": 0.143,
        "thumb_thj1_gain": 0.857,
        "finger_mcp_base": 0.143,
        "finger_mcp_gain": 0.857,
        "little_lfj5_gain": 0.400,
    },
    "empirical_bounds": {
        "extension": [
            156.2,
            167.7,
            164.9,
            152.3,
            141.2,
            171.2,
            161.1,
            133.3,
            168.1,
            147.8,
            122.6,
            165.3,
            123.3,
            134.3,
            163.6,
        ],
        "flexion": [
            155.6,
            160.6,
            155.3,
            141.9,
            126.3,
            156.0,
            146.8,
            116.4,
            152.6,
            130.7,
            106.9,
            149.9,
            106.0,
            117.6,
            149.4,
        ],
    },
}


def _load_kinematics_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve kinematics config from params dict, params.yaml, or canonical defaults."""
    if params is not None:
        if "kinematics" in params and isinstance(params["kinematics"], dict):
            return {**CANONICAL_DEFAULTS, **params["kinematics"]}
        return {**CANONICAL_DEFAULTS, **params}

    if DEFAULT_CONFIG_PATH.is_file():
        try:
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if loaded and isinstance(loaded, dict) and "kinematics" in loaded:
                return {**CANONICAL_DEFAULTS, **loaded["kinematics"]}
        except Exception:
            pass

    return CANONICAL_DEFAULTS


def map_angles_to_controls(
    angles_deg: np.ndarray | list[float] | tuple[float, ...],
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Translate 15 decoded finger joint angles (deg) to 20 normalized [0, 1] actuator commands."""
    angles = np.asarray(angles_deg, dtype=np.float32)
    if angles.size != 15:
        raise ValueError(f"angles_deg must contain exactly 15 elements, got {angles.size}")
    angles = angles.reshape(15)

    cfg = _load_kinematics_config(params)

    rest_angle = float(cfg.get("rest_angle_deg", 180.0))
    neutral_wrist = cfg.get("neutral_wrist", {"wrj2": 0.750, "wrj1": 0.588})
    if isinstance(neutral_wrist, (list, tuple)):
        wrj2_neutral, wrj1_neutral = float(neutral_wrist[0]), float(neutral_wrist[1])
    elif isinstance(neutral_wrist, dict):
        wrj2_neutral = float(neutral_wrist.get("wrj2", 0.750))
        wrj1_neutral = float(neutral_wrist.get("wrj1", 0.588))
    else:
        wrj2_neutral, wrj1_neutral = 0.750, 0.588

    neutral_abduction = float(cfg.get("neutral_abduction", 0.500))

    flex_ranges = cfg.get("flex_ranges", {})
    th_cmc_range = float(flex_ranges.get("thumb_cmc", 60.0))
    th_mcp_range = float(flex_ranges.get("thumb_mcp", 80.0))
    th_ip_range = float(flex_ranges.get("thumb_ip", 80.0))
    mcp_range = float(flex_ranges.get("mcp", 90.0))
    pip_range = float(flex_ranges.get("pip", 110.0))
    dip_range = float(flex_ranges.get("dip", 90.0))

    tendon_weights = cfg.get("tendon_weights", {})
    w_pip = float(tendon_weights.get("pip", 0.6))
    w_dip = float(tendon_weights.get("dip", 0.4))

    scales = cfg.get("actuator_scales", {})
    thj5_base = float(scales.get("thumb_thj5_base", 0.500))
    thj5_gain = float(scales.get("thumb_thj5_gain", 0.350))
    thj4_gain = float(scales.get("thumb_thj4_gain", 0.850))
    thj2_base = float(scales.get("thumb_thj2_base", 0.500))
    thj2_gain = float(scales.get("thumb_thj2_gain", 0.450))
    thj1_base = float(scales.get("thumb_thj1_base", 0.143))
    thj1_gain = float(scales.get("thumb_thj1_gain", 0.857))
    finger_mcp_base = float(scales.get("finger_mcp_base", 0.143))
    finger_mcp_gain = float(scales.get("finger_mcp_gain", 0.857))
    lfj5_gain = float(scales.get("little_lfj5_gain", 0.400))

    mode = str(cfg.get("mode", "linear")).lower()
    if mode == "empirical":
        bounds = cfg.get("empirical_bounds", {})
        ext_cfg = bounds.get("extension", CANONICAL_DEFAULTS["empirical_bounds"]["extension"])
        flex_cfg = bounds.get("flexion", CANONICAL_DEFAULTS["empirical_bounds"]["flexion"])
        ext_arr = np.asarray(ext_cfg, dtype=np.float32)
        flex_arr = np.asarray(flex_cfg, dtype=np.float32)
        span = np.maximum(ext_arr - flex_arr, 1.0)
        headroom = float(cfg.get("headroom", 1.35))
        curve = str(cfg.get("curve", "smooth")).lower()

        raw_fracs = np.clip((ext_arr - angles) / (span * headroom), 0.0, 1.0)
        fracs = 3.0 * (raw_fracs**2) - 2.0 * (raw_fracs**3) if curve == "smooth" else raw_fracs

        f_th_cmc, f_th_mcp, f_th_ip = float(fracs[0]), float(fracs[1]), float(fracs[2])
        f_ff_mcp = float(fracs[3])
        f_ff_0 = w_pip * float(fracs[4]) + w_dip * float(fracs[5])
        f_mf_mcp = float(fracs[6])
        f_mf_0 = w_pip * float(fracs[7]) + w_dip * float(fracs[8])
        f_rf_mcp = float(fracs[9])
        f_rf_0 = w_pip * float(fracs[10]) + w_dip * float(fracs[11])
        f_lf_mcp = float(fracs[12])
        f_lf_0 = w_pip * float(fracs[13]) + w_dip * float(fracs[14])
    else:

        def flex(angle: float, flex_range: float) -> float:
            if flex_range <= 0:
                return 0.0
            return float(np.clip((rest_angle - angle) / flex_range, 0.0, 1.0))

        f_th_cmc = flex(float(angles[0]), th_cmc_range)
        f_th_mcp = flex(float(angles[1]), th_mcp_range)
        f_th_ip = flex(float(angles[2]), th_ip_range)
        f_ff_mcp = flex(float(angles[3]), mcp_range)
        f_ff_0 = w_pip * flex(float(angles[4]), pip_range) + w_dip * flex(float(angles[5]), dip_range)
        f_mf_mcp = flex(float(angles[6]), mcp_range)
        f_mf_0 = w_pip * flex(float(angles[7]), pip_range) + w_dip * flex(float(angles[8]), dip_range)
        f_rf_mcp = flex(float(angles[9]), mcp_range)
        f_rf_0 = w_pip * flex(float(angles[10]), pip_range) + w_dip * flex(float(angles[11]), dip_range)
        f_lf_mcp = flex(float(angles[12]), mcp_range)
        f_lf_0 = w_pip * flex(float(angles[13]), pip_range) + w_dip * flex(float(angles[14]), dip_range)

    ctrl = np.zeros(20, dtype=np.float32)
    ctrl[0] = wrj2_neutral
    ctrl[1] = wrj1_neutral

    ctrl[2] = np.clip(thj5_base + thj5_gain * f_th_cmc, 0.0, 1.0)
    ctrl[3] = np.clip(thj4_gain * f_th_cmc, 0.0, 1.0)
    ctrl[4] = neutral_abduction
    ctrl[5] = np.clip(thj2_base + thj2_gain * f_th_mcp, 0.0, 1.0)
    ctrl[6] = np.clip(thj1_base + thj1_gain * f_th_ip, 0.0, 1.0)

    ctrl[7] = neutral_abduction
    ctrl[8] = np.clip(finger_mcp_base + finger_mcp_gain * f_ff_mcp, 0.0, 1.0)
    ctrl[9] = np.clip(f_ff_0, 0.0, 1.0)

    ctrl[10] = neutral_abduction
    ctrl[11] = np.clip(finger_mcp_base + finger_mcp_gain * f_mf_mcp, 0.0, 1.0)
    ctrl[12] = np.clip(f_mf_0, 0.0, 1.0)

    ctrl[13] = neutral_abduction
    ctrl[14] = np.clip(finger_mcp_base + finger_mcp_gain * f_rf_mcp, 0.0, 1.0)
    ctrl[15] = np.clip(f_rf_0, 0.0, 1.0)

    ctrl[16] = np.clip(lfj5_gain * f_lf_mcp, 0.0, 1.0)
    ctrl[17] = neutral_abduction
    ctrl[18] = np.clip(finger_mcp_base + finger_mcp_gain * f_lf_mcp, 0.0, 1.0)
    ctrl[19] = np.clip(f_lf_0, 0.0, 1.0)

    return np.clip(ctrl, 0.0, 1.0).astype(np.float32)
