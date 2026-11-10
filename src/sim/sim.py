import json
import socket
import struct
import time
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np
from mjviser import Viewer

from src.sim.config import HAND, MODEL, UDP_HOST, UDP_PORT

SPECS_PATH = Path(__file__).resolve().parent.parent.parent / "shadow_hand_specs.json"


def load_shadow_hand() -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mm.load(MODEL, HAND)
    d = mujoco.MjData(m)
    return m, d


model, data = load_shadow_hand()

if SPECS_PATH.is_file():
    with open(SPECS_PATH, "r", encoding="utf-8") as f:
        actuator_specs = json.load(f)["actuators"]
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    ctrl_min = np.array([actuator_specs[n]["min"] for n in actuator_names], dtype=np.float64)
    ctrl_max = np.array([actuator_specs[n]["max"] for n in actuator_names], dtype=np.float64)
else:
    ctrl_min = model.actuator_ctrlrange[:, 0].astype(np.float64)
    ctrl_max = model.actuator_ctrlrange[:, 1].astype(np.float64)

sock: socket.socket | None = None
payload_size = struct.calcsize("20f")

calib_html_handle = None
calib_btn_handle = None
local_calib_start: float = 0.0
last_remote_calib_ts: float = 0.0


def decode_controls(normalized: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Map normalized [0, 1] controls to each actuator's physical range."""
    normalized = np.clip(np.asarray(normalized), 0.0, 1.0)
    return ctrl_min + normalized * (ctrl_max - ctrl_min)


def _format_calib_banner(phase: int, title: str, subtitle: str) -> str:
    colors = {
        1: ("#3a2a00", "#ffd600", "#ffd600"),
        2: ("#4a1a00", "#ff9100", "#ff9100"),
        3: ("#003820", "#00e676", "#00e676"),
    }
    bg, border, fg = colors.get(phase, ("#1e222d", "#3a4052", "#d0d4e0"))
    return (
        f'<div style="background:{bg}; border:1px solid {border}; border-radius:6px; '
        f'padding:8px 10px; margin:4px 0 8px 0; font-size:0.82em; line-height:1.35;">'
        f'<div style="color:{fg}; font-weight:700;">{title}</div>'
        f'<div style="color:#b0b6c6; font-size:0.9em; margin-top:2px;">{subtitle}</div>'
        f"</div>"
    )


def trigger_calibration(_event=None) -> None:
    global local_calib_start
    local_calib_start = time.perf_counter()
    msg = json.dumps({"cmd": "calibrate"}).encode("utf-8")
    if sock is not None:
        for port in (5006, 5007, 5008, 5009):
            try:
                sock.sendto(msg, ("127.0.0.1", port))
            except Exception:
                pass
    if calib_html_handle is not None:
        calib_html_handle.content = _format_calib_banner(
            1,
            "CALIB (1/2): EXTEND HAND OPEN FLAT (2.0s)",
            "Hold all 5 fingers extended wide open",
        )


def custom_step_fn(m, d) -> None:
    global local_calib_start, last_remote_calib_ts
    latest_controls_packet = None
    latest_status_msg = None

    if sock is not None:
        while True:
            try:
                pkt, _ = sock.recvfrom(4096)
            except Exception:
                break

            if len(pkt) == payload_size:
                latest_controls_packet = pkt
            elif pkt.startswith(b"{"):
                try:
                    info = json.loads(pkt.decode("utf-8"))
                    if info.get("type") == "calib_status":
                        latest_status_msg = info
                except Exception:
                    pass

    if latest_controls_packet is not None:
        controls = struct.unpack("20f", latest_controls_packet)
        d.ctrl[:20] = decode_controls(controls)

    now = time.perf_counter()
    if latest_status_msg is not None and calib_html_handle is not None:
        last_remote_calib_ts = now
        phase = int(latest_status_msg.get("phase", 0))
        text = str(latest_status_msg.get("text", "Calibrating..."))
        detail = str(latest_status_msg.get("detail", ""))
        calib_html_handle.content = _format_calib_banner(phase, text, detail)
        if phase == 3:
            local_calib_start = 0.0
    elif local_calib_start > 0.0 and (now - last_remote_calib_ts) > 0.35 and calib_html_handle is not None:
        elapsed = now - local_calib_start
        if elapsed < 2.0:
            rem = max(0.0, 2.0 - elapsed)
            calib_html_handle.content = _format_calib_banner(
                1,
                f"CALIB (1/2): EXTEND HAND OPEN FLAT ({rem:3.1f}s)",
                "Hold all 5 fingers extended wide open",
            )
        elif elapsed < 4.0:
            rem = max(0.0, 4.0 - elapsed)
            calib_html_handle.content = _format_calib_banner(
                2,
                f"CALIB (2/2): MAKE A TIGHT FIST ({rem:3.1f}s)",
                "Close hand into a firm fist",
            )
        else:
            local_calib_start = 0.0
            calib_html_handle.content = _format_calib_banner(
                3,
                "CALIBRATION COMPLETE!",
                "Pod rotation & gains aligned",
            )

    mujoco.mj_step(m, d)


def main() -> None:
    global sock, calib_html_handle, calib_btn_handle
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.setblocking(False)

    print(f"Launching MuJoCo Shadow Hand Viewer on http://localhost:8080 (UDP listener on :{UDP_PORT})...")
    viewer = Viewer(model, data, step_fn=custom_step_fn)
    with viewer._server.gui.add_folder("Myo sEMG Calibration", order=-10):
        calib_html_handle = viewer._server.gui.add_html(
            _format_calib_banner(
                0,
                "Myo Calibration Ready",
                "Click below: 2s Open Hand -> 2s Tight Fist",
            )
        )
        calib_btn_handle = viewer._server.gui.add_button(
            "Calibrate Myo (Open -> Fist)",
            color="amber",
            hint="Run 4-second guided calibration (2s Open Hand -> 2s Tight Fist)",
        )
        calib_btn_handle.on_click(trigger_calibration)
    viewer.run()


if __name__ == "__main__":
    main()
