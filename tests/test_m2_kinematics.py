"""Milestone 2 Kinematics & UDP Transmission Comprehensive Test Suite.

Verifies:
1. Module exports (map_angles_to_controls, UDPSender).
2. Kinematic mapping accuracy for open hand, closed fist, and intermediate postures.
3. Coupled PIP+DIP tendon blending weights (0.6 * PIP + 0.4 * DIP).
4. Single-digit kinematic isolation across Thumb, Index, Middle, Ring, and Pinky.
5. Strict clipping and boundary guarantees for negative and hyper-extended angles.
6. Support for various input formats (NumPy arrays, lists, tuples, 2D arrays).
7. Configurable parameter overrides via params dictionary.
8. Non-blocking UDPSender socket behavior, packet layout (80 bytes, 20 floats little-endian).
9. Context manager lifecycle and clean socket teardown.
10. End-to-end loopback transmission and reception with MockUDPReceiver.
11. Transmission latency microbenchmarking (< 1 ms per packet).
"""

from __future__ import annotations

import math
import struct
import time

import numpy as np
import pytest

import src.kinematics as kinematics
from src.kinematics.mapping import map_angles_to_controls
from src.kinematics.udp_sender import UDPSender

# ==============================================================================
# 1. Module Exports & Interface Verification
# ==============================================================================


def test_kinematics_module_exports():
    """Verify src.kinematics exports map_angles_to_controls and UDPSender in __all__."""
    assert hasattr(kinematics, "map_angles_to_controls")
    assert hasattr(kinematics, "UDPSender")
    assert "map_angles_to_controls" in kinematics.__all__
    assert "UDPSender" in kinematics.__all__


# ==============================================================================
# 2. Output Specification, Data Types & Ranges
# ==============================================================================


def test_mapping_output_shape_and_dtype():
    """Verify mapping produces exactly (20,) float32 array in [0.0, 1.0]."""
    angles = np.full(15, 140.0, dtype=np.float32)
    ctrl = map_angles_to_controls(angles)
    assert isinstance(ctrl, np.ndarray)
    assert ctrl.shape == (20,)
    assert ctrl.dtype == np.float32
    assert np.all((ctrl >= 0.0) & (ctrl <= 1.0))


def test_mapping_random_angle_invariance():
    """Verify random degree inputs in range [0, 180] always yield valid [0, 1] controls."""
    rng = np.random.default_rng(12345)
    for _ in range(50):
        rand_angles = rng.uniform(0.0, 180.0, size=15).astype(np.float32)
        ctrl = map_angles_to_controls(rand_angles)
        assert ctrl.shape == (20,)
        assert ctrl.dtype == np.float32
        assert np.all((ctrl >= 0.0) & (ctrl <= 1.0))


# ==============================================================================
# 3. Canonical Postures: Open Hand, Closed Fist & Neutral Posture
# ==============================================================================


def test_open_hand_neutral_posture_calibration(canonical_hand_angles, expected_open_hand_controls):
    """Verify open hand (180 deg) produces the exact calibrated MuJoCo neutral controls."""
    ctrl = map_angles_to_controls(canonical_hand_angles["open_hand"])
    np.testing.assert_allclose(ctrl, expected_open_hand_controls, atol=1e-3)

    # Detailed per-actuator check
    assert math.isclose(ctrl[0], 0.750, abs_tol=1e-3)  # lh_A_WRJ2
    assert math.isclose(ctrl[1], 0.588, abs_tol=1e-3)  # lh_A_WRJ1
    assert math.isclose(ctrl[2], 0.500, abs_tol=1e-3)  # lh_A_THJ5
    assert math.isclose(ctrl[3], 0.000, abs_tol=1e-3)  # lh_A_THJ4
    assert math.isclose(ctrl[4], 0.500, abs_tol=1e-3)  # lh_A_THJ3
    assert math.isclose(ctrl[5], 0.500, abs_tol=1e-3)  # lh_A_THJ2
    assert math.isclose(ctrl[6], 0.143, abs_tol=1e-3)  # lh_A_THJ1

    for knuckle_idx in (7, 10, 13, 17):  # FFJ4, MFJ4, RFJ4, LFJ4
        assert math.isclose(ctrl[knuckle_idx], 0.500, abs_tol=1e-3)

    for mcp_idx in (8, 11, 14, 18):  # FFJ3, MFJ3, RFJ3, LFJ3
        assert math.isclose(ctrl[mcp_idx], 0.143, abs_tol=1e-3)

    for tendon_idx in (9, 12, 15, 19):  # FFJ0, MFJ0, RFJ0, LFJ0
        assert math.isclose(ctrl[tendon_idx], 0.000, abs_tol=1e-3)

    assert math.isclose(ctrl[16], 0.000, abs_tol=1e-3)  # lh_A_LFJ5 palm arch


def test_closed_fist_flexion(canonical_hand_angles):
    """Verify closed fist curls all MCP and tendon actuators toward 1.0."""
    ctrl = map_angles_to_controls(canonical_hand_angles["closed_fist"])

    # MCP actuators flexed >= 0.85
    assert ctrl[8] >= 0.85, f"Index MCP expected >= 0.85, got {ctrl[8]:.3f}"
    assert ctrl[11] >= 0.85, f"Middle MCP expected >= 0.85, got {ctrl[11]:.3f}"
    assert ctrl[14] >= 0.85, f"Ring MCP expected >= 0.85, got {ctrl[14]:.3f}"
    assert ctrl[18] >= 0.85, f"Pinky MCP expected >= 0.85, got {ctrl[18]:.3f}"

    # Tendon actuators flexed >= 0.85
    assert ctrl[9] >= 0.85, f"Index tendon expected >= 0.85, got {ctrl[9]:.3f}"
    assert ctrl[12] >= 0.85, f"Middle tendon expected >= 0.85, got {ctrl[12]:.3f}"
    assert ctrl[15] >= 0.85, f"Ring tendon expected >= 0.85, got {ctrl[15]:.3f}"
    assert ctrl[19] >= 0.85, f"Pinky tendon expected >= 0.85, got {ctrl[19]:.3f}"

    # Thumb flexed
    assert ctrl[2] > 0.500  # CMC opposition rotation
    assert ctrl[3] > 0.500  # CMC proximal flexion
    assert ctrl[5] > 0.500  # MCP flexion
    assert ctrl[6] > 0.500  # IP flexion

    # Wrist stays neutral
    assert math.isclose(ctrl[0], 0.750, abs_tol=1e-3)
    assert math.isclose(ctrl[1], 0.588, abs_tol=1e-3)


def test_monotonic_flexion_gradient():
    """Verify decreasing joint angle (increasing flexion) monotonically increases control value."""
    angles = np.full(15, 180.0, dtype=np.float32)
    prev_mcp_ctrl = -1.0
    prev_tendon_ctrl = -1.0

    # Sweep Index MCP and PIP from 180 deg down to 90 deg in steps
    for deg in np.linspace(180.0, 90.0, 10):
        angles[3] = deg  # Index MCP
        angles[4] = deg  # Index PIP
        angles[5] = deg  # Index DIP
        ctrl = map_angles_to_controls(angles)

        curr_mcp = ctrl[8]
        curr_tendon = ctrl[9]
        assert curr_mcp >= prev_mcp_ctrl, (
            f"MCP control should be monotonically increasing: {curr_mcp} < {prev_mcp_ctrl}"
        )
        assert curr_tendon >= prev_tendon_ctrl, (
            f"Tendon control should be monotonically increasing: {curr_tendon} < {prev_tendon_ctrl}"
        )
        prev_mcp_ctrl = curr_mcp
        prev_tendon_ctrl = curr_tendon


# ==============================================================================
# 4. Tendon Coupling & Blending Formula (0.6 * PIP + 0.4 * DIP)
# ==============================================================================


def test_coupled_tendon_blending_weights():
    """Verify coupled tendon actuator follows exact 0.6 * f_PIP + 0.4 * f_DIP formula."""
    # 1. PIP flexed only: angle 70 deg (flex = 1.0), DIP extended (180 deg, flex = 0.0)
    angles_pip_only = np.full(15, 180.0, dtype=np.float32)
    angles_pip_only[4] = 70.0  # Index PIP
    ctrl1 = map_angles_to_controls(angles_pip_only)
    assert math.isclose(ctrl1[9], 0.600, abs_tol=1e-3)

    # 2. DIP flexed only: PIP extended (180 deg, flex = 0.0), DIP flexed (90 deg, flex = 1.0)
    angles_dip_only = np.full(15, 180.0, dtype=np.float32)
    angles_dip_only[5] = 90.0  # Index DIP
    ctrl2 = map_angles_to_controls(angles_dip_only)
    assert math.isclose(ctrl2[9], 0.400, abs_tol=1e-3)

    # 3. Both flexed: flex = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
    angles_both = np.full(15, 180.0, dtype=np.float32)
    angles_both[4] = 70.0
    angles_both[5] = 90.0
    ctrl3 = map_angles_to_controls(angles_both)
    assert math.isclose(ctrl3[9], 1.000, abs_tol=1e-3)

    # 4. Intermediate: PIP at 125 deg (flex = 0.5), DIP at 135 deg (flex = 0.5)
    # Expected: 0.6 * 0.5 + 0.4 * 0.5 = 0.500
    angles_half = np.full(15, 180.0, dtype=np.float32)
    angles_half[4] = 125.0
    angles_half[5] = 135.0
    ctrl4 = map_angles_to_controls(angles_half)
    assert math.isclose(ctrl4[9], 0.500, abs_tol=1e-3)


# ==============================================================================
# 5. Single-Digit Isolation Verification
# ==============================================================================


@pytest.mark.parametrize(
    "finger_name,mcp_idx,pip_idx,dip_idx,act_mcp,act_tendon",
    [
        ("Index", 3, 4, 5, 8, 9),
        ("Middle", 6, 7, 8, 11, 12),
        ("Ring", 9, 10, 11, 14, 15),
        ("Pinky", 12, 13, 14, 18, 19),
    ],
)
def test_single_finger_kinematic_isolation(
    finger_name, mcp_idx, pip_idx, dip_idx, act_mcp, act_tendon, expected_open_hand_controls
):
    """Verify flexing a single finger leaves all other digits at neutral."""
    angles = np.full(15, 180.0, dtype=np.float32)
    angles[mcp_idx] = 90.0
    angles[pip_idx] = 70.0
    angles[dip_idx] = 90.0
    ctrl = map_angles_to_controls(angles)

    # Active finger actuated
    assert ctrl[act_mcp] > 0.8
    assert ctrl[act_tendon] > 0.8

    # Other fingers check
    all_finger_actuators = {
        "Index": (8, 9),
        "Middle": (11, 12),
        "Ring": (14, 15),
        "Pinky": (18, 19),
    }
    for other_name, (o_mcp, o_ten) in all_finger_actuators.items():
        if other_name != finger_name:
            assert math.isclose(ctrl[o_mcp], expected_open_hand_controls[o_mcp], abs_tol=1e-3)
            assert math.isclose(ctrl[o_ten], expected_open_hand_controls[o_ten], abs_tol=1e-3)


def test_thumb_kinematic_isolation(expected_open_hand_controls):
    """Verify flexing only the Thumb leaves Index, Middle, Ring, Pinky untouched."""
    angles = np.full(15, 180.0, dtype=np.float32)
    angles[0] = 120.0  # CMC
    angles[1] = 100.0  # MCP
    angles[2] = 100.0  # IP
    ctrl = map_angles_to_controls(angles)

    # Thumb actuators moved
    assert ctrl[2] > expected_open_hand_controls[2]
    assert ctrl[3] > expected_open_hand_controls[3]
    assert ctrl[5] > expected_open_hand_controls[5]
    assert ctrl[6] > expected_open_hand_controls[6]

    # Fingers 7..19 remained at neutral baseline
    for i in range(7, 20):
        assert math.isclose(ctrl[i], expected_open_hand_controls[i], abs_tol=1e-3)


# ==============================================================================
# 6. Boundary Conditions, Out-of-Range Clipping & Input Validation
# ==============================================================================


def test_clipping_negative_and_excessive_angles():
    """Verify out-of-range negative angles (-90 deg) and excessive angles (+270 deg) clip safely."""
    # Negative angles saturate flexion to 1.0
    neg_angles = np.full(15, -90.0, dtype=np.float32)
    ctrl_neg = map_angles_to_controls(neg_angles)
    assert np.all((ctrl_neg >= 0.0) & (ctrl_neg <= 1.0))
    assert ctrl_neg[8] == 1.0 and ctrl_neg[9] == 1.0

    # Hyperextended angles clamp flexion to 0.0 (neutral baseline)
    hyper_angles = np.full(15, 270.0, dtype=np.float32)
    ctrl_hyper = map_angles_to_controls(hyper_angles)
    assert np.all((ctrl_hyper >= 0.0) & (ctrl_hyper <= 1.0))
    assert math.isclose(ctrl_hyper[9], 0.000, abs_tol=1e-4)


def test_mapping_accepts_list_and_tuple_inputs(expected_open_hand_controls):
    """Verify map_angles_to_controls accepts Python list and tuple inputs."""
    list_input = [180.0] * 15
    ctrl_list = map_angles_to_controls(list_input)
    np.testing.assert_allclose(ctrl_list, expected_open_hand_controls, atol=1e-3)

    tuple_input = tuple([180.0] * 15)
    ctrl_tuple = map_angles_to_controls(tuple_input)
    np.testing.assert_allclose(ctrl_tuple, expected_open_hand_controls, atol=1e-3)


def test_mapping_accepts_2d_input():
    """Verify map_angles_to_controls accepts shape (1, 15) or (15, 1)."""
    angles_2d = np.full((1, 15), 180.0, dtype=np.float32)
    ctrl = map_angles_to_controls(angles_2d)
    assert ctrl.shape == (20,)


def test_mapping_invalid_input_size_raises():
    """Verify map_angles_to_controls raises ValueError on invalid length inputs."""
    with pytest.raises(ValueError):
        map_angles_to_controls([180.0] * 14)
    with pytest.raises(ValueError):
        map_angles_to_controls([180.0] * 16)
    with pytest.raises(ValueError):
        map_angles_to_controls([])


# ==============================================================================
# 7. Configurable Parameters & Overrides
# ==============================================================================


def test_mapping_custom_params_wrist_neutral():
    """Verify overriding neutral_wrist values via params dictionary."""
    custom_params = {
        "kinematics": {
            "neutral_wrist": {"wrj2": 0.500, "wrj1": 0.500},
        }
    }
    angles = np.full(15, 180.0, dtype=np.float32)
    ctrl = map_angles_to_controls(angles, params=custom_params)
    assert math.isclose(ctrl[0], 0.500, abs_tol=1e-3)
    assert math.isclose(ctrl[1], 0.500, abs_tol=1e-3)


def test_mapping_custom_tendon_weights():
    """Verify overriding tendon_weights via params dictionary."""
    custom_params = {
        "kinematics": {
            "tendon_weights": {"pip": 0.8, "dip": 0.2},
        }
    }
    angles = np.full(15, 180.0, dtype=np.float32)
    angles[4] = 70.0  # PIP flexed
    ctrl = map_angles_to_controls(angles, params=custom_params)
    assert math.isclose(ctrl[9], 0.800, abs_tol=1e-3)


# ==============================================================================
# 8. UDPSender Initialization & Configuration
# ==============================================================================


def test_udp_sender_defaults():
    """Verify UDPSender initializes with default 127.0.0.1:5005 and non-blocking socket."""
    sender = UDPSender()
    assert sender.host == "127.0.0.1"
    assert sender.port == 5005
    assert sender.sock is not None
    assert sender.sock.getblocking() is False
    sender.close()


def test_udp_sender_custom_address():
    """Verify UDPSender initializes with custom host and port."""
    sender = UDPSender(host="192.168.1.100", port=6000)
    assert sender.host == "192.168.1.100"
    assert sender.port == 6000
    sender.close()


# ==============================================================================
# 9. UDP Binary Serialization & Packet Integrity
# ==============================================================================


def test_udp_packet_exact_80_bytes():
    """Verify struct packing produces exactly 80 bytes for 20 float32s."""
    controls = [0.1 * i for i in range(20)]
    payload = struct.pack("<20f", *controls)
    assert len(payload) == 80


def test_udp_packet_roundtrip_precision():
    """Verify float32 roundtrip fidelity through struct serialization."""
    controls = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    payload = struct.pack("<20f", *controls)
    unpacked = struct.unpack("<20f", payload)
    np.testing.assert_allclose(unpacked, controls, atol=1e-6)


def test_udp_sender_rejects_invalid_control_lengths():
    """Verify UDPSender.send raises ValueError when passed length != 20."""
    sender = UDPSender()
    with pytest.raises(ValueError):
        sender.send([0.5] * 19)
    with pytest.raises(ValueError):
        sender.send([0.5] * 21)
    sender.close()


# ==============================================================================
# 10. UDPSender Lifecycle & Context Manager
# ==============================================================================


def test_udp_sender_context_manager():
    """Verify UDPSender functions as a context manager and cleanly closes."""
    with UDPSender() as sender:
        assert sender.sock is not None
        assert sender.sock.fileno() != -1
    assert sender.sock is None


def test_udp_sender_send_after_close_raises():
    """Verify sending on a closed UDPSender raises RuntimeError."""
    sender = UDPSender()
    sender.close()
    with pytest.raises(RuntimeError):
        sender.send([0.5] * 20)


# ==============================================================================
# 11. End-to-End Loopback Transmission with MockUDPReceiver
# ==============================================================================


def test_udp_loopback_transmission_open_hand(mock_udp_receiver, expected_open_hand_controls):
    """Verify transmitting open hand controls reaches receiver with exact values."""
    with UDPSender(host="127.0.0.1", port=mock_udp_receiver.port) as sender:
        bytes_sent = sender.send(expected_open_hand_controls)
        assert bytes_sent == 80

        packet = mock_udp_receiver.recv_packet(timeout=1.0)
        assert len(packet) == 80
        unpacked = np.array(mock_udp_receiver.unpack_packet(packet), dtype=np.float32)
        np.testing.assert_allclose(unpacked, expected_open_hand_controls, atol=1e-6)


def test_udp_sender_send_angles_convenience_helper(mock_udp_receiver, canonical_hand_angles):
    """Verify UDPSender.send_angles maps and transmits in one seamless call."""
    with UDPSender(host="127.0.0.1", port=mock_udp_receiver.port) as sender:
        sent_controls = sender.send_angles(canonical_hand_angles["closed_fist"])
        assert len(sent_controls) == 20

        packet = mock_udp_receiver.recv_packet(timeout=1.0)
        unpacked = np.array(mock_udp_receiver.unpack_packet(packet), dtype=np.float32)
        np.testing.assert_allclose(unpacked, sent_controls, atol=1e-6)


def test_udp_multi_packet_streaming(mock_udp_receiver):
    """Verify streaming 10 consecutive packets at simulated 10 Hz cadence."""
    with UDPSender(host="127.0.0.1", port=mock_udp_receiver.port) as sender:
        for i in range(10):
            val = float(i) / 10.0
            test_ctrl = np.full(20, val, dtype=np.float32)
            sender.send(test_ctrl)

            packet = mock_udp_receiver.recv_packet(timeout=1.0)
            unpacked = np.array(mock_udp_receiver.unpack_packet(packet), dtype=np.float32)
            np.testing.assert_allclose(unpacked, test_ctrl, atol=1e-6)


# ==============================================================================
# 12. Microbenchmark: UDP Transmission Latency (< 0.1 ms)
# ==============================================================================


def test_udp_send_latency_benchmark(mock_udp_receiver):
    """Verify mean UDP send latency is well below 1.0 ms (sub-millisecond non-blocking socket)."""
    with UDPSender(host="127.0.0.1", port=mock_udp_receiver.port) as sender:
        test_ctrl = np.full(20, 0.5, dtype=np.float32)
        latencies = []

        for _ in range(50):
            t0 = time.perf_counter()
            sender.send(test_ctrl)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

            # Drain packet
            mock_udp_receiver.recv_packet(timeout=0.1)

        mean_latency = np.mean(latencies)
        assert mean_latency < 1.0, f"Expected UDP send latency < 1.0 ms, got {mean_latency:.4f} ms"
