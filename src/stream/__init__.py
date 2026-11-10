"""Streaming acquisition module for Myo armband BLE and mock replay streaming."""

from src.stream.ble_myo import (
    CMD_NEVER_SLEEP,
    CMD_NORMAL_SLEEP,
    CMD_SET_EMG_FILTERED,
    CMD_SET_EMG_RAW,
    MYO_EMG_HANDLES,
    MYO_EMG_UUIDS,
    UUID_COMMAND,
    UUID_CONTROL_SVC,
    UUID_EMG0_DATA,
    UUID_EMG1_DATA,
    UUID_EMG2_DATA,
    UUID_EMG3_DATA,
    UUID_EMG_SVC,
    BleMyoClient,
    unpack_emg_notification,
)
from src.stream.mock_myo import MockMyoStreamer
from src.stream.ring_buffer import EMGRingBuffer
from src.stream.worker import LatencyMonitor, StepResult, StreamingInferenceWorker

__all__ = [
    "EMGRingBuffer",
    "MockMyoStreamer",
    "BleMyoClient",
    "unpack_emg_notification",
    "StreamingInferenceWorker",
    "LatencyMonitor",
    "StepResult",
    "UUID_CONTROL_SVC",
    "UUID_COMMAND",
    "UUID_EMG_SVC",
    "UUID_EMG0_DATA",
    "UUID_EMG1_DATA",
    "UUID_EMG2_DATA",
    "UUID_EMG3_DATA",
    "MYO_EMG_UUIDS",
    "MYO_EMG_HANDLES",
    "CMD_SET_EMG_RAW",
    "CMD_SET_EMG_FILTERED",
    "CMD_NEVER_SLEEP",
    "CMD_NORMAL_SLEEP",
]
