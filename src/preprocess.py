"""Causal and batch digital signal preprocessing for sEMG channels."""

from typing import Sequence

import numpy as np
from scipy import signal


class Preprocess:
    """Causal digital signal filtering: power-line notch followed by 4th-order Butterworth bandpass."""

    def __init__(
        self,
        fs: int = 200,
        bandpass: Sequence[float] = (15, 95),
        notch: float = 50.0,
        notch_q: float = 30.0,
        channels: int = 8,
        bands: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        self.fs = fs
        self.bandpass_cutoffs = tuple(bandpass)
        self.notch_freq = notch
        self.notch_q = notch_q
        self.channels = channels

        self.notch = signal.tf2sos(*signal.iirnotch(notch, Q=notch_q, fs=fs))
        self.bandpass = signal.butter(4, bandpass, btype="bandpass", fs=fs, output="sos")

        default_bands = ((15.0, 40.0), (40.0, 65.0), (65.0, 95.0))
        self.bands = tuple((float(b[0]), float(b[1])) for b in (bands or default_bands))
        self.sos_bands = [signal.butter(4, b, btype="bandpass", fs=fs, output="sos") for b in self.bands]

        self.zi_notch: np.ndarray | None = None
        self.zi_notch_bands: np.ndarray | None = None
        self.zi_bandpass: np.ndarray | None = None
        self.zi_bands: list[np.ndarray] | None = None
        self.reset_state(channels=channels)

    def set_bands(self, bands: Sequence[tuple[float, float]]) -> None:
        """Update the 3-band filter bank cutoffs and reset band filter states."""
        self.bands = tuple((float(b[0]), float(b[1])) for b in bands)
        self.sos_bands = [signal.butter(4, b, btype="bandpass", fs=self.fs, output="sos") for b in self.bands]
        self.reset_state(channels=self.channels)

    def reset_state(self, channels: int | None = None) -> None:
        """Reset internal filter delay states to zero for all channels."""
        if channels is not None:
            self.channels = channels

        n_notch = self.notch.shape[0]
        n_bp = self.bandpass.shape[0]

        self.zi_notch = np.zeros((n_notch, self.channels, 2), dtype=np.float64)
        self.zi_notch_bands = np.zeros((n_notch, self.channels, 2), dtype=np.float64)
        self.zi_bandpass = np.zeros((n_bp, self.channels, 2), dtype=np.float64)
        self.zi_bands = [np.zeros((sos.shape[0], self.channels, 2), dtype=np.float64) for sos in self.sos_bands]

    def filter_chunk(self, emg_chunk: np.ndarray) -> np.ndarray:
        """Filter a streaming chunk of EMG data causally, updating internal filter states."""
        emg = np.asarray(emg_chunk, dtype=np.float64)
        is_1d = emg.ndim == 1
        if is_1d:
            emg = emg[:, np.newaxis]

        num_channels = emg.shape[0]
        if self.zi_notch is None or self.zi_bandpass is None or self.channels != num_channels:
            self.reset_state(channels=num_channels)

        notch_out, self.zi_notch = signal.sosfilt(self.notch, emg, axis=-1, zi=self.zi_notch)
        bp_out, self.zi_bandpass = signal.sosfilt(self.bandpass, notch_out, axis=-1, zi=self.zi_bandpass)

        out = bp_out.astype(np.float32)
        return out[:, 0] if is_1d else out

    def filter_bands(self, emg_chunk: np.ndarray) -> np.ndarray:
        """Filter a streaming chunk of EMG data across the 3 frequency bands."""
        emg = np.asarray(emg_chunk, dtype=np.float64)
        is_1d = emg.ndim == 1
        if is_1d:
            emg = emg[:, np.newaxis]

        num_channels = emg.shape[0]
        if self.zi_notch_bands is None or self.zi_bands is None or self.channels != num_channels:
            self.reset_state(channels=num_channels)

        notch_out, self.zi_notch_bands = signal.sosfilt(self.notch, emg, axis=-1, zi=self.zi_notch_bands)
        band_outputs = []
        for i, sos in enumerate(self.sos_bands):
            b_out, self.zi_bands[i] = signal.sosfilt(sos, notch_out, axis=-1, zi=self.zi_bands[i])
            band_outputs.append(b_out.astype(np.float32))

        out = np.stack(band_outputs, axis=0)
        return out[:, :, 0] if is_1d else out

    def __call__(self, emg: np.ndarray) -> np.ndarray:
        """Offline batch filter raw EMG sampled at self.fs."""
        emg_arr = np.asarray(emg, dtype=np.float64)
        notch_out = signal.sosfilt(self.notch, emg_arr, axis=-1)
        bp_out = signal.sosfilt(self.bandpass, notch_out, axis=-1)
        return bp_out.astype(np.float32)
