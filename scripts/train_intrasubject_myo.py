r"""Train an Intra-Subject TRR model from a recorded Myo + MediaPipe session (.npz).

Usage:
    .venv\Scripts\python.exe scripts/train_intrasubject_myo.py --data data/recordings/myo_intrasubject.npz
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.model import Model  # noqa: E402
from src.preprocess import Preprocess  # noqa: E402

INTRASUBJECT_BANDS: tuple[tuple[float, float], ...] = (
    (5.0, 40.0),
    (40.0, 70.0),
    (70.0, 95.0),
)


def apply_biomechanical_dip_coupling(angles_deg: np.ndarray) -> np.ndarray:
    """Softly couple 4 finger DIP joints (5, 8, 11, 14) with PIP flexion (4, 7, 10, 13)."""
    out = np.array(angles_deg, dtype=np.float32, copy=True)
    for pip_i, dip_i in ((4, 5), (7, 8), (10, 11), (13, 14)):
        coupled_dip = 180.0 - 0.68 * (180.0 - out[..., pip_i])
        out[..., dip_i] = np.clip(0.75 * out[..., dip_i] + 0.25 * coupled_dip, 0.0, 180.0)
    return out


def _fast_ledoit_wolf(X_centered: np.ndarray) -> np.ndarray:
    """Fast analytical Ledoit-Wolf shrinkage covariance for (p, n) zero-mean matrix."""
    p, n = X_centered.shape
    if n <= 1:
        return np.eye(p, dtype=np.float32)
    X_t = X_centered.T.astype(np.float64)
    emp_cov = (X_t.T @ X_t) / float(n)
    mu = float(np.trace(emp_cov)) / float(p)
    eye_p = np.eye(p, dtype=np.float64)
    delta = float(np.sum((emp_cov - mu * eye_p) ** 2)) / float(p)
    if delta <= 0.0:
        return emp_cov.astype(np.float32)
    X2 = X_t**2
    beta_0 = (float(np.sum(X2.T @ X2)) / float(p * n * n)) - (float(np.sum(emp_cov**2)) / float(p * n))
    beta = min(delta, max(0.0, beta_0))
    shrinkage = beta / delta
    return ((1.0 - shrinkage) * emp_cov + shrinkage * mu * eye_p).astype(np.float32)


def compute_riemannian_mean_and_whitening(covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Riemannian geometric mean and inverse square root whitening W = C_mean^{-1/2}."""
    try:
        try:
            from pyriemann.geometry.mean import mean_covariance
        except ImportError:
            from pyriemann.utils.mean import mean_covariance

        sub_sample = covs[:: max(1, len(covs) // 500)]
        mean_cov = mean_covariance(sub_sample.astype(np.float64), metric="riemann").astype(np.float32)
    except Exception:
        mean_cov = np.mean(covs, axis=0).astype(np.float32)

    evals, evecs = np.linalg.eigh(mean_cov)
    evals = np.maximum(evals, 1e-5)
    inv_sqrt = (evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T).astype(np.float32)
    return mean_cov, inv_sqrt


def extract_intrasubject_dataset(
    emg: np.ndarray,
    targets: np.ndarray,
    session_lengths: Sequence[int] | np.ndarray | None = None,
    fs: int = 200,
    sub_len: int = 100,
    step: int = 20,
    seq_len: int = 10,
) -> dict[str, np.ndarray]:
    """Extract 10-step 124-feature Hybrid CMTS + Phasic/Tonic sequences and calibration matrices.

    Uses paper-matched 3-band decomposition ((5-40 Hz), (40-70 Hz), (70-95 Hz)), a 100-sample
    (500 ms) covariance window for lower SPD estimation variance at 200 Hz, and 16 channel-wise
    phasic/tonic move/hold envelope features from Band 0.
    """
    from scipy.signal import butter, filtfilt

    T_total = int(emg.shape[0])
    min_seq_samples = sub_len + (seq_len - 1) * step
    if session_lengths is None or len(session_lengths) == 0:
        raw_lengths = [T_total]
    else:
        raw_lengths = [int(x) for x in session_lengths if int(x) > 0]
        if sum(raw_lengths) != T_total or not raw_lengths:
            raw_lengths = [T_total]

    has_long_session = any(x >= 12000 for x in raw_lengths)  # >= 60s
    row, col = np.triu_indices(8)
    triu_weights = np.where(row == col, 1.0, np.sqrt(2.0)).astype(np.float32)
    eye_8 = np.eye(8, dtype=np.float32)
    b_lp, a_lp = butter(3, 3.5 / (0.5 * fs), btype="low")

    X_train_list: list[np.ndarray] = []
    Y_train_list: list[np.ndarray] = []
    X_val_list: list[np.ndarray] = []
    Y_val_list: list[np.ndarray] = []
    X_all_list: list[np.ndarray] = []
    Y_all_list: list[np.ndarray] = []
    val_session_slices: list[tuple[int, int, int]] = []

    latest_channel_std = np.zeros((3, 8, 1), dtype=np.float32)
    latest_inv_sqrt = np.zeros((3, 8, 8), dtype=np.float32)
    valid_session_count = 0
    val_offset = 0

    offset = 0
    for s_idx, s_len in enumerate(raw_lengths):
        s_emg = emg[offset : offset + s_len]
        s_tgt = targets[offset : offset + s_len]
        offset += s_len

        if s_len < min_seq_samples or (has_long_session and s_len < 12000):
            print(
                f"  [Skip] Session {s_idx + 1} is an aborted/short stub "
                f"({s_len} samples = {s_len / fs:.1f}s < 60s); excluding from training."
            )
            continue

        valid_session_count += 1
        if s_len > 100:
            s_tgt = np.clip(filtfilt(b_lp, a_lp, s_tgt, axis=0), 30.0, 180.0).astype(np.float32)

        prep = Preprocess(
            fs=fs,
            bandpass=[5.0, 95.0],
            notch=50.0,
            channels=8,
            bands=INTRASUBJECT_BANDS,
        )
        s_bands = prep.filter_bands(s_emg.T)  # (3, 8, s_len)

        K_s = (s_len - sub_len) // step + 1
        raw_dc = s_emg - np.mean(s_emg, axis=0, keepdims=True)
        sub_rms = np.array(
            [np.sqrt(np.mean(raw_dc[k * step : k * step + sub_len] ** 2)) for k in range(K_s)],
            dtype=np.float32,
        )
        active_thresh = float(np.percentile(sub_rms, 20.0))
        active_mask = sub_rms >= max(active_thresh, 1.5)
        if np.sum(active_mask) < 5:
            active_mask = np.ones(K_s, dtype=bool)

        s_channel_std = np.zeros((3, 8, 1), dtype=np.float32)
        s_inv_sqrt = np.zeros((3, 8, 8), dtype=np.float32)
        band_tangents: list[np.ndarray] = []
        log_rms_b0 = np.zeros((K_s, 8), dtype=np.float32)

        for b_idx in range(3):
            X_b = s_bands[b_idx]
            std_b = np.std(X_b, axis=-1, keepdims=True)
            s_channel_std[b_idx] = np.maximum(std_b, 1.5).astype(np.float32)

            covs_b = np.zeros((K_s, 8, 8), dtype=np.float32)
            for k in range(K_s):
                s0 = k * step
                sub = X_b[:, s0 : s0 + sub_len]
                sub_mean = np.mean(sub, axis=-1, keepdims=True)
                sub_scaled = (sub - sub_mean) / s_channel_std[b_idx]
                if b_idx == 0:
                    log_rms_b0[k] = np.log(np.sqrt(np.mean(sub_scaled**2, axis=-1)) + 1e-4)
                cov_k = _fast_ledoit_wolf(sub_scaled)
                covs_b[k] = cov_k + 1e-6 * eye_8

            _, W_b = compute_riemannian_mean_and_whitening(covs_b[active_mask])
            s_inv_sqrt[b_idx] = W_b

            cov_w = W_b[None, :, :] @ covs_b @ W_b.T[None, :, :]
            evals, evecs = np.linalg.eigh(cov_w)
            evals = np.maximum(evals, 1e-8)
            log_c = np.einsum("kij,kj,kjl->kil", evecs, np.log(evals), np.swapaxes(evecs, -1, -2))
            tan_b = (log_c[:, row, col] * triu_weights[None, :]).astype(np.float32)
            band_tangents.append(tan_b)

        latest_channel_std = s_channel_std
        latest_inv_sqrt = s_inv_sqrt

        move_cmd_b0 = np.zeros_like(log_rms_b0)
        move_cmd_b0[1:] = np.maximum(0.0, log_rms_b0[1:] - log_rms_b0[:-1])

        all_tangents_s = np.concatenate([*band_tangents, log_rms_b0, move_cmd_b0], axis=-1)  # (K_s, 124)
        N_s = K_s - seq_len + 1
        X_s = np.stack([all_tangents_s[i : i + seq_len] for i in range(N_s)], axis=0).astype(np.float32)
        target_idx = step * np.arange(N_s) + (seq_len - 1) * step + sub_len - 1
        Y_s = s_tgt[target_idx].astype(np.float32)

        n_tr_s = int(0.8 * N_s)
        n_va_s = N_s - n_tr_s
        X_train_list.append(X_s[:n_tr_s])
        Y_train_list.append(Y_s[:n_tr_s])
        X_val_list.append(X_s[n_tr_s:])
        Y_val_list.append(Y_s[n_tr_s:])
        X_all_list.append(X_s)
        Y_all_list.append(Y_s)
        val_session_slices.append((s_idx + 1, val_offset, val_offset + n_va_s))
        val_offset += n_va_s

    return {
        "X_seq": np.concatenate(X_all_list, axis=0),
        "Y_deg": np.concatenate(Y_all_list, axis=0),
        "X_train_raw": np.concatenate(X_train_list, axis=0),
        "Y_train_deg": np.concatenate(Y_train_list, axis=0),
        "X_val_raw": np.concatenate(X_val_list, axis=0),
        "Y_val_deg": np.concatenate(Y_val_list, axis=0),
        "channel_std": latest_channel_std,
        "inv_sqrt_mean_cov": latest_inv_sqrt,
        "n_sessions": np.int64(valid_session_count),
        "val_session_slices": np.asarray(val_session_slices, dtype=np.int64),
    }


def train_intrasubject(
    data_path: Path,
    out_ckpt: Path,
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 5e-4,
) -> dict[str, float]:
    """Train intra-subject TRR model and export self-contained checkpoint."""
    torch.manual_seed(42)
    np.random.seed(42)

    with np.load(data_path) as archive:
        emg = archive["emg"].astype(np.float32)
        targets = archive["targets"].astype(np.float32)
        fs = int(archive["fs"]) if "fs" in archive else 200
        session_lengths = (
            archive["session_lengths"].astype(np.int64).tolist() if "session_lengths" in archive else [int(len(emg))]
        )

    total_min = len(emg) / float(max(fs, 1)) / 60.0
    print(
        f"Loaded dataset from {data_path}: {len(emg):,} samples "
        f"({total_min:.1f} min across {len(session_lengths)} session(s))"
    )

    sub_len = 100
    seq_len = 10
    ds = extract_intrasubject_dataset(
        emg,
        targets,
        session_lengths=session_lengths,
        fs=fs,
        sub_len=sub_len,
        seq_len=seq_len,
    )
    X_train_raw = ds["X_train_raw"]
    Y_train_deg = ds["Y_train_deg"]
    X_val_raw = ds["X_val_raw"]
    Y_val_deg = ds["Y_val_deg"]
    in_dim = int(X_train_raw.shape[-1])

    # Standardize using training split statistics across all sessions
    f_mean = np.mean(X_train_raw, axis=(0, 1), keepdims=True).astype(np.float32)
    f_std = np.maximum(np.std(X_train_raw, axis=(0, 1), keepdims=True), 0.05).astype(np.float32)
    t_mean = np.mean(Y_train_deg, axis=0).astype(np.float32)
    t_std = np.maximum(np.std(Y_train_deg, axis=0), 1.0).astype(np.float32)

    X_train_norm = np.clip((X_train_raw - f_mean) / f_std, -4.0, 4.0)
    Y_train_norm = (Y_train_deg - t_mean) / t_std
    X_val_norm = np.clip((X_val_raw - f_mean) / f_std, -4.0, 4.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(channels=8, targets=15, hidden=64)
    # Configure 124-feature Hybrid CMTS + Phasic/Tonic GRU input
    from src.model.gru import GRUEncoder

    model.gru = GRUEncoder(in_dim=in_dim, dense=256, units=(256, 128), dropout=0.1)
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(Y_train_norm)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val_norm).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.25)

    best_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model.head(model.gru(xb))
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_val_norm = model.head(model.gru(X_val_t)).cpu().numpy()
            pred_val_deg = apply_biomechanical_dip_coupling(pred_val_norm * t_std + t_mean)
            val_mae = float(np.mean(np.abs(Y_val_deg - pred_val_deg)))
            if val_mae < best_mae:
                best_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_pred_deg = apply_biomechanical_dip_coupling(model.head(model.gru(X_val_t)).cpu().numpy() * t_std + t_mean)

    val_slices = ds.get("val_session_slices", np.empty((0, 3), dtype=np.int64))
    for s_num, v0, v1 in val_slices:
        y_true_s = Y_val_deg[v0:v1]
        y_pred_s = best_pred_deg[v0:v1]
        s_mae = float(np.mean(np.abs(y_true_s - y_pred_s)))
        s_num_err = np.sum((y_true_s - y_pred_s) ** 2, axis=0)
        s_den_err = np.sum((y_true_s - np.mean(y_true_s, axis=0)) ** 2, axis=0)
        s_nmse = float(np.mean(s_num_err / s_den_err))
        s_rs = [
            float(np.corrcoef(y_true_s[:, j], y_pred_s[:, j])[0, 1])
            for j in range(15)
            if np.std(y_true_s[:, j]) > 1e-3 and np.std(y_pred_s[:, j]) > 1e-3
        ]
        s_r = float(np.nanmean(s_rs)) if s_rs else 0.0
        print(
            f"  Session {int(s_num)} held-out validation: MAE = {s_mae:.2f} deg | "
            f"NMSE = {s_nmse:.3f} | Pearson r = {s_r:.3f}"
        )

    overall_nmse = float(
        np.mean(
            np.sum((Y_val_deg - best_pred_deg) ** 2, axis=0)
            / np.sum((Y_val_deg - np.mean(Y_val_deg, axis=0)) ** 2, axis=0)
        )
    )
    ckpt = {
        "model_state_dict": best_state,
        "feature_mean": f_mean,
        "feature_std": f_std,
        "target_mean": t_mean,
        "target_std": t_std,
        "channel_std": ds["channel_std"],
        "inv_sqrt_mean_cov": ds["inv_sqrt_mean_cov"],
        "intrasubject_calibrated": True,
        "intrasubject_bands": INTRASUBJECT_BANDS,
        "intrasubject_sub_len": sub_len,
        "intrasubject_seq_len": seq_len,
        "biomechanical_dip_coupling": True,
        "val_mae": best_mae,
        "val_nmse": overall_nmse,
    }
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_ckpt)
    print(
        f"Intra-subject TRR training complete! Held-out MAE = {best_mae:.2f} deg | "
        f"NMSE = {overall_nmse:.3f} -> Saved to {out_ckpt}"
    )
    return {"val_mae": best_mae, "val_nmse": overall_nmse}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intra-subject TRR model from Myo recording.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "recordings" / "myo_intrasubject.npz",
        help="Input .npz recording path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "models" / "best_model.pt",
        help="Output model checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=45)
    args = parser.parse_args()
    train_intrasubject(args.data, args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
