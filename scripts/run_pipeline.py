"""Command-Line Interface (CLI) runner for BRI Real-Time Inference Pipeline.

Bridges 8-channel 200 Hz sEMG acquisition (BLE Myo armband or high-precision Mock replay),
model-agnostic streaming inference at 10 Hz cadence, closed-form kinematic mapping,
and binary UDP transmission to the MuJoCo Shadow Hand simulation viewer.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import signal
import sys
import threading
import time
from typing import Any, Sequence

import yaml

# Ensure project root is present on sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kinematics.udp_sender import UDPSender  # noqa: E402
from src.model.model import Model  # noqa: E402
from src.preprocess import Preprocess  # noqa: E402
from src.stream.ble_myo import BleMyoClient  # noqa: E402
from src.stream.mock_myo import MockMyoStreamer  # noqa: E402
from src.stream.ring_buffer import EMGRingBuffer  # noqa: E402
from src.stream.worker import StreamingInferenceWorker  # noqa: E402

logger = logging.getLogger("run_pipeline")

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "params.yaml"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "models" / "best_model.pt"
DEFAULT_RECORDING_PATH = PROJECT_ROOT / "data" / "recordings" / "myo_intrasubject.npz"
DEFAULT_DATA_PATH = (
    DEFAULT_RECORDING_PATH
    if DEFAULT_RECORDING_PATH.is_file()
    else PROJECT_ROOT.parent / "BRI-TrainingPipeline" / "data" / "raw" / "emg_fk" / "subject_01_trial_01.npz"
)


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for run_pipeline."""
    parser = argparse.ArgumentParser(
        description="BRI Real-Time Continuous sEMG Inference Pipeline & MuJoCo Shadow Hand Bridge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "ble", "auto"],
        default="auto",
        help=(
            "Acquisition mode: 'ble' / 'auto' (continuously searches for physical Myo armband without mock fallback), "
            "or 'mock' (synthetic/replay 200 Hz streamer)."
        ),
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="Path to recorded .npz dataset for mock replay (falls back to default or synthetic).",
    )
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to pretrained TRR model weights checkpoint (.pt file).",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to centralized YAML configuration file.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Target IP address for MuJoCo simulation UDP receiver.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="Target UDP port for MuJoCo simulation controls.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Streaming inference cadence in Hz (frequency of model evaluations).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Execution duration in seconds. Set to 0 to run continuously until interrupted.",
    )
    parser.add_argument(
        "--log-latency",
        action="store_true",
        default=False,
        help=("Print real-time per-step compute and transport latency measurements during streaming."),
    )
    parser.add_argument(
        "--mac",
        type=str,
        default=None,
        help="Optional Bluetooth MAC address of target Myo armband (e.g. 'DC:9C:FE:88:60:AC').",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose debug logging output.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.port <= 0 or args.port > 65535:
        parser.error(f"--port must be between 1 and 65535, got {args.port}")
    if args.rate <= 0:
        parser.error(f"--rate must be positive, got {args.rate}")
    if args.duration < 0:
        parser.error(f"--duration must be non-negative, got {args.duration}")

    return args


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    """Load configuration from YAML file or return robust fallback defaults."""
    resolved_path = config_path.resolve()
    if resolved_path.is_file():
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                logger.info("Loaded configuration from %s", resolved_path)
                return cfg
        except Exception as exc:
            logger.warning(
                "Failed to parse config file %s: %s. Using default settings.",
                resolved_path,
                exc,
            )

    logger.info(
        "Config file not found at %s. Using built-in configuration defaults.",
        resolved_path,
    )
    return {
        "preprocessing": {"fs": 200, "bandpass": [15, 95], "notch": 50, "channels": 8},
        "windowing": {"sub_window": 60, "step": 20, "queue_len": 50},
        "socket": {"host": "127.0.0.1", "port": 5005},
        "model": {"channels": 8, "targets": 15, "hidden": 64},
    }


def format_latency_report(
    summary: dict[str, Any],
    mode: str,
    steps_completed: int,
    samples_processed: int,
    elapsed_sec: float,
    target_rate_hz: float,
) -> str:
    """Format a detailed latency and throughput summary table."""
    comp = summary.get("compute", {})
    trans = summary.get("transport", {})

    actual_rate = (steps_completed / elapsed_sec) if elapsed_sec > 0 else 0.0

    c_mean = comp.get("mean", 0.0)
    c_std = comp.get("std", 0.0)
    c_jit = comp.get("jitter", 0.0)
    c_min = comp.get("min", 0.0)
    c_max = comp.get("max", 0.0)
    c_p50 = comp.get("p50", 0.0)
    c_p95 = comp.get("p95", 0.0)
    c_p99 = comp.get("p99", 0.0)

    t_mean = trans.get("mean", 0.0)
    t_std = trans.get("std", 0.0)
    t_jit = trans.get("jitter", 0.0)
    t_min = trans.get("min", 0.0)
    t_max = trans.get("max", 0.0)
    t_p50 = trans.get("p50", 0.0)
    t_p95 = trans.get("p95", 0.0)
    t_p99 = trans.get("p99", 0.0)

    lines = [
        "=" * 78,
        "                    INFERENCE PIPELINE EXECUTION REPORT",
        "=" * 78,
        f"Mode:              {mode.upper()}",
        f"Elapsed Time:      {elapsed_sec:.2f} s",
        (f"Steps Executed:    {steps_completed} (Target: {target_rate_hz:.1f} Hz | Actual: {actual_rate:.1f} Hz)"),
        f"Samples Processed: {samples_processed} (@ 200 Hz)",
        "-" * 78,
        "Forward Compute Latency (Model Inference):",
        f"  Mean:   {c_mean:6.2f} ms   | Std:    {c_std:6.2f} ms   | Jitter: {c_jit:6.2f} ms",
        f"  Min:    {c_min:6.2f} ms   | Max:    {c_max:6.2f} ms",
        f"  p50:    {c_p50:6.2f} ms   | p95:    {c_p95:6.2f} ms   | p99:    {c_p99:6.2f} ms",
        "-" * 78,
        "End-to-End Transport Latency (Acquisition -> UDP Send):",
        f"  Mean:   {t_mean:6.2f} ms   | Std:    {t_std:6.2f} ms   | Jitter: {t_jit:6.2f} ms",
        f"  Min:    {t_min:6.2f} ms   | Max:    {t_max:6.2f} ms",
        f"  p50:    {t_p50:6.2f} ms   | p95:    {t_p95:6.2f} ms   | p99:    {t_p99:6.2f} ms",
        "=" * 78,
    ]
    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> int:
    """Initialize and run the streaming inference pipeline.

    Args:
        args: Parsed CLI arguments namespace.

    Returns:
        int: Process exit code (0 on clean exit).
    """
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Initializing BRI Real-Time Inference Pipeline...")
    logger.info(
        "Arguments: mode=%s, rate=%.1f Hz, target=%s:%d, duration=%.1f s",
        args.mode,
        args.rate,
        args.host,
        args.port,
        args.duration,
    )

    # 1. Load configuration
    config = load_config(args.config)
    prep_cfg = config.get("preprocessing", {})
    win_cfg = config.get("windowing", {})
    model_cfg = config.get("model", {})

    fs = int(prep_cfg.get("fs", 200))
    channels = int(prep_cfg.get("channels", 8))
    bandpass = prep_cfg.get("bandpass", [15, 95])
    notch = float(prep_cfg.get("notch", 50.0))
    sub_window = int(win_cfg.get("sub_window", 60))
    step = int(win_cfg.get("step", 20))
    queue_len = int(win_cfg.get("queue_len", 50))
    targets = int(model_cfg.get("targets", 15))
    hidden = int(model_cfg.get("hidden", 64))
    length = int(model_cfg.get("length", queue_len))
    window_size = int(win_cfg.get("total_window", sub_window + (length - 1) * step))

    # 2. Initialize Model and load weights
    model = Model(
        channels=channels,
        targets=targets,
        window_size=window_size,
        length=length,
        activation=str(model_cfg.get("activation", "relu")),
        hidden=hidden,
        downsample=int(model_cfg.get("downsample", 1)),
        ssm_speed=float(model_cfg.get("ssm_speed", 4.0)),
    )

    weights_path = pathlib.Path(args.weights).resolve()
    if weights_path.is_file():
        logger.info("Loading pretrained weights from %s", weights_path)
        try:
            model.load_pretrained_weights(weights_path, map_location="cpu")
            logger.info("Pretrained model weights loaded successfully.")
        except Exception as exc:
            logger.error("Failed loading model weights from %s: %s", weights_path, exc)
            raise
    else:
        logger.warning("Weights file not found at %s. Running with initialized weights.", weights_path)

    model.eval()

    # 3. Initialize Preprocessor
    preprocessor = Preprocess(
        fs=fs,
        bandpass=bandpass,
        notch=notch,
        channels=channels,
    )

    # 4. Initialize Ring Buffer & UDP Sender
    ring_buffer = EMGRingBuffer(capacity=max(2000, model.window_size * 2), channels=channels)
    udp_sender = UDPSender(host=args.host, port=args.port)

    # 5. Initialize Streaming Worker
    worker = StreamingInferenceWorker(
        ring_buffer=ring_buffer,
        model=model,
        preprocessor=preprocessor,
        udp_sender=udp_sender,
        params=config,
        window_size=model.window_size,
        sub_window=sub_window,
        step=step,
        queue_len=model.length,
        fs=fs,
        cadence_hz=args.rate,
        pace_loop=True,
    )

    # 6. Initialize Acquisition Source based on mode
    data_source_path = args.data
    if data_source_path is None and DEFAULT_DATA_PATH.is_file():
        data_source_path = DEFAULT_DATA_PATH

    streamer: MockMyoStreamer | None = None
    ble_client: BleMyoClient | None = None
    active_mode_str = args.mode

    target_mac = args.mac or config.get("ble", {}).get("device_address")

    if args.mode == "mock":
        logger.info("Starting acquisition in MOCK mode (data_source=%s)", data_source_path)
        streamer = MockMyoStreamer(
            destination=ring_buffer,
            data_source=data_source_path,
            sample_rate=float(fs),
            channels=channels,
        )
    elif args.mode in ("ble", "auto"):
        logger.info(
            "Starting acquisition in BLE mode (continuous scanning for physical Myo armband, target=%s)...",
            target_mac or "scan",
        )
        ble_client = BleMyoClient(
            destination=ring_buffer,
            device_address=target_mac,
            mock_fallback=False,
            force_mock=False,
            data_path=data_source_path,
        )
    else:
        raise ValueError(f"Unsupported acquisition mode: {args.mode}")

    # 7. Setup graceful shutdown handler
    shutdown_event = threading.Event()

    def _signal_handler(sig: int, _frame: Any) -> None:
        signame = signal.Signals(sig).name if hasattr(signal, "Signals") else str(sig)
        logger.info("Received shutdown signal %s (%d). Cleaning up...", signame, sig)
        shutdown_event.set()

    # Register signals
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    try:
        # Start acquisition
        if streamer is not None:
            streamer.start()
            active_mode_str = f"mock ({streamer.data_source_type})"
        elif ble_client is not None:
            ble_client.start(wait_ready=False)
            active_mode_str = "ble (physical armband)"

        # Start inference worker
        worker.start()
        logger.info("Inference worker started at %.1f Hz. Pipeline active.", args.rate)

        t_start = time.perf_counter()
        last_log_step = 0

        # Loop until shutdown requested or duration reached
        while not shutdown_event.is_set():
            now = time.perf_counter()
            elapsed = now - t_start

            if args.duration > 0 and elapsed >= args.duration:
                logger.info("Specified duration (%.2f s) reached. Stopping...", args.duration)
                break

            # Handle per-step latency logging
            if args.log_latency:
                current_step = worker.steps_completed
                if current_step > last_log_step:
                    latest_res = worker.latest_step_result
                    if latest_res is not None:
                        logger.info(
                            "[LAT] Step %5d | Comp: %5.2f ms | Trans: %5.2f ms | Mean: %5.2f ms",
                            current_step,
                            latest_res.compute_latency_ms,
                            latest_res.transport_latency_ms,
                            worker.mean_compute_latency_ms,
                        )
                    last_log_step = current_step

            if ble_client is not None:
                if getattr(ble_client, "is_shared_follower", False):
                    if worker.udp_sender is not None:
                        worker.udp_sender = None
                elif worker.udp_sender is None:
                    worker.udp_sender = udp_sender

            # Check if background acquisition or worker died unexpectedly
            if streamer is not None and not streamer.is_alive() and not shutdown_event.is_set():
                logger.warning("Mock streamer stopped unexpectedly.")
                break
            if ble_client is not None and not ble_client.is_alive() and not shutdown_event.is_set():
                logger.warning("BLE client stopped unexpectedly.")
                break

            # Responsive wait
            shutdown_event.wait(timeout=0.05)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        total_elapsed = time.perf_counter() - t_start
        logger.info("Shutting down pipeline components...")

        # Stop worker
        worker.stop(timeout=1.0)

        # Stop acquisition
        if streamer is not None:
            streamer.stop(timeout=1.0)
        if ble_client is not None:
            ble_client.stop(timeout=1.0)

        # Close UDP transmitter
        udp_sender.close()

        # Display latency summary report
        summary = worker.get_latency_stats()
        report = format_latency_report(
            summary=summary,
            mode=active_mode_str,
            steps_completed=worker.steps_completed,
            samples_processed=worker.samples_processed,
            elapsed_sec=total_elapsed,
            target_rate_hz=args.rate,
        )
        print("\n" + report + "\n", flush=True)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
