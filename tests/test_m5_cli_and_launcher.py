"""Unit and integration tests for CLI runner, bash launcher scripts, and integrated MuJoCo simulator."""

from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Any

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_WEIGHTS_PATH,
    build_parser,
    format_latency_report,
    load_config,
    parse_args,
    run_pipeline,
)


class TestCliArgumentParsing:
    """Tests for scripts/run_pipeline.py CLI parser and defaults."""

    def test_default_arguments(self) -> None:
        args = parse_args([])
        assert args.mode == "auto"
        assert args.data is None
        assert args.weights == DEFAULT_WEIGHTS_PATH
        assert args.config == DEFAULT_CONFIG_PATH
        assert args.host == "127.0.0.1"
        assert args.port == 5005
        assert args.rate == 10.0
        assert args.duration == 0.0
        assert args.log_latency is False
        assert args.verbose is False

    def test_custom_overrides(self, tmp_path: pathlib.Path) -> None:
        custom_weights = tmp_path / "custom_weights.pt"
        custom_config = tmp_path / "custom_config.yaml"
        custom_data = tmp_path / "custom_data.npz"

        argv = [
            "--mode",
            "mock",
            "--data",
            str(custom_data),
            "--weights",
            str(custom_weights),
            "--config",
            str(custom_config),
            "--host",
            "192.168.1.50",
            "--port",
            "6000",
            "--rate",
            "20.5",
            "--duration",
            "5.2",
            "--log-latency",
            "--verbose",
        ]
        args = parse_args(argv)
        assert args.mode == "mock"
        assert args.data == custom_data
        assert args.weights == custom_weights
        assert args.config == custom_config
        assert args.host == "192.168.1.50"
        assert args.port == 6000
        assert args.rate == 20.5
        assert args.duration == 5.2
        assert args.log_latency is True
        assert args.verbose is True

    def test_invalid_mode_choice_raises_error(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "invalid_mode"])

    def test_invalid_port_boundary(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--port", "0"])
        with pytest.raises(SystemExit):
            parse_args(["--port", "70000"])

    def test_invalid_rate_boundary(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--rate", "0"])
        with pytest.raises(SystemExit):
            parse_args(["--rate", "-5.0"])

    def test_invalid_duration_boundary(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--duration", "-1.0"])


class TestConfigAndReportHelpers:
    """Tests for configuration loading and report formatting utilities."""

    def test_load_config_existing_file(self) -> None:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        assert isinstance(cfg, dict)
        assert "preprocessing" in cfg
        assert "windowing" in cfg
        assert "socket" in cfg
        assert cfg["preprocessing"]["fs"] == 200

    def test_load_config_nonexistent_fallback(self, tmp_path: pathlib.Path) -> None:
        missing_file = tmp_path / "nonexistent_params.yaml"
        cfg = load_config(missing_file)
        assert isinstance(cfg, dict)
        assert "preprocessing" in cfg
        assert cfg["preprocessing"]["fs"] == 200
        assert cfg["socket"]["port"] == 5005

    def test_format_latency_report_structure(self) -> None:
        summary: dict[str, Any] = {
            "compute": {
                "mean": 5.25,
                "std": 0.45,
                "min": 4.10,
                "max": 7.80,
                "p50": 5.15,
                "p95": 6.10,
                "p99": 7.20,
                "jitter": 0.35,
            },
            "transport": {
                "mean": 11.50,
                "std": 1.20,
                "min": 9.20,
                "max": 15.60,
                "p50": 11.20,
                "p95": 13.80,
                "p99": 15.10,
                "jitter": 0.85,
            },
        }
        report = format_latency_report(
            summary=summary,
            mode="mock (real)",
            steps_completed=50,
            samples_processed=1000,
            elapsed_sec=5.0,
            target_rate_hz=10.0,
        )
        assert "INFERENCE PIPELINE EXECUTION REPORT" in report
        assert "MOCK (REAL)" in report
        assert "Steps Executed:    50" in report
        assert "Mean:     5.25 ms" in report
        assert "Mean:    11.50 ms" in report
        assert "Jitter:   0.35 ms" in report


class TestMockPipelineExecution:
    """Integration tests for live short-run execution in mock mode."""

    def test_mock_pipeline_short_run_direct_call(self) -> None:
        args = parse_args(["--mode", "mock", "--duration", "0.4", "--rate", "10.0"])
        exit_code = run_pipeline(args)
        assert exit_code == 0

    def test_mock_pipeline_subprocess_execution(self) -> None:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
            "--mode",
            "mock",
            "--duration",
            "0.4",
            "--rate",
            "10.0",
            "--log-latency",
        ]
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "Initializing BRI Real-Time Inference Pipeline" in result.stderr
        assert "INFERENCE PIPELINE EXECUTION REPORT" in result.stdout
        assert "MOCK" in result.stdout


class TestBashLaunchersAndSimulator:
    """Verification of bash launcher scripts and integrated MuJoCo Shadow Hand simulator."""

    def test_bash_scripts_exist_and_no_ps1_or_bat(self) -> None:
        assert not list(PROJECT_ROOT.glob("*.ps1")), "No .ps1 scripts should exist"
        assert not list(PROJECT_ROOT.glob("*.bat")), "No .bat scripts should exist"
        for sh_name in ("run.sh", "view_emg.sh", "scan_myo.sh", "record_and_train.sh"):
            sh_path = PROJECT_ROOT / sh_name
            assert sh_path.is_file(), f"Missing {sh_name}"

    def test_run_sh_structure(self) -> None:
        content = (PROJECT_ROOT / "run.sh").read_text(encoding="utf-8")
        assert "src.sim" in content
        assert "run_pipeline.py" in content
        assert "8080" in content
        assert "mujoco-rampup" not in content

    def test_integrated_shadow_hand_simulator_loads(self) -> None:
        from src.sim.config import HAND, MODEL, UDP_PORT
        from src.sim.sim import decode_controls, load_shadow_hand

        model, data = load_shadow_hand()
        assert model.nu == 20
        assert data.ctrl.shape == (20,)
        assert UDP_PORT == 5005
        assert MODEL == "shadow_hand"
        assert HAND == "right_hand"
        decoded = decode_controls([0.5] * 20)
        assert decoded.shape == (20,)

    def test_readme_exists_and_documents_bash_workflow(self) -> None:
        readme_path = PROJECT_ROOT / "README.md"
        assert readme_path.is_file()
        content = readme_path.read_text(encoding="utf-8")
        assert "./run.sh" in content
        assert "src.sim" in content
        assert "pytest" in content
