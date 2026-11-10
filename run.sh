#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    echo "Error: Virtual environment not found at .venv" >&2
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    "$PYTHON" scripts/run_pipeline.py --help
    exit 0
fi

echo "Starting MuJoCo Shadow Hand simulator on http://localhost:8080..."
"$PYTHON" -m src.sim &
SIM_PID=$!
trap 'kill "$SIM_PID" 2>/dev/null || true' EXIT

sleep 2
if command -v start >/dev/null 2>&1; then
    start http://localhost:8080
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://localhost:8080 >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
    open http://localhost:8080 || true
fi

if [ $# -eq 0 ]; then
    "$PYTHON" scripts/run_pipeline.py --mode ble --log-latency
else
    "$PYTHON" scripts/run_pipeline.py "$@"
fi
