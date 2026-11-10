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

"$PYTHON" scripts/record_intrasubject_myo.py --append --duration 600 "$@"
"$PYTHON" scripts/train_intrasubject_myo.py --data data/recordings/myo_intrasubject.npz --out models/best_model.pt
