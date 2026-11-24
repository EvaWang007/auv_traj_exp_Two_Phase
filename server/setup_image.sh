#!/usr/bin/env bash
# Reuse the rented image's torch/CUDA without downloading another CUDA stack.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AUV_PYTHON:-python}"
VENV_DIR="${AUV_VENV_DIR:-$ROOT_DIR/.venv-image}"
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver tools not found.' >&2; exit 1; }
nvidia-smi
"$PYTHON_BIN" "$ROOT_DIR/server/check_runtime.py"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi
# Install project-specific versions in this venv, leaving image-wide packages alone.
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/server/requirements-gpu.txt"
"$VENV_DIR/bin/python" "$ROOT_DIR/server/check_runtime.py"
"$VENV_DIR/bin/python" -c 'import numpy,pandas,matplotlib; print("numpy",numpy.__version__,"pandas",pandas.__version__,"matplotlib",matplotlib.__version__)'
printf '\nReady. Activate in your current terminal:\nsource %q/bin/activate\n' "$VENV_DIR"
echo 'Then run: bash server/run_phase1.sh'
