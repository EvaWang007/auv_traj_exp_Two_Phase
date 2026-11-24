#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${AUV_PYTHON:-python}"
OUTPUT_DIR="${1:-results/formal_phase1_30ep_seed42}"
CONFIG_FILE="${AUV_CONFIG:-server/phase1_30ep.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
mkdir -p "$OUTPUT_DIR/logs"
exec 9>"$OUTPUT_DIR/.launch.lock"
flock -n 9 || { echo 'Another launcher is already using this output directory.' >&2; exit 1; }
"$PYTHON_BIN" server/check_runtime.py
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
nvidia-smi > "$OUTPUT_DIR/logs/nvidia-smi-$STAMP.txt"
"$PYTHON_BIN" -m pip freeze > "$OUTPUT_DIR/logs/pip-freeze-$STAMP.txt"
"$PYTHON_BIN" -u formal_phase1.py --mode preflight --config "$CONFIG_FILE" --output "$OUTPUT_DIR" --device cuda 2>&1 | tee -a "$OUTPUT_DIR/logs/preflight.log"
"$PYTHON_BIN" -u formal_phase1.py --mode train --config "$CONFIG_FILE" --output "$OUTPUT_DIR" --device cuda --resume 2>&1 | tee -a "$OUTPUT_DIR/logs/train.log"
"$PYTHON_BIN" -u formal_phase1.py --mode evaluate --config "$CONFIG_FILE" --output "$OUTPUT_DIR" --device cuda 2>&1 | tee -a "$OUTPUT_DIR/logs/evaluate.log"
echo "Completed. Results: $ROOT_DIR/$OUTPUT_DIR/test"
