#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${AUV_PYTHON:-python}"
OUTPUT_DIR="${1:-results/Pi_LSTM_30ep_seed42}"
DATA_DIR="${AUV_DATA:-data}"
REFERENCE_RUN="${AUV_REFERENCE_RUN:-results/formal_phase1_30ep_seed42}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
mkdir -p "$OUTPUT_DIR/logs"
exec 9>"$OUTPUT_DIR/.launch.lock"
flock -n 9 || { echo 'Pi_LSTM launcher already active.' >&2; exit 1; }
"$PYTHON_BIN" -m pip freeze > "$OUTPUT_DIR/logs/environment_pip_freeze.txt"
"$PYTHON_BIN" -u Pi_LSTM_train.py --data "$DATA_DIR" --reference-run "$REFERENCE_RUN" \
  --output "$OUTPUT_DIR" --config "${AUV_PI_CONFIG:-server/Pi_LSTM_30ep.json}" --device cuda --resume 2>&1 | tee -a "$OUTPUT_DIR/logs/train.log"
echo "Pi_LSTM training finished. Run Pi_LSTM_compare.py separately for frozen comparison."
