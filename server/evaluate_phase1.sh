#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
"${AUV_PYTHON:-python}" -u formal_phase1.py --mode evaluate --config "${AUV_CONFIG:-server/phase1_30ep.json}" --output "${1:-results/formal_phase1_30ep_seed42}" --device cuda
