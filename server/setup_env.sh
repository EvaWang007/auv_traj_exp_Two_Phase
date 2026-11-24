#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${AUV_ENV_NAME:-auv_traj_gpu}"
command -v conda >/dev/null || { echo 'conda not found. Source your Miniconda conda.sh first.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi not found. Use an NVIDIA GPU server image with its driver installed.' >&2; exit 1; }
nvidia-smi
if ! conda run -n "$ENV_NAME" python -c 'import sys; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then
    if conda env list | awk -v name="$ENV_NAME" '$1 == name {found=1} END {exit !found}'; then
        echo "Existing environment $ENV_NAME is not Python 3.10. Choose a new AUV_ENV_NAME; it will not be overwritten." >&2
        exit 1
    fi
    conda create -n "$ENV_NAME" python=3.10 pip -y
fi
if conda run -n "$ENV_NAME" python -c 'import torch' >/dev/null 2>&1; then
    conda run -n "$ENV_NAME" python -c 'import torch; assert torch.__version__.split("+")[0] == "2.7.1" and torch.version.cuda == "11.8", "Existing torch differs from 2.7.1/cu118. Choose a fresh AUV_ENV_NAME rather than modifying the old environment."'
else
    conda run -n "$ENV_NAME" python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
fi
conda run -n "$ENV_NAME" python -m pip install -r "$ROOT_DIR/server/requirements-gpu.txt"
conda run -n "$ENV_NAME" python -m pip check
conda run -n "$ENV_NAME" python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: inspect driver/GPU visibility"; print("torch",torch.__version__,"CUDA",torch.version.cuda,"GPU",torch.cuda.get_device_name(0)); x=torch.randn(32,32,device="cuda"); print("CUDA operation",(x@x).mean().item())'
echo
echo "Environment ready. In your CURRENT shell run: conda activate $ENV_NAME"
echo "Then: bash server/run_phase1.sh"
