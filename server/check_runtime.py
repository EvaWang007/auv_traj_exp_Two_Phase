"""Validate either the current rented image or the previously used GPU environment."""
import json
import sys

SUPPORTED = {((3, 10), '2.7.1', '11.8'), ((3, 12), '2.8.0', '12.8')}


def validate_runtime(python_version, torch_version, cuda_runtime, cuda_available):
    key = (tuple(python_version[:2]), torch_version.split('+')[0], cuda_runtime)
    if key not in SUPPORTED:
        raise RuntimeError(f'Unsupported runtime {key}. Expected Python 3.12 / torch 2.8.0 / CUDA 12.8, '
                           'or the legacy Python 3.10 / torch 2.7.1 / CUDA 11.8 environment.')
    if not cuda_available:
        raise RuntimeError('CUDA is unavailable. Check NVIDIA driver and GPU visibility; formal training will not fall back to CPU.')


def main():
    import torch
    validate_runtime(sys.version_info, torch.__version__, torch.version.cuda, torch.cuda.is_available())
    x = torch.randn(32, 32, device='cuda')
    result = (x @ x).mean().item()
    print(json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, cuda_runtime=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(0), cuda_test_result=result), indent=2), flush=True)


if __name__ == '__main__': main()
