"""pytest hook: without a CUDA GPU, run Triton kernels in the CPU interpreter.

TRITON_INTERPRET must be set before `triton` is imported anywhere, so it lives here
(pytest imports conftest.py before any test module).
"""
import os

try:
    import torch
except ImportError:  # the MLX venv (X track) has no torch
    torch = None

if torch is not None and not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
