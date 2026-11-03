"""Shared marker: CUDA C++ drills need a real GPU + nvcc (Colab / cloud GPU)."""
import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA drills need an NVIDIA GPU")
