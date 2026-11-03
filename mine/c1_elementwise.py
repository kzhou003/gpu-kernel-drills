"""C1 · Vector add + fused bias-GELU — Python wrapper. Edit c1_elementwise.cu, not this file.

The CUDA source next to this file is compiled on first use with torch's load_inline
(needs a CUDA GPU + nvcc; the first build takes ~30-60 s, later runs hit the cache).
"""
from pathlib import Path

from common.cuda_ext import load

_DECLS = """
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor bias);
"""


def _ext():
    return load(str(Path(__file__).with_suffix(".cu")), _DECLS, ("vector_add", "bias_gelu",))


def vector_add(a, b):
    return _ext().vector_add(a, b)


def bias_gelu(x, bias):
    return _ext().bias_gelu(x, bias)
