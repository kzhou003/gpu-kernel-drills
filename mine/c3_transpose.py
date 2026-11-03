"""C3 · Matrix transpose — Python wrapper. Edit c3_transpose.cu, not this file.

The CUDA source next to this file is compiled on first use with torch's load_inline
(needs a CUDA GPU + nvcc; the first build takes ~30-60 s, later runs hit the cache).
"""
from pathlib import Path

from common.cuda_ext import load

_DECLS = """
torch::Tensor transpose_naive(torch::Tensor x);
torch::Tensor transpose_tiled(torch::Tensor x);
"""


def _ext():
    return load(str(Path(__file__).with_suffix(".cu")), _DECLS, ("transpose_naive", "transpose_tiled",))


def transpose_naive(x):
    return _ext().transpose_naive(x)


def transpose_tiled(x):
    return _ext().transpose_tiled(x)
