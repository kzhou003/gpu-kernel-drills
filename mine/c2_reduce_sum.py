"""C2 · Reduction sum — Python wrapper. Edit c2_reduce_sum.cu, not this file.

The CUDA source next to this file is compiled on first use with torch's load_inline
(needs a CUDA GPU + nvcc; the first build takes ~30-60 s, later runs hit the cache).
"""
from pathlib import Path

from common.cuda_ext import load

_DECLS = """
torch::Tensor reduce_sum(torch::Tensor x);
"""


def _ext():
    return load(str(Path(__file__).with_suffix(".cu")), _DECLS, ("reduce_sum",))


def reduce_sum(x):
    return _ext().reduce_sum(x)
