"""C4 · GEMM: naive -> shared memory -> register tiling — Python wrapper. Edit c4_gemm.cu, not this file.

The CUDA source next to this file is compiled on first use with torch's load_inline
(needs a CUDA GPU + nvcc; the first build takes ~30-60 s, later runs hit the cache).
"""
from pathlib import Path

from common.cuda_ext import load

_DECLS = """
torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_smem(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_regtile(torch::Tensor A, torch::Tensor B);
"""


def _ext():
    return load(str(Path(__file__).with_suffix(".cu")), _DECLS, ("matmul_naive", "matmul_smem", "matmul_regtile",))


def matmul_naive(A, B):
    return _ext().matmul_naive(A, B)


def matmul_smem(A, B):
    return _ext().matmul_smem(A, B)


def matmul_regtile(A, B):
    return _ext().matmul_regtile(A, B)
