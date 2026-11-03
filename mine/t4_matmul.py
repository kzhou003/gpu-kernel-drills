"""T4 · Tiled GEMM with tl.dot, grouped ordering and autotuning — YOUR IMPLEMENTATION.

C[M, N] = A[M, K] @ B[K, N]; fp16/bf16 in, fp32 accumulate, C in the input dtype.
The wrapper and autotune configs are done; implement matmul_kernel.
"""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    # TODO:
    # 1. pid -> (pid_m, pid_n) with grouped ordering (GROUP_M rows of tiles per group)
    # 2. offs_m, offs_n, offs_k; a_ptrs [BLOCK_M, BLOCK_K], b_ptrs [BLOCK_K, BLOCK_N]
    # 3. acc = zeros fp32; loop over K tiles: masked loads, acc = tl.dot(a, b, acc), advance pointers
    # 4. cast and masked store of the C tile
    pass


# Candidate configs for the autotuner. num_stages = software-pipelining depth (how many K tiles
# are prefetched into shared memory ahead of use); num_warps = threads per program / 32.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
]

# autotune wraps the same jitted kernel: the first call for each new (M, N, K) benchmarks all
# configs and caches the fastest one; later calls with that key reuse it.
matmul_kernel_autotuned = triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])(matmul_kernel)


def matmul(a: torch.Tensor, b: torch.Tensor, config: dict | None = None) -> torch.Tensor:
    """config=None -> autotuned on GPU; pass e.g. {"BLOCK_M": 64, ...} to force one config."""
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], "shape mismatch"
    assert a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16)
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    if M == 0 or N == 0:
        return c
    args = (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))

    # The CPU interpreter cannot time configs, so it always uses one fixed config.
    if config is None and os.environ.get("TRITON_INTERPRET") == "1":
        config = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}
    if config is not None:
        grid = (triton.cdiv(M, config["BLOCK_M"]) * triton.cdiv(N, config["BLOCK_N"]),)
        matmul_kernel[grid](*args, **config)
    else:
        # A 1-D grid sized from whichever config the autotuner is trying.
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        matmul_kernel_autotuned[grid](*args)
    return c
