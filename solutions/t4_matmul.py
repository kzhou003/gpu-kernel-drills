"""T4 · Tiled GEMM with tl.dot, grouped ordering and autotuning — reference solution.

C[M, N] = A[M, K] @ B[K, N]; A, B fp16 or bf16, fp32 accumulation, C in the input dtype.
Each program computes one BLOCK_M x BLOCK_N tile of C, walking K in BLOCK_K steps.
tl.dot lowers to Tensor Core instructions (mma / wgmma) on NVIDIA GPUs.
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
    # ---- 1. which C tile does this program own? (grouped / "super-grouping" order)
    # Plain row-major order (pid_m = pid // num_pid_n) makes consecutive programs sweep a
    # whole row of tiles: they share one A row-panel but each needs a different B column-panel.
    # Grouping GROUP_M rows of tiles together and going down the columns first means programs
    # running at the same time reuse both A and B panels while they are still in L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)   # the last group may be shorter
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 2. pointers to the first A and B tiles (2-D blocks of pointers via broadcasting)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn   # [BLOCK_K, BLOCK_N]

    # ---- 3. main loop over K: accumulate in fp32 registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        # Masked loads return 0 outside the matrix; zeros add nothing to the dot product.
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)          # acc += a @ b on Tensor Cores
        a_ptrs += BLOCK_K * stride_ak    # slide both tiles along K
        b_ptrs += BLOCK_K * stride_bk

    # ---- 4. epilogue: cast and store the C tile
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


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
