"""M3 · FP8 fine-grained (block-scaled) quantisation + GEMM — reference solution.

The DeepSeek-V3 / DeepGEMM recipe, written in Triton:
  activations: per-token-group quantisation, one fp32 scale per 1 x 128 group
  weights:     per-block quantisation,       one fp32 scale per 128 x 128 block
  C[m, n] = sum_kb  (A_q[m, kb] @ B_q[n, kb]^T) * sa[m, kb] * sb[n // 128, kb]
Each 128-wide K step is one fp8 tl.dot (Tensor Cores, fp32 accumulation) followed by a scale
multiply, so the scales change every K block without leaving fp32.
FP8 e4m3: largest finite value 448. Needs sm_89+ (Ada, Hopper, Blackwell incl. RTX 50xx).
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
GROUP = 128


@triton.jit
def per_token_group_quant_kernel(x_ptr, q_ptr, s_ptr, K, eps, GROUP_SIZE: tl.constexpr):
    # One program per (row, group): the scale makes the group's max |x| map to 448.
    row = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    offs = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = offs < K
    x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), eps)         # eps: an all-zero group gets a tiny scale, not 0
    scale = amax / 448.0
    q = tl.clamp(x / scale, -448.0, 448.0)
    tl.store(q_ptr + row * K + offs, q.to(q_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + row * tl.cdiv(K, GROUP_SIZE) + g, scale)


def per_token_group_quant_fp8(x: torch.Tensor, group_size: int = GROUP, eps: float = 1e-10):
    """x [M, K] -> (x_q [M, K] fp8 e4m3, scales [M, ceil(K/group)] fp32)."""
    x = x.contiguous()
    M, K = x.shape
    q = torch.empty((M, K), dtype=FP8, device=x.device)
    s = torch.empty((M, triton.cdiv(K, group_size)), dtype=torch.float32, device=x.device)
    per_token_group_quant_kernel[(M, triton.cdiv(K, group_size))](x, q, s, K, eps, GROUP_SIZE=group_size)
    return q, s


def block_quant_fp8(w: torch.Tensor, block: int = GROUP):
    """Weights [N, K] -> (w_q fp8 [N, K], scales [ceil(N/128), ceil(K/128)]). Offline, so plain PyTorch."""
    N, K = w.shape
    nb, kb = triton.cdiv(N, block), triton.cdiv(K, block)
    wp = torch.zeros(nb * block, kb * block, dtype=torch.float32, device=w.device)
    wp[:N, :K] = w.float()
    blocks = wp.view(nb, block, kb, block)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-10)              # [nb, kb]
    scale = amax / FP8_MAX
    q = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.view(nb * block, kb * block)[:N, :K].contiguous(), scale


@triton.jit
def fp8_block_gemm_kernel(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, M, N, K,
                          stride_am, stride_bn, stride_cm, stride_sam, stride_sbn,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :].to(tl.int64) * stride_bn + offs_k[:, None]      # B^T tile [BK, BN]
    m_mask = offs_m < M
    n_mask = offs_n < N
    # BLOCK_N divides 128, so all columns of this tile share one weight-scale row: n // 128.
    sb_row = (pid_n * BLOCK_N) // 128

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):                  # BLOCK_K == 128 == one scale group
        k_left = K - kb * BLOCK_K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & n_mask[None, :], other=0.0)
        sa = tl.load(sa_ptr + offs_m * stride_sam + kb, mask=m_mask, other=0.0)   # [BLOCK_M]
        sb = tl.load(sb_ptr + sb_row * stride_sbn + kb)                             # scalar
        acc += tl.dot(a, b) * (sa[:, None] * sb)              # fp8 x fp8 -> fp32, then rescale
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def fp8_block_gemm(a_q, a_s, b_q, b_s, out_dtype=torch.bfloat16, block_m=64, block_n=128):
    """a_q [M, K] fp8 + a_s [M, K/128]; b_q [N, K] fp8 + b_s [N/128, K/128] -> C [M, N] = A @ B^T."""
    M, K = a_q.shape
    N = b_q.shape[0]
    assert 128 % block_n == 0 and a_q.is_contiguous() and b_q.is_contiguous()
    c = torch.empty((M, N), dtype=out_dtype, device=a_q.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    fp8_block_gemm_kernel[grid](a_q, b_q, c, a_s, b_s, M, N, K,
                                a_q.stride(0), b_q.stride(0), c.stride(0), a_s.stride(0), b_s.stride(0),
                                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128, num_warps=4, num_stages=3)
    return c
