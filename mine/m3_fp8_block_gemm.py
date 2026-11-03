"""M3 · FP8 per-token-group quant + block-scaled GEMM — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
GROUP = 128


@triton.jit
def per_token_group_quant_kernel(x_ptr, q_ptr, s_ptr, K, eps, GROUP_SIZE: tl.constexpr):
    # TODO: program (row, group): amax = max(|x|, eps); scale = amax / 448; q = clamp(x / scale); store q (fp8) and scale
    pass


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
    # TODO: for each 128-wide K block: acc += tl.dot(a_fp8, b_fp8) * sa[:, None] * sb
    # TODO: sa = a_s[offs_m, kb] (one per row); sb = b_s[(pid_n * BLOCK_N) // 128, kb] (one per tile)
    pass


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
