"""T3 · LayerNorm / RMSNorm forward in Triton — YOUR IMPLEMENTATION.

layer_norm(x[M, N], w, b, eps) -> (y, mean[M], rstd[M]); rms_norm(x[M, N], w, eps) -> y.
The wrappers are done; implement the two kernels.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rstd_ptr,
                      x_row_stride, y_row_stride, N, eps, BLOCK_N: tl.constexpr):
    # TODO: row, cols, mask; load x as fp32; mean; xc = where(mask, x - mean, 0);
    # var (biased); rstd = 1 / sqrt(var + eps); y = xc * rstd * w + b; store y, mean, rstd
    pass


@triton.jit
def rms_norm_kernel(x_ptr, w_ptr, y_ptr, x_row_stride, y_row_stride, N, eps, BLOCK_N: tl.constexpr):
    # TODO: rstd = 1 / sqrt(mean(x^2) + eps); y = x * rstd * w
    pass


def _launch_params(N: int) -> tuple[int, int]:
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > 16384:
        raise ValueError("this single-block kernel supports N <= 16384; loop over the row for more")
    num_warps = min(max(BLOCK_N // 256, 1), 8)   # ~256 elements per warp, 1..8 warps
    return BLOCK_N, num_warps


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])                # [..., N] -> [M, N]
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M, N = x2.shape
    y = torch.empty_like(x2)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    if M > 0:
        BLOCK_N, num_warps = _launch_params(N)
        layer_norm_kernel[(M,)](x2, w, b, y, mean, rstd, x2.stride(0), y.stride(0), N, eps,
                                BLOCK_N=BLOCK_N, num_warps=num_warps)
    return y.reshape(shape), mean, rstd


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M, N = x2.shape
    y = torch.empty_like(x2)
    if M > 0:
        BLOCK_N, num_warps = _launch_params(N)
        rms_norm_kernel[(M,)](x2, w, y, x2.stride(0), y.stride(0), N, eps, BLOCK_N=BLOCK_N, num_warps=num_warps)
    return y.reshape(shape)
