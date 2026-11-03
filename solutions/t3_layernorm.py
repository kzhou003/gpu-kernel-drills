"""T3 · LayerNorm / RMSNorm forward in Triton — reference solution.

layer_norm(x[M, N], w, b, eps) -> (y, mean[M], rstd[M])
    y = (x - mean) * rstd * w + b,  rstd = 1 / sqrt(var + eps),  var is the biased variance.
rms_norm(x[M, N], w, eps) -> y
    y = x * rsqrt(mean(x^2) + eps) * w

One program per row, the full row in one block (N <= 16384). mean and rstd are returned
because the backward pass needs exactly these two numbers per row.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rstd_ptr,
                      x_row_stride, y_row_stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)

    # Statistics in fp32. Padding lanes loaded 0, so they do not disturb the sum ...
    mean = tl.sum(x, axis=0) / N
    # ... but x - mean is NOT zero there: zero it again before squaring.
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N          # two-pass variance: stabler than E[x^2] - E[x]^2
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b

    tl.store(y_ptr + row * y_row_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(mean_ptr + row, mean)             # scalar stores: one value per program
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rms_norm_kernel(x_ptr, w_ptr, y_ptr, x_row_stride, y_row_stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)   # no mean subtraction, no bias
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * y_row_stride + cols, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


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
