"""F3 · fused_add_rmsnorm (in place) + silu_and_mul — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, x_stride, r_stride, N, eps, BLOCK_N: tl.constexpr):
    # TODO: h = x + residual (fp32); store h into residual; x = h * rsqrt(mean(h^2) + eps) * w
    pass


@triton.jit
def silu_and_mul_kernel(x_ptr, out_ptr, x_stride, out_stride, d, BLOCK: tl.constexpr):
    # TODO: gate = x[row, offs], up = x[row, d + offs]; out = gate * sigmoid(gate) * up
    pass


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> None:
    assert x.shape == residual.shape and x.shape[-1] == weight.shape[0]
    assert x.stride(-1) == 1 and residual.stride(-1) == 1
    N = x.shape[-1]
    x2, r2 = x.view(-1, N), residual.view(-1, N)       # view, not reshape: we must write in place
    M = x2.shape[0]
    if M == 0:
        return
    BLOCK_N = triton.next_power_of_2(N)
    fused_add_rmsnorm_kernel[(M,)](x2, r2, weight, x2.stride(0), r2.stride(0), N, eps,
                                   BLOCK_N=BLOCK_N, num_warps=min(max(BLOCK_N // 256, 1), 8))


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0 and x.stride(-1) == 1
    d = x.shape[-1] // 2
    x2 = x.reshape(-1, 2 * d)
    out = torch.empty((x2.shape[0], d), dtype=x.dtype, device=x.device)
    if x2.shape[0]:
        BLOCK = min(1024, triton.next_power_of_2(d))
        silu_and_mul_kernel[(x2.shape[0], triton.cdiv(d, BLOCK))](x2, out, x2.stride(0), out.stride(0), d, BLOCK=BLOCK)
    return out.reshape(*x.shape[:-1], d)
