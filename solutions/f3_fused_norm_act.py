"""F3 · fused_add_rmsnorm (in place) + silu_and_mul, with FlashInfer's semantics — reference solution.

fused_add_rmsnorm(x, residual, w, eps)   -> None, both tensors updated IN PLACE:
    h = x + residual          (computed in fp32)
    residual <- h             (stored in residual's dtype: the next layer's residual stream)
    x        <- h * rsqrt(mean(h^2) + eps) * w
This is the pre-norm Transformer step "add the sublayer output to the residual, then normalise
for the next sublayer" in one pass: read x, residual once; write both once.

silu_and_mul(x[..., 2d]) -> [..., d]:  silu(x[..., :d]) * x[..., d:]   (the SwiGLU MLP gate)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, x_stride, r_stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * r_stride + cols, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    tl.store(r_ptr + row * r_stride + cols, h.to(r_ptr.dtype.element_ty), mask=mask)   # new residual
    rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(x_ptr + row * x_stride + cols, (h * rstd * w).to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def silu_and_mul_kernel(x_ptr, out_ptr, x_stride, out_stride, d, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)   # 2-D grid: rows x column blocks
    mask = offs < d
    gate = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * x_stride + d + offs, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up                        # silu(g) = g * sigmoid(g)
    tl.store(out_ptr + row * out_stride + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


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
