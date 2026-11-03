"""T1 · Fused add + GELU in Triton — reference solution.

out = gelu_tanh(x + y), any shape, fp32 / fp16 / bf16. Math in fp32, result in x.dtype.

Triton's programming model in one kernel:
  * a "program" (≈ a CUDA thread block) handles BLOCK_SIZE consecutive elements;
  * tl.arange gives the program a *vector* of offsets — you write block-level code,
    the compiler spreads the vector across the block's threads;
  * every load/store past the end is disabled with a mask.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_gelu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                       # which block of elements this program owns
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE] element offsets
    mask = offs < n_elements                          # the last program is usually partial

    # Load, upcast to fp32 (fp16 x^3 overflows/loses precision quickly).
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = x + y

    # GELU (tanh approximation): 0.5 z (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3))).
    # tanh(u) = 2 sigmoid(2u) - 1; tl.sigmoid is available on every backend.
    inner = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    out = 0.5 * z * (1.0 + t)

    # Cast back to the output's element type before storing.
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def add_gelu(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.shape == y.shape and x.dtype == y.dtype and x.device == y.device
    # The kernel indexes memory as a flat array, so it needs contiguous inputs.
    x, y = x.contiguous(), y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    # One program per BLOCK_SIZE elements. grid can also be a lambda over the meta-params:
    #   grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    grid = (triton.cdiv(n, block_size),)
    add_gelu_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out
