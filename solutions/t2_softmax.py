"""T2 · Fused row softmax in Triton — reference solution.

softmax(x) over the last dim of a 2-D tensor [M, N]. One program per row; the whole row
fits in one block (BLOCK_SIZE = next_power_of_2(N)), so x is read once and y written once.
PyTorch's unfused version reads x for max, again for exp/sum, again to normalise.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # program_id is int32; promote before multiplying by a stride so row * stride cannot
    # overflow for tensors with more than 2^31 elements.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)        # BLOCK_SIZE is a power of two >= n_cols
    mask = cols < n_cols

    # Padding lanes load -inf: they lose the max, and exp(-inf) = 0 adds nothing to the sum.
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf")).to(tl.float32)

    # Numerically stable softmax: softmax(x) = softmax(x - max(x)); the largest exponent is 0,
    # so exp never overflows even for fp16 inputs around 1e4.
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "expects [M, N]"
    M, N = x.shape
    # The kernel walks a row with unit stride; the rows themselves may be strided.
    if x.stride(1) != 1:
        x = x.contiguous()
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0 or N == 0:
        return y
    BLOCK_SIZE = triton.next_power_of_2(N)
    # More warps for wider rows, so each thread holds a reasonable number of elements.
    num_warps = 4 if BLOCK_SIZE <= 2048 else (8 if BLOCK_SIZE <= 8192 else 16)
    softmax_kernel[(M,)](y, x, x.stride(0), y.stride(0), N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y
