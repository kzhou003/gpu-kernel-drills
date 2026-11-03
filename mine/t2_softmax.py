"""T2 · Fused row softmax in Triton — YOUR IMPLEMENTATION.

softmax over the last dim of x: [M, N]; one program per row, whole row in one block.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # TODO: row = program_id (as int64); cols = arange(0, BLOCK_SIZE); mask;
    #       load with other=-inf, subtract max, exp, divide by sum, store
    pass


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "expects [M, N]"
    M, N = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0 or N == 0:
        return y
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_SIZE <= 2048 else (8 if BLOCK_SIZE <= 8192 else 16)
    softmax_kernel[(M,)](y, x, x.stride(0), y.stride(0), N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y
