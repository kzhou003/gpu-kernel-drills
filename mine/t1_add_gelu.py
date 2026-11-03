"""T1 · Fused add + GELU in Triton — YOUR IMPLEMENTATION.

out = gelu_tanh(x + y), any shape, fp32 / fp16 / bf16; compute in fp32, store in x.dtype.
gelu_tanh(z) = 0.5 * z * (1 + tanh(0.7978845608028654 * (z + 0.044715 * z^3)))
Hint for tanh: tanh(u) = 2 * sigmoid(2u) - 1  (tl.sigmoid).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_gelu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # TODO: pid -> offsets -> mask -> load x, y (as fp32) -> gelu(x + y) -> store (cast to out dtype)
    pass


def add_gelu(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.shape == y.shape and x.dtype == y.dtype and x.device == y.device
    # TODO: make inputs contiguous, allocate out, compute the grid, launch add_gelu_kernel
    raise NotImplementedError
