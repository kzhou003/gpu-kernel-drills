"""T5 · FlashAttention forward (tiled + online softmax) in Triton — YOUR IMPLEMENTATION.

o = softmax(q k^T * sm_scale [+ causal mask]) v, q/k/v/o: [B, H, S, D] fp16/bf16.
The wrapper is done; implement flash_attn_fwd_kernel.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd_kernel(Q, K, V, O, sm_scale,
                          stride_b, stride_h, stride_s,       # same strides for q, k, v, o; stride_d == 1
                          H, S,
                          CAUSAL: tl.constexpr, HEAD_DIM: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # TODO:
    # start_m, bh -> off_b, off_h -> base offset
    # load the q tile [BLOCK_M, HEAD_DIM]
    # m_i = -inf, l_i = 0, acc = 0 (fp32)
    # hi = min((start_m + 1) * BLOCK_M, S) if CAUSAL else S
    # for start_n in range(0, hi, BLOCK_N):
    #     load k^T [HEAD_DIM, BLOCK_N]; s = dot(q, k) * sm_scale; mask invalid keys with -inf
    #     online softmax update of m_i, l_i, acc (acc += p @ v)
    # store acc / l_i
    pass


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False,
                    sm_scale: float | None = None, block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    assert q.shape == k.shape == v.shape and q.dim() == 4, "q, k, v: [B, H, S, D]"
    assert q.dtype == k.dtype == v.dtype and q.dtype in (torch.float16, torch.bfloat16)
    B, H, S, D = q.shape
    assert D in (16, 32, 64, 128), "HEAD_DIM must be a power of two in [16, 128]"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(S, block_m), B * H)
    flash_attn_fwd_kernel[grid](q, k, v, o, sm_scale, q.stride(0), q.stride(1), q.stride(2), H, S,
                                CAUSAL=causal, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
                                num_warps=4, num_stages=2)
    return o
