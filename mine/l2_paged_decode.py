"""L2 · Paged decode attention with GQA — YOUR IMPLEMENTATION.

q [B, Hq, D], k/v_cache [num_blocks, PAGE, Hkv, D], block_tables [B, max_blocks] int32,
context_lens [B] int32 -> out [B, Hq, D]. One program per (sequence, kv head).
The wrapper is done; implement paged_decode_kernel.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def paged_decode_kernel(Q, K_cache, V_cache, block_tables, context_lens, O, scale,
                        stride_qb, stride_qh,
                        stride_cb, stride_ct, stride_ch,          # cache strides: block, token, head (dim stride 1)
                        stride_btb,                               # block_tables row stride
                        stride_ob, stride_oh,
                        GROUP: tl.constexpr, GROUP_PAD: tl.constexpr,
                        PAGE: tl.constexpr, HEAD_DIM: tl.constexpr):
    # TODO:
    # b, kvh, ctx; offs_g (GROUP_PAD), offs_t (PAGE), offs_d; q_heads = kvh * GROUP + offs_g
    # load q [GROUP_PAD, D] (padded rows 0)
    # for p in range(cdiv(ctx, PAGE)): phys = block_tables[b, p]; load k, v pages [PAGE, D];
    #     s = dot(q, trans(k)) * scale; mask tokens >= ctx; online softmax update
    # store acc / l_i for the real heads
    pass


def paged_decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                           block_tables: torch.Tensor, context_lens: torch.Tensor,
                           scale: float | None = None) -> torch.Tensor:
    B, Hq, D = q.shape
    NB, PAGE, Hkv, D2 = k_cache.shape
    assert D == D2 and v_cache.shape == k_cache.shape and Hq % Hkv == 0
    assert k_cache.is_contiguous() and v_cache.is_contiguous() and q.stride(-1) == 1
    assert PAGE >= 16 and PAGE & (PAGE - 1) == 0, "page size: a power of two >= 16 (tl.dot)"
    assert D >= 16 and D & (D - 1) == 0, "head_dim: a power of two >= 16"
    GROUP = Hq // Hkv
    GROUP_PAD = max(16, triton.next_power_of_2(GROUP))
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    paged_decode_kernel[(B, Hkv)](
        q, k_cache, v_cache, block_tables, context_lens, out, scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        GROUP=GROUP, GROUP_PAD=GROUP_PAD, PAGE=PAGE, HEAD_DIM=D, num_warps=4)
    return out
