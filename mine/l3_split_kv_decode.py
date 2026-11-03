"""L3 · Split-KV decode attention (Flash-Decoding) + log-sum-exp merge — YOUR IMPLEMENTATION.

Pass 1: each (seq, kv head, split) program -> partial output + LSE. Pass 2: merge.
The wrapper is done; implement both kernels.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def split_kv_partial_kernel(Q, K_cache, V_cache, block_tables, context_lens, O_part, LSE, scale,
                            stride_qb, stride_qh,
                            stride_cb, stride_ct, stride_ch,
                            stride_btb,
                            stride_pb, stride_ph, stride_ps,      # O_part [B, Hq, NUM_SPLITS, D] fp32
                            stride_lb, stride_lh,                 # LSE    [B, Hq, NUM_SPLITS]    fp32
                            NUM_SPLITS: tl.constexpr,
                            GROUP: tl.constexpr, GROUP_PAD: tl.constexpr,
                            PAGE: tl.constexpr, HEAD_DIM: tl.constexpr):
    # TODO: like L2, but only over pages [p_start, p_end) of this split;
    # write o = acc / l_i (0 for an empty split) and lse = m_i + log(l_i) (-inf for an empty split)
    pass


@triton.jit
def merge_kernel(O_part, LSE, O,
                 stride_pb, stride_ph, stride_ps, stride_lb, stride_lh, stride_ob, stride_oh,
                 NUM_SPLITS: tl.constexpr, SPLIT_PAD: tl.constexpr, HEAD_DIM: tl.constexpr):
    # TODO: load the NUM_SPLITS lse values (pad with -inf) and partial outputs;
    # w = exp(lse - max(lse)); out = sum(w * o) / sum(w)
    pass


def split_kv_decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                              block_tables: torch.Tensor, context_lens: torch.Tensor,
                              num_splits: int = 4, scale: float | None = None) -> torch.Tensor:
    B, Hq, D = q.shape
    NB, PAGE, Hkv, D2 = k_cache.shape
    assert D == D2 and v_cache.shape == k_cache.shape and Hq % Hkv == 0
    assert k_cache.is_contiguous() and v_cache.is_contiguous() and q.stride(-1) == 1
    assert PAGE >= 16 and PAGE & (PAGE - 1) == 0 and D >= 16 and D & (D - 1) == 0
    assert num_splits >= 1
    GROUP = Hq // Hkv
    GROUP_PAD = max(16, triton.next_power_of_2(GROUP))
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Partial results stay in fp32: they are combined again, so keep the extra precision.
    o_part = torch.empty((B, Hq, num_splits, D), dtype=torch.float32, device=q.device)
    lse = torch.empty((B, Hq, num_splits), dtype=torch.float32, device=q.device)
    split_kv_partial_kernel[(B, Hkv, num_splits)](
        q, k_cache, v_cache, block_tables, context_lens, o_part, lse, scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_tables.stride(0),
        o_part.stride(0), o_part.stride(1), o_part.stride(2),
        lse.stride(0), lse.stride(1),
        NUM_SPLITS=num_splits, GROUP=GROUP, GROUP_PAD=GROUP_PAD, PAGE=PAGE, HEAD_DIM=D, num_warps=4)

    out = torch.empty_like(q)
    merge_kernel[(B, Hq)](
        o_part, lse, out,
        o_part.stride(0), o_part.stride(1), o_part.stride(2), lse.stride(0), lse.stride(1),
        out.stride(0), out.stride(1),
        NUM_SPLITS=num_splits, SPLIT_PAD=max(2, triton.next_power_of_2(num_splits)), HEAD_DIM=D, num_warps=1)
    return out
