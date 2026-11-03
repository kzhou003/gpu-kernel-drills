"""L3 · Split-KV decode attention (Flash-Decoding) + log-sum-exp merge — reference solution.

L2 launches B * Hkv programs. With a small batch and a long context that is far fewer
programs than the GPU has SMs, and each one loops over thousands of tokens: the GPU sits
mostly idle. Split-KV also splits the context of each sequence into NUM_SPLITS ranges:

  pass 1 (grid B x Hkv x NUM_SPLITS): attention over one range of pages -> a partial output
          o_s (already normalised within the range) and its log-sum-exp lse_s = m_s + log(l_s)
  pass 2 (grid B x Hq): merge the partials exactly:
          w_s = exp(lse_s - max_s lse_s),  out = sum_s w_s * o_s / sum_s w_s
An empty range (short sequences) gets lse = -inf and weight 0.
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
    b = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1).to(tl.int64)
    split = tl.program_id(2)
    ctx = tl.load(context_lens + b)

    # This program's contiguous range of pages: [p_start, p_end).
    num_pages = tl.cdiv(ctx, PAGE)
    pages_per_split = tl.cdiv(num_pages, NUM_SPLITS)
    p_start = split * pages_per_split
    p_end = tl.minimum(p_start + pages_per_split, num_pages)   # may be <= p_start: empty split

    offs_g = tl.arange(0, GROUP_PAD)
    offs_t = tl.arange(0, PAGE)
    offs_d = tl.arange(0, HEAD_DIM)
    g_mask = offs_g < GROUP
    q_heads = kvh * GROUP + offs_g
    q = tl.load(Q + b * stride_qb + q_heads[:, None] * stride_qh + offs_d[None, :],
                mask=g_mask[:, None], other=0.0)

    m_i = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([GROUP_PAD], dtype=tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], dtype=tl.float32)

    for p in range(p_start, p_end):                  # identical body to L2
        phys = tl.load(block_tables + b * stride_btb + p).to(tl.int64)
        tok = p * PAGE + offs_t
        t_mask = tok < ctx
        kv_off = phys * stride_cb + offs_t[:, None] * stride_ct + kvh * stride_ch + offs_d[None, :]
        k = tl.load(K_cache + kv_off, mask=t_mask[:, None], other=0.0)
        v = tl.load(V_cache + kv_off, mask=t_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(t_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p_ = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p_, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p_.to(v.dtype), v, acc)
        m_i = m_new

    # Finalise this range. An empty range has l_i == 0: write o = 0 and lse = -inf so the
    # merge gives it weight exp(-inf) = 0 (and never divide by zero).
    has = l_i > 0
    o = acc / tl.where(has, l_i, 1.0)[:, None]
    lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))

    tl.store(O_part + b * stride_pb + q_heads[:, None] * stride_ph + split * stride_ps + offs_d[None, :],
             o, mask=g_mask[:, None])
    tl.store(LSE + b * stride_lb + q_heads * stride_lh + split, lse, mask=g_mask)


@triton.jit
def merge_kernel(O_part, LSE, O,
                 stride_pb, stride_ph, stride_ps, stride_lb, stride_lh, stride_ob, stride_oh,
                 NUM_SPLITS: tl.constexpr, SPLIT_PAD: tl.constexpr, HEAD_DIM: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    offs_s = tl.arange(0, SPLIT_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    s_mask = offs_s < NUM_SPLITS

    lse = tl.load(LSE + b * stride_lb + h * stride_lh + offs_s, mask=s_mask, other=float("-inf"))
    m = tl.max(lse, axis=0)                  # finite: context_len >= 1, so some split is non-empty
    w = tl.exp(lse - m)                      # [SPLIT_PAD]; empty or padded splits -> 0
    o = tl.load(O_part + b * stride_pb + h * stride_ph + offs_s[:, None] * stride_ps + offs_d[None, :],
                mask=s_mask[:, None], other=0.0)                       # [SPLIT_PAD, D]
    out = tl.sum(w[:, None] * o, axis=0) / tl.sum(w, axis=0)
    tl.store(O + b * stride_ob + h * stride_oh + offs_d, out.to(O.dtype.element_ty))


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
