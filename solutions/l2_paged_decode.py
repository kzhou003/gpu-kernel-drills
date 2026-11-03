"""L2 · Paged decode attention with GQA — reference solution.

Decode step: each sequence has ONE new query token per head, and attends over its whole
history stored in the paged KV cache.
    q            [B, Hq, D]
    k/v_cache    [num_blocks, PAGE, Hkv, D]       (Hq = GROUP * Hkv)
    block_tables [B, max_blocks] int32, context_lens [B] int32 (>= 1)
    out          [B, Hq, D]

One program per (sequence, kv head). It handles the GROUP query heads that share this kv head
together, so every K/V page is read from memory once for the whole group — decode is
memory-bound, and this is the main saving GQA offers. The group is padded to >= 16 rows so
q @ k^T can use tl.dot (Tensor Cores need M >= 16).
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
    b = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1).to(tl.int64)
    ctx = tl.load(context_lens + b)

    offs_g = tl.arange(0, GROUP_PAD)                 # query heads within the group (padded)
    offs_t = tl.arange(0, PAGE)                      # token offsets within a page
    offs_d = tl.arange(0, HEAD_DIM)
    g_mask = offs_g < GROUP
    q_heads = kvh * GROUP + offs_g                   # query heads served by this kv head

    q = tl.load(Q + b * stride_qb + q_heads[:, None] * stride_qh + offs_d[None, :],
                mask=g_mask[:, None], other=0.0)     # [GROUP_PAD, D]; padded rows are 0

    m_i = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([GROUP_PAD], dtype=tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], dtype=tl.float32)

    num_pages = tl.cdiv(ctx, PAGE)
    for p in range(0, num_pages):
        # Indirection: logical page p of this sequence -> physical block in the cache.
        phys = tl.load(block_tables + b * stride_btb + p).to(tl.int64)
        tok = p * PAGE + offs_t
        t_mask = tok < ctx                           # the last page is usually partial

        kv_off = phys * stride_cb + offs_t[:, None] * stride_ct + kvh * stride_ch + offs_d[None, :]
        k = tl.load(K_cache + kv_off, mask=t_mask[:, None], other=0.0)   # [PAGE, D]
        v = tl.load(V_cache + kv_off, mask=t_mask[:, None], other=0.0)   # [PAGE, D]

        s = tl.dot(q, tl.trans(k)) * scale           # [GROUP_PAD, PAGE] fp32
        s = tl.where(t_mask[None, :], s, float("-inf"))

        # Same online-softmax update as FlashAttention (T5). Page 0 always has token 0, so
        # m_new is finite from the first iteration.
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p_ = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p_, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p_.to(v.dtype), v, acc)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(O + b * stride_ob + q_heads[:, None] * stride_oh + offs_d[None, :],
             out.to(O.dtype.element_ty), mask=g_mask[:, None])


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
