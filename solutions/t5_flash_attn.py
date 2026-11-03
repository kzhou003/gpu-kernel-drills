"""T5 · FlashAttention forward (tiled + online softmax) in Triton — reference solution.

o = softmax(q k^T * sm_scale [+ causal mask]) v, q/k/v/o: [B, H, S, D] fp16 or bf16.
One program = BLOCK_M queries of one (batch, head). It streams K/V in BLOCK_N chunks and
never materialises the S x S score matrix: only a BLOCK_M x BLOCK_N tile exists at a time.

Online softmax state per query row i (all fp32):
    m_i  running max of the scores seen so far
    l_i  running sum of exp(score - m_i)
    acc  running sum of exp(score - m_i) * v
When a new chunk raises the max to m_new, old terms are rescaled by alpha = exp(m_i - m_new).
At the end o_i = acc_i / l_i.
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
    start_m = tl.program_id(0)                 # which block of queries
    bh = tl.program_id(1)                      # which (batch, head)
    off_b = bh // H
    off_h = bh % H
    base = off_b.to(tl.int64) * stride_b + off_h.to(tl.int64) * stride_h

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)    # query positions
    offs_n = tl.arange(0, BLOCK_N)                        # key positions within a chunk
    offs_d = tl.arange(0, HEAD_DIM)

    # The query tile is loaded once and stays in registers for the whole loop.
    q = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :],
                mask=offs_m[:, None] < S, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal: queries in this block never look past their own last position, so key chunks
    # beyond (start_m + 1) * BLOCK_M are skipped entirely (about half the work).
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, S)
    else:
        hi = S

    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        # Load K already transposed: a [HEAD_DIM, BLOCK_N] tile, so q @ k is a plain tl.dot.
        k = tl.load(K + base + cols[None, :] * stride_s + offs_d[:, None],
                    mask=cols[None, :] < S, other=0.0)
        s = tl.dot(q, k) * sm_scale                        # [BLOCK_M, BLOCK_N] fp32 scores

        valid = cols[None, :] < S                          # keys past the end of the sequence
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= cols[None, :])
        s = tl.where(valid, s, float("-inf"))

        # Online softmax update. Every real query row has at least one valid key (key 0) in
        # the first chunk, so m_new is finite from the first iteration on and no
        # (-inf) - (-inf) = NaN can appear.
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])                     # [BLOCK_M, BLOCK_N]
        alpha = tl.exp(m_i - m_new)                        # rescales the old state; 0 on the first chunk
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(V + base + cols[:, None] * stride_s + offs_d[None, :],
                    mask=cols[:, None] < S, other=0.0)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)                # acc += p @ v (p in fp16/bf16 for Tensor Cores)
        m_i = m_new

    o = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * stride_s + offs_d[None, :],
             o.to(O.dtype.element_ty), mask=offs_m[:, None] < S)


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
