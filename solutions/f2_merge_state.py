"""F2 · merge_state / merge_states with FlashInfer's base-2 LSE — reference solution.

An attention "state" over a subset of keys is (v, s): v is the softmax-normalised output over
that subset, s the log-sum-exp of its scores. FlashInfer stores s in BASE 2:
    s = log2( sum_j exp(score_j) )          (natural-log LSE = s * ln 2)
Merging two states (same queries, disjoint key sets):
    m   = max(s_a, s_b)
    w_a = 2^(s_a - m),  w_b = 2^(s_b - m)
    v   = (w_a v_a + w_b v_b) / (w_a + w_b),   s = m + log2(w_a + w_b)
Shapes follow flashinfer.merge_state / merge_states:
    merge_state : v_a, v_b [S, H, D]; s_a, s_b [S, H] fp32  -> (v [S, H, D], s [S, H])
    merge_states: v [S, N, H, D];     s [S, N, H]   fp32    -> (v [S, H, D], s [S, H])
"""
import torch
import triton
import triton.language as tl


@triton.jit
def merge_state_kernel(va_ptr, sa_ptr, vb_ptr, sb_ptr, vo_ptr, so_ptr, H, D,
                       BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)       # one program per (position, head) = pid
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    sa = tl.load(sa_ptr + pid)
    sb = tl.load(sb_ptr + pid)
    m = tl.maximum(sa, sb)
    wa = tl.exp2(sa - m)                      # base 2 throughout: exp2 / log2
    wb = tl.exp2(sb - m)
    va = tl.load(va_ptr + pid * D + offs, mask=mask, other=0.0).to(tl.float32)
    vb = tl.load(vb_ptr + pid * D + offs, mask=mask, other=0.0).to(tl.float32)
    v = (wa * va + wb * vb) / (wa + wb)
    tl.store(vo_ptr + pid * D + offs, v.to(vo_ptr.dtype.element_ty), mask=mask)
    tl.store(so_ptr + pid, m + tl.log2(wa + wb))


@triton.jit
def merge_states_kernel(v_ptr, s_ptr, vo_ptr, so_ptr, N, H, D,
                        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)       # (position, head)
    pos = pid // H
    h = pid % H
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < N
    # s[pos, n, h] lives at (pos * N + n) * H + h
    s = tl.load(s_ptr + (pos * N + offs_n) * H + h, mask=n_mask, other=float("-inf"))
    m = tl.max(s, axis=0)
    w = tl.exp2(s - m)                        # padded states -> 2^-inf = 0
    v = tl.load(v_ptr + ((pos * N + offs_n[:, None]) * H + h) * D + offs_d[None, :],
                mask=n_mask[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)
    w_sum = tl.sum(w, axis=0)
    out = tl.sum(w[:, None] * v, axis=0) / w_sum
    tl.store(vo_ptr + pid * D + offs_d, out.to(vo_ptr.dtype.element_ty), mask=offs_d < D)
    tl.store(so_ptr + pid, m + tl.log2(w_sum))


def merge_state(v_a, s_a, v_b, s_b):
    S, H, D = v_a.shape
    v_a, v_b = v_a.contiguous(), v_b.contiguous()
    s_a, s_b = s_a.float().contiguous(), s_b.float().contiguous()
    v_out = torch.empty_like(v_a)
    s_out = torch.empty((S, H), dtype=torch.float32, device=v_a.device)
    if S * H:
        merge_state_kernel[(S * H,)](v_a, s_a, v_b, s_b, v_out, s_out, H, D,
                                     BLOCK_D=triton.next_power_of_2(D))
    return v_out, s_out


def merge_states(v, s):
    S, N, H, D = v.shape
    v, s = v.contiguous(), s.float().contiguous()
    v_out = torch.empty((S, H, D), dtype=v.dtype, device=v.device)
    s_out = torch.empty((S, H), dtype=torch.float32, device=v.device)
    if S * H:
        merge_states_kernel[(S * H,)](v, s, v_out, s_out, N, H, D,
                                      BLOCK_N=max(2, triton.next_power_of_2(N)),
                                      BLOCK_D=triton.next_power_of_2(D))
    return v_out, s_out
