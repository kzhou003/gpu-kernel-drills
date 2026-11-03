"""M2 · Fused MoE grouped GEMM in Triton (the vLLM fused_moe_kernel structure) — reference solution.

One kernel launch computes the expert matmul for every (token, slot) pair:
    out[f, :] = x[f // K, :] @ W[expert(f)]^T  (* topk_weight[f] if MUL_WEIGHT)
using the padded, expert-grouped order from M1. Grid = (padded blocks of BLOCK_M rows) x (N tiles);
each M-block belongs to exactly one expert, so its B operand is a single expert's weight tile.

moe_layer() is a full SwiGLU MoE MLP on top of it:
    route -> align -> GEMM1 (x @ W13^T, [.., 2I]) -> silu_and_mul -> GEMM2 (h @ W2^T, * weight) -> sum over K
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from drill import load

_m1 = load("m1_moe_align")   # your M1 (or the reference with DRILL_IMPL=solutions)
route, moe_align_block_size = _m1.route, _m1.moe_align_block_size


@triton.jit
def fused_moe_kernel(a_ptr, w_ptr, out_ptr, topk_w_ptr, sorted_ids_ptr, expert_ids_ptr,
                     N, Kdim, num_valid, top_k,
                     stride_am, stride_we, stride_wn, stride_om,
                     A_ROWS_ARE_TOKENS: tl.constexpr, MUL_WEIGHT: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)                       # which padded block of BLOCK_M (token, slot) rows
    pid_n = tl.program_id(1)                       # which N tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    flat_ids = tl.load(sorted_ids_ptr + offs_m)    # (token, slot) ids; padding rows = num_valid
    row_mask = flat_ids < num_valid
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # GEMM1 reads x[token]: several slots of one token share the same input row (gather).
    # GEMM2 reads h[flat_id]: every (token, slot) has its own intermediate row.
    if A_ROWS_ARE_TOKENS:
        a_rows = flat_ids // top_k
    else:
        a_rows = flat_ids
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + a_rows[:, None].to(tl.int64) * stride_am + offs_k[None, :]
    # W[e] is [N, Kdim] row-major; load a [BLOCK_K, BLOCK_N] tile of W[e]^T.
    w_ptrs = w_ptr + expert * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(Kdim, BLOCK_K)):
        k_left = Kdim - k0 * BLOCK_K
        a = tl.load(a_ptrs, mask=row_mask[:, None] & (offs_k[None, :] < k_left), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    if MUL_WEIGHT:                                 # fold the router weight into the epilogue
        tw = tl.load(topk_w_ptr + flat_ids, mask=row_mask, other=0.0)
        acc = acc * tw[:, None]
    out_ptrs = out_ptr + flat_ids[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=row_mask[:, None] & (offs_n[None, :] < N))


def fused_moe_gemm(a, w, sorted_ids, expert_ids, num_valid, top_k, topk_weights=None,
                   a_rows_are_tokens=True, block_m=64, block_n=64, block_k=32):
    """a [T, Kdim] (tokens) or [T*K, Kdim]; w [E, N, Kdim] -> out [T*K, N] in a.dtype."""
    E, N, Kdim = w.shape
    assert a.stride(-1) == 1 and w.is_contiguous()
    out = torch.zeros((num_valid, N), dtype=a.dtype, device=a.device)
    tw = topk_weights.reshape(-1).float().contiguous() if topk_weights is not None else out
    grid = (sorted_ids.numel() // block_m, triton.cdiv(N, block_n))
    fused_moe_kernel[grid](a, w, out, tw, sorted_ids, expert_ids, N, Kdim, num_valid, top_k,
                           a.stride(0), w.stride(0), w.stride(1), out.stride(0),
                           A_ROWS_ARE_TOKENS=a_rows_are_tokens, MUL_WEIGHT=topk_weights is not None,
                           BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return out


def moe_layer(x, router_logits, w13, w2, top_k, block_m=64):
    """x [T, H]; w13 [E, 2I, H] (gate rows then up rows); w2 [E, H, I] -> [T, H]."""
    T, H = x.shape
    E = w13.shape[0]
    topk_w, topk_ids = route(router_logits, top_k)
    sorted_ids, expert_ids, _ = moe_align_block_size(topk_ids, E, block_m)
    n = T * top_k
    h = fused_moe_gemm(x, w13, sorted_ids, expert_ids, n, top_k, block_m=block_m)          # [T*K, 2I]
    I = h.shape[1] // 2
    h = (F.silu(h[:, :I].float()) * h[:, I:].float()).to(x.dtype)                         # F3's silu_and_mul
    y = fused_moe_gemm(h, w2, sorted_ids, expert_ids, n, top_k, topk_weights=topk_w,
                       a_rows_are_tokens=False, block_m=block_m)                          # [T*K, H]
    return y.view(T, top_k, H).float().sum(1).to(x.dtype)                                 # combine slots
