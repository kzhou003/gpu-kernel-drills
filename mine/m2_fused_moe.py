"""M2 · fused MoE grouped GEMM — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
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
    # TODO: flat_ids = sorted_ids[block]; row_mask = flat_ids < num_valid; expert = expert_ids[pid_m]
    # TODO: A rows = flat_ids // top_k (GEMM1) or flat_ids (GEMM2); B tile = W[expert]^T [BLOCK_K, BLOCK_N]
    # TODO: K loop with masked loads + tl.dot; optionally * topk_weight[flat_ids]; store to out[flat_ids]
    pass


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
