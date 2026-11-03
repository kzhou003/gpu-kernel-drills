"""F4 · Sorting-free top-k / top-p renormalisation — reference solution.

top_k_renorm_probs(probs [B, V], k)  : keep the k largest probabilities per row, renormalise.
top_p_renorm_probs(probs [B, V], p)  : keep the smallest set of largest probabilities whose
                                       mass reaches p (nucleus), renormalise.
Same semantics as flashinfer.top_k_renorm_probs / top_p_renorm_probs (for rows without ties).

Sorting a 150k-entry row per request per step is the slow part of naive samplers. FlashInfer
never sorts: it repeatedly reduces (mass, count) of the entries above a pivot and moves the
pivot. Here the pivot search is an exact bisection over the float's bit pattern: for x >= 0,
the int32 view of an fp32 value increases with the value, so 31 halvings of [0, bits(max)+1)
pin down the exact threshold. Each round is one pass over the row.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _renorm_kernel(probs_ptr, out_ptr, k_ptr, p_ptr, V, IS_TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = probs_ptr + row * V
    offs = tl.arange(0, BLOCK)

    # hi = bits(max) + 1 so that "count(x >= hi) == 0"; lo = 0 so that "count(x >= lo) == V".
    mx = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, V, BLOCK):
        x = tl.load(base + start + offs, mask=start + offs < V, other=0.0)
        mx = tl.maximum(mx, x)
    lo = tl.zeros([], dtype=tl.int32)
    hi = tl.max(mx, axis=0).to(tl.int32, bitcast=True) + 1

    if IS_TOP_K:
        k = tl.load(k_ptr + row)
    else:
        p = tl.load(p_ptr + row)

    # Invariant: f(lo) satisfies the condition, f(hi) does not. 31 rounds shrink hi - lo to 1.
    for _ in range(31):
        mid = lo + (hi - lo) // 2
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for start in range(0, V, BLOCK):
            x = tl.load(base + start + offs, mask=start + offs < V, other=0.0)
            above = x.to(tl.int32, bitcast=True) >= mid      # x >= value(mid), compared as bits
            if IS_TOP_K:
                acc += above.to(tl.float32)                  # count above the pivot
            else:
                acc += tl.where(above, x, 0.0)               # mass above the pivot
        total = tl.sum(acc, axis=0)
        if IS_TOP_K:
            ok = total >= k
        else:
            ok = total >= p
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid)

    # Keep x >= value(lo): exactly k entries (top-k) / the nucleus (top-p). Renormalise.
    kept = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, V, BLOCK):
        x = tl.load(base + start + offs, mask=start + offs < V, other=0.0)
        kept += tl.where(x.to(tl.int32, bitcast=True) >= lo, x, 0.0)
    denom = tl.sum(kept, axis=0)
    for start in range(0, V, BLOCK):
        m = start + offs < V
        x = tl.load(base + start + offs, mask=m, other=0.0)
        y = tl.where(x.to(tl.int32, bitcast=True) >= lo, x / denom, 0.0)
        tl.store(out_ptr + row * V + start + offs, y, mask=m)


def _per_row(val, B, dtype, device):
    if isinstance(val, torch.Tensor):
        return val.to(device=device, dtype=dtype).expand(B).contiguous()
    return torch.full((B,), val, dtype=dtype, device=device)


def top_k_renorm_probs(probs: torch.Tensor, top_k) -> torch.Tensor:
    probs = probs.float().contiguous()
    B, V = probs.shape
    out = torch.empty_like(probs)
    k = _per_row(top_k, B, torch.float32, probs.device)       # compared with a float count
    _renorm_kernel[(B,)](probs, out, k, k, V, IS_TOP_K=True, BLOCK=4096, num_warps=16)
    return out


def top_p_renorm_probs(probs: torch.Tensor, top_p) -> torch.Tensor:
    probs = probs.float().contiguous()
    B, V = probs.shape
    out = torch.empty_like(probs)
    p = _per_row(top_p, B, torch.float32, probs.device)
    _renorm_kernel[(B,)](probs, out, p, p, V, IS_TOP_K=False, BLOCK=4096, num_warps=16)
    return out
