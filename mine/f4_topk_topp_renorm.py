"""F4 · sorting-free top-k / top-p renorm — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
import torch
import triton
import triton.language as tl


@triton.jit
def _renorm_kernel(probs_ptr, out_ptr, k_ptr, p_ptr, V, IS_TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    # TODO: 1. row max -> hi = bits(max) + 1, lo = 0 (int32 bit patterns of the fp32 probabilities)
    # TODO: 2. 31 rounds: mid = lo + (hi - lo) // 2; one pass over the row: count (top-k) or mass (top-p) of x with bits >= mid
    # TODO:    keep the invariant f(lo) satisfies the condition, f(hi) does not
    # TODO: 3. kept = sum of x with bits >= lo; write x / kept for those, 0 elsewhere
    pass


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
