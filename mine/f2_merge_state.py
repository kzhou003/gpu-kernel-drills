"""F2 · merge_state / merge_states (base-2 LSE) — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
import torch
import triton
import triton.language as tl


@triton.jit
def merge_state_kernel(va_ptr, sa_ptr, vb_ptr, sb_ptr, vo_ptr, so_ptr, H, D,
                       BLOCK_D: tl.constexpr):
    # TODO: pid = (position, head); load s_a, s_b and the two D-vectors
    # TODO: m = max; w = exp2(s - m); v = (wa va + wb vb) / (wa + wb); s = m + log2(wa + wb)
    pass


@triton.jit
def merge_states_kernel(v_ptr, s_ptr, vo_ptr, so_ptr, N, H, D,
                        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    # TODO: load the N lse values of (pos, h) (pad with -inf) and the [N, D] outputs
    # TODO: w = exp2(s - max); v = sum(w v) / sum(w); s = max + log2(sum(w))
    pass


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
