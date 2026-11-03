import math

import pytest
import torch

from common.env import DEVICE, HAS_FLASHINFER
from drill import load

LN2 = math.log(2)


def attn_state(q, k, v):
    """Ground truth: (softmax(q k^T) v, log2-sum-exp) for q [S, H, D], k/v [T, H, D]."""
    s = torch.einsum("shd,thd->sht", q.float(), k.float()) / math.sqrt(q.shape[-1])
    return torch.einsum("sht,thd->shd", s.softmax(-1), v.float()), torch.logsumexp(s, -1) / LN2


def test_merge_two_halves_equals_full_attention():
    m = load("f2_merge_state")
    S, T, H, D = 5, 90, 4, 64
    q, k, v = torch.randn(S, H, D, device=DEVICE), torch.randn(T, H, D, device=DEVICE), torch.randn(T, H, D, device=DEVICE)
    va, sa = attn_state(q, k[:40], v[:40])
    vb, sb = attn_state(q, k[40:], v[40:])
    vf, sf = attn_state(q, k, v)
    vm, sm = m.merge_state(va.half(), sa, vb.half(), sb)
    assert vm.dtype == torch.float16 and sm.dtype == torch.float32
    torch.testing.assert_close(vm.float(), vf, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(sm, sf, rtol=1e-4, atol=1e-4)


def test_merge_with_empty_state():
    m = load("f2_merge_state")
    va = torch.randn(3, 2, 32, device=DEVICE)
    sa = torch.randn(3, 2, device=DEVICE)
    vb = torch.zeros_like(va)
    sb = torch.full_like(sa, float("-inf"))           # an empty key set has weight 2^-inf = 0
    vm, sm = m.merge_state(va, sa, vb, sb)
    torch.testing.assert_close(vm, va)
    torch.testing.assert_close(sm, sa)


@pytest.mark.parametrize("N", [1, 3, 8])
def test_merge_states(N):
    m = load("f2_merge_state")
    S, T, H, D = 4, 16 * N, 3, 64
    q, k, v = torch.randn(S, H, D, device=DEVICE), torch.randn(T, H, D, device=DEVICE), torch.randn(T, H, D, device=DEVICE)
    parts = [attn_state(q, k[i * 16:(i + 1) * 16], v[i * 16:(i + 1) * 16]) for i in range(N)]
    V = torch.stack([p[0] for p in parts], 1).half()          # [S, N, H, D]
    Sx = torch.stack([p[1] for p in parts], 1)                # [S, N, H]
    vm, sm = m.merge_states(V, Sx)
    vf, sf = attn_state(q, k, v)
    torch.testing.assert_close(vm.float(), vf, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(sm, sf, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not HAS_FLASHINFER, reason="needs a GPU with flashinfer installed")
def test_matches_flashinfer():
    import flashinfer
    m = load("f2_merge_state")
    va, vb = (torch.randn(64, 32, 128, device="cuda").half() for _ in range(2))
    sa, sb = (torch.randn(64, 32, device="cuda") * 4 for _ in range(2))
    v1, s1 = m.merge_state(va, sa, vb, sb)
    v2, s2 = flashinfer.merge_state(va, sa, vb, sb)
    torch.testing.assert_close(v1, v2, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(s1, s2, rtol=1e-5, atol=1e-5)
    V, Sx = torch.randn(64, 5, 32, 128, device="cuda").half(), torch.randn(64, 5, 32, device="cuda")
    v1, s1 = m.merge_states(V, Sx)
    v2, s2 = flashinfer.merge_states(V, Sx)
    torch.testing.assert_close(v1, v2, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(s1, s2, rtol=1e-5, atol=1e-5)
