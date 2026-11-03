import pytest
import torch
import torch.nn.functional as F

from common.env import DEVICE, HAS_CUDA
from drill import load


def ref_moe(x, logits, w13, w2, top_k):
    p = torch.softmax(logits.float(), -1)
    w, ids = torch.topk(p, top_k, -1)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    I = w13.shape[1] // 2
    for t in range(x.shape[0]):
        for k in range(top_k):
            e = int(ids[t, k])
            h = x[t].float() @ w13[e].float().t()
            h = F.silu(h[:I]) * h[I:]
            out[t] += w[t, k] * (h @ w2[e].float().t())
    return out


@pytest.mark.parametrize("T,E,K,H,I", [(1, 4, 2, 64, 32), (37, 8, 2, 128, 96), (64, 16, 4, 64, 64)])
def test_moe_layer(T, E, K, H, I):
    m = load("m2_fused_moe")
    torch.manual_seed(0)
    x = (torch.randn(T, H, device=DEVICE) * 0.5).half()
    logits = torch.randn(T, E, device=DEVICE)
    w13 = (torch.randn(E, 2 * I, H, device=DEVICE) / H ** 0.5).half()
    w2 = (torch.randn(E, H, I, device=DEVICE) / I ** 0.5).half()
    out = m.moe_layer(x, logits, w13, w2, K, block_m=16 if not HAS_CUDA else 64)
    assert out.shape == (T, H) and out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref_moe(x, logits, w13, w2, K), rtol=2e-2, atol=2e-2)


def test_single_gemm_matches_gather():
    m = load("m2_fused_moe")
    from solutions.m1_moe_align import moe_align_block_size
    T, K, E, N, Kd = 20, 2, 5, 48, 64
    x = torch.randn(T, Kd, device=DEVICE).half()
    w = torch.randn(E, N, Kd, device=DEVICE).half()
    ids = torch.stack([torch.randperm(E)[:K] for _ in range(T)]).int().to(DEVICE)
    sorted_ids, expert_ids, _ = moe_align_block_size(ids, E, 16)
    out = m.fused_moe_gemm(x, w, sorted_ids, expert_ids, T * K, K, block_m=16, block_n=32, block_k=32)
    ref = torch.einsum("tkd,tknd->tkn", x.float()[:, None].expand(T, K, Kd), w.float()[ids.long()])
    torch.testing.assert_close(out.view(T, K, N).float(), ref, rtol=1e-2, atol=5e-2)
