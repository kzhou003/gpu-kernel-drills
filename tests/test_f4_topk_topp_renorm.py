import pytest
import torch

from common.env import DEVICE, HAS_CUDA, HAS_FLASHINFER
from drill import load


def make_probs(B, V, temp=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(B, V, generator=g) / temp, -1).to(DEVICE)


def ref_top_k(p, k):
    kth = torch.topk(p, k, dim=-1).values[:, -1:]
    out = torch.where(p >= kth, p, 0.0)
    return out / out.sum(-1, keepdim=True)


def ref_top_p(p, top_p):
    sp, idx = torch.sort(p, -1, descending=True)
    keep_sorted = (sp.cumsum(-1) - sp) < top_p            # mass strictly before this entry < p
    keep = torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)
    out = torch.where(keep, p, 0.0)
    return out / out.sum(-1, keepdim=True)


V_SIZES = [1000, 32000] if HAS_CUDA else [1000, 5000]


@pytest.mark.parametrize("V", V_SIZES)
@pytest.mark.parametrize("k", [1, 10, 50])
def test_top_k(V, k):
    m = load("f4_topk_topp_renorm")
    p = make_probs(4, V)
    out = m.top_k_renorm_probs(p, k)
    assert ((out > 0).sum(-1) == k).all()
    torch.testing.assert_close(out, ref_top_k(p, k), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("V", V_SIZES)
@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9])
def test_top_p(V, top_p):
    m = load("f4_topk_topp_renorm")
    p = make_probs(4, V, temp=0.5)
    out = m.top_p_renorm_probs(p, top_p)
    torch.testing.assert_close(out, ref_top_p(p, top_p), rtol=1e-5, atol=1e-7)


def test_per_row_parameters():
    m = load("f4_topk_topp_renorm")
    p = make_probs(3, 2000)
    ks = torch.tensor([1, 7, 100], device=DEVICE)
    out = m.top_k_renorm_probs(p, ks)
    assert (out > 0).sum(-1).tolist() == [1, 7, 100]


@pytest.mark.skipif(not HAS_FLASHINFER, reason="needs a GPU with flashinfer installed")
def test_matches_flashinfer():
    import flashinfer
    m = load("f4_topk_topp_renorm")
    p = make_probs(32, 151936, temp=0.7)                   # Qwen-sized vocabulary
    torch.testing.assert_close(m.top_k_renorm_probs(p, 20), flashinfer.top_k_renorm_probs(p, 20), rtol=1e-4, atol=1e-7)
    # top-p: where the nucleus boundary lands within ~1e-6 of p, fp32 accumulation order decides
    # whether the boundary token is in. Allow a 1-token difference there; elsewhere be identical.
    a, b = m.top_p_renorm_probs(p, 0.8), flashinfer.top_p_renorm_probs(p, 0.8)
    na, nb = (a > 0).sum(-1), (b > 0).sum(-1)
    assert ((na - nb).abs() <= 1).all()
    same = na == nb
    assert same.float().mean() > 0.5
    torch.testing.assert_close(a[same], b[same], rtol=1e-4, atol=1e-7)
