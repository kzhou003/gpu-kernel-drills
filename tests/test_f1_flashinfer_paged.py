import math

import pytest
import torch

from common.env import DEVICE, HAS_FLASHINFER
from drill import load
from tests._paged import make_paged, reference_decode


def test_block_table_to_csr_by_hand():
    m = load("f1_flashinfer_paged")
    bt = torch.tensor([[7, 2, 5, 0], [3, 0, 0, 0], [1, 4, 6, 8]], dtype=torch.int32)
    ctx = torch.tensor([33, 16, 49], dtype=torch.int32)        # page_size 16: 3, 1, 4 pages
    indptr, indices, last = m.block_table_to_csr(bt, ctx, 16)
    assert indptr.dtype == indices.dtype == last.dtype == torch.int32
    assert indptr.tolist() == [0, 3, 4, 8]
    assert indices.tolist() == [7, 2, 5, 3, 1, 4, 6, 8]
    assert last.tolist() == [1, 16, 1]                        # 33 = 2*16 + 1; 16 fills one page


def test_roundtrip():
    m = load("f1_flashinfer_paged")
    g = torch.Generator().manual_seed(0)
    ctx = torch.randint(1, 200, (9,), generator=g).int()
    P = 16
    pages = (ctx + P - 1) // P
    bt = torch.zeros(9, int(pages.max()), dtype=torch.int32)
    perm = torch.randperm(int(pages.sum()), generator=g).int()
    used = 0
    for b, n in enumerate(pages.tolist()):
        bt[b, :n] = perm[used:used + n]
        used += n
    bt2, ctx2 = m.csr_to_block_table(*m.block_table_to_csr(bt, ctx, P), P)
    assert torch.equal(ctx2, ctx)
    for b, n in enumerate(pages.tolist()):
        assert torch.equal(bt2[b, :n], bt[b, :n])


@pytest.mark.skipif(not HAS_FLASHINFER, reason="needs a GPU with flashinfer installed")
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
def test_flashinfer_decode_matches_reference_and_lse_is_base2(Hq, Hkv):
    m = load("f1_flashinfer_paged")
    D = 128
    ctx = [1, 37, 300, 1000]
    k_cache, v_cache, bt, cl = make_paged(ctx, Hkv, D, 16, torch.float16, DEVICE)
    q = torch.randn(len(ctx), Hq, D, device=DEVICE).half()
    out, lse = m.flashinfer_decode(q, k_cache, v_cache, bt, cl, return_lse=True)
    ref = reference_decode(q, k_cache, v_cache, bt, cl, 1 / math.sqrt(D))
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    # natural-log LSE for one head, computed directly
    b, h = 2, 0
    blocks = bt[b, : (ctx[b] + 15) // 16].long()
    k = k_cache[blocks].reshape(-1, Hkv, D)[: ctx[b], h // (Hq // Hkv)].float()
    lse_ln = torch.logsumexp(k @ q[b, h].float() / math.sqrt(D), 0)
    assert abs(lse[b, h].item() - lse_ln.item() / math.log(2)) < 1e-2, "FlashInfer LSE is log2-based"
