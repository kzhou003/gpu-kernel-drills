import math

import pytest
import torch

from common.env import DEVICE
from drill import load
from tests._paged import make_paged, reference_decode


@pytest.mark.parametrize("Hq,Hkv", [(4, 4), (8, 2), (6, 1)])
@pytest.mark.parametrize("page", [16, 32])
def test_paged_decode(Hq, Hkv, page):
    m = load("l2_paged_decode")
    D = 64
    ctx = [1, 17, 100]
    k_cache, v_cache, bt, cl = make_paged(ctx, Hkv, D, page, torch.float16, DEVICE)
    q = torch.randn(len(ctx), Hq, D, device=DEVICE).half()
    out = m.paged_decode_attention(q, k_cache, v_cache, bt, cl)
    assert out.shape == q.shape and out.dtype == q.dtype
    ref = reference_decode(q, k_cache, v_cache, bt, cl, 1 / math.sqrt(D))
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
