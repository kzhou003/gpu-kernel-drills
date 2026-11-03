import math

import pytest
import torch

from common.env import DEVICE
from drill import load
from tests._paged import make_paged, reference_decode


@pytest.mark.parametrize("num_splits", [1, 3, 4, 8])
def test_split_kv_decode(num_splits):
    m = load("l3_split_kv_decode")
    Hq, Hkv, D, page = 8, 2, 64, 16
    ctx = [1, 20, 150]                      # seq 0 has 1 page: most of its splits are empty
    k_cache, v_cache, bt, cl = make_paged(ctx, Hkv, D, page, torch.float16, DEVICE)
    q = torch.randn(len(ctx), Hq, D, device=DEVICE).half()
    out = m.split_kv_decode_attention(q, k_cache, v_cache, bt, cl, num_splits=num_splits)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert torch.isfinite(out.float()).all(), "empty splits must not produce NaN"
    ref = reference_decode(q, k_cache, v_cache, bt, cl, 1 / math.sqrt(D))
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
