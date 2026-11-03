import pytest
import torch
import torch.nn.functional as F

from common.env import BF16_OK, DEVICE, HAS_CUDA
from drill import load


def reference(q, k, v, causal):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,S,D", [(1, 1, 16, 16), (2, 3, 100, 64), (1, 2, 257, 32)])
def test_flash_attention(B, H, S, D, causal):
    m = load("t5_flash_attn")
    q, k, v = (torch.randn(B, H, S, D, device=DEVICE).half() for _ in range(3))
    out = m.flash_attention(q, k, v, causal=causal)
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out.float(), reference(q, k, v, causal), rtol=1e-2, atol=1e-2)


def test_custom_scale():
    m = load("t5_flash_attn")
    q, k, v = (torch.randn(1, 2, 64, 32, device=DEVICE).half() for _ in range(3))
    out = m.flash_attention(q, k, v, sm_scale=0.05)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=0.05)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_CUDA, reason="larger shapes are slow in the CPU interpreter")
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_large(causal):
    m = load("t5_flash_attn")
    dtype = torch.bfloat16 if BF16_OK else torch.float16
    q, k, v = (torch.randn(2, 8, 2048, 128, device=DEVICE, dtype=dtype) for _ in range(3))
    torch.testing.assert_close(m.flash_attention(q, k, v, causal=causal).float(), reference(q, k, v, causal),
                               rtol=2e-2, atol=2e-2)
