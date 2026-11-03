import pytest
import torch
import torch.nn.functional as F

from common.env import DEVICE
from drill import load


@pytest.mark.parametrize("shape", [(1, 8), (5, 7), (64, 768), (3, 4, 1000)])
def test_layer_norm(shape):
    m = load("t3_layernorm")
    N = shape[-1]
    x = torch.randn(shape, device=DEVICE) * 2 + 1
    w = torch.randn(N, device=DEVICE)
    b = torch.randn(N, device=DEVICE)
    y, mean, rstd = m.layer_norm(x, w, b, 1e-5)
    torch.testing.assert_close(y, F.layer_norm(x, (N,), w, b, 1e-5), rtol=1e-4, atol=1e-4)
    x2 = x.reshape(-1, N)
    torch.testing.assert_close(mean, x2.mean(-1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rstd, torch.rsqrt(x2.var(-1, unbiased=False) + 1e-5), rtol=1e-4, atol=1e-4)


def test_layer_norm_fp16():
    m = load("t3_layernorm")
    x = torch.randn(16, 512, device=DEVICE).half()
    w = torch.ones(512, device=DEVICE).half()
    b = torch.zeros(512, device=DEVICE).half()
    y, _, _ = m.layer_norm(x, w, b, 1e-5)
    assert y.dtype == torch.float16
    ref = F.layer_norm(x.float(), (512,), w.float(), b.float(), 1e-5)
    torch.testing.assert_close(y.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [(1, 16), (7, 333), (32, 4096)])
def test_rms_norm(shape):
    m = load("t3_layernorm")
    x = torch.randn(shape, device=DEVICE)
    w = torch.randn(shape[-1], device=DEVICE)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(m.rms_norm(x, w, 1e-6), ref, rtol=1e-4, atol=1e-4)
