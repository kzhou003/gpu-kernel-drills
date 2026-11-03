import pytest
import torch
import torch.nn.functional as F

from common.env import DEVICE
from drill import load


@pytest.mark.parametrize("n", [1, 1023, 1024, 1025, 100_003])
def test_add_gelu_fp32(n):
    m = load("t1_add_gelu")
    x = torch.randn(n, device=DEVICE) * 3
    y = torch.randn(n, device=DEVICE)
    torch.testing.assert_close(m.add_gelu(x, y), F.gelu(x + y, approximate="tanh"), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_add_gelu_half(dtype):
    m = load("t1_add_gelu")
    x = torch.randn(7, 333, device=DEVICE).to(dtype)
    y = torch.randn(7, 333, device=DEVICE).to(dtype)
    out = m.add_gelu(x, y)
    assert out.dtype == dtype and out.shape == x.shape
    ref = F.gelu(x.float() + y.float(), approximate="tanh").to(dtype)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_add_gelu_noncontiguous_input():
    m = load("t1_add_gelu")
    x = torch.randn(64, 48, device=DEVICE).t()          # a transposed view
    y = torch.randn(48, 64, device=DEVICE)
    torch.testing.assert_close(m.add_gelu(x, y), F.gelu(x + y, approximate="tanh"), rtol=1e-5, atol=1e-5)
