import pytest
import torch
import torch.nn.functional as F

from drill import load
from tests._cuda import requires_cuda

pytestmark = requires_cuda


@pytest.mark.parametrize("n", [1, 255, 256, 257, 1_000_003])
def test_vector_add(n):
    m = load("c1_elementwise")
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    torch.testing.assert_close(m.vector_add(a, b), a + b)


def test_vector_add_empty():
    m = load("c1_elementwise")
    a = torch.empty(0, device="cuda")
    assert m.vector_add(a, a).numel() == 0


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (128, 768), (1000, 4097)])
def test_bias_gelu(shape):
    m = load("c1_elementwise")
    x = torch.randn(shape, device="cuda") * 3
    bias = torch.randn(shape[1], device="cuda")
    ref = F.gelu(x + bias, approximate="tanh")
    torch.testing.assert_close(m.bias_gelu(x, bias), ref, rtol=1e-5, atol=1e-5)
