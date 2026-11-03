import pytest
import torch

from drill import load
from tests._cuda import requires_cuda

pytestmark = requires_cuda


@pytest.mark.parametrize("n", [1, 31, 32, 33, 255, 256, 1000, 65_537, 10_000_019])
def test_reduce_sum(n):
    m = load("c2_reduce_sum")
    x = torch.rand(n, device="cuda")            # positive values: relative error is meaningful
    out = m.reduce_sum(x)
    assert out.shape == () and out.dtype == torch.float32
    ref = x.double().sum().float()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_reduce_sum_empty():
    m = load("c2_reduce_sum")
    assert m.reduce_sum(torch.empty(0, device="cuda")).item() == 0.0
