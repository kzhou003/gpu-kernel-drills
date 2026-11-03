import pytest
import torch

from drill import load
from tests._cuda import requires_cuda

pytestmark = requires_cuda

SHAPES = [(1, 1), (1, 100), (100, 1), (32, 32), (33, 31), (257, 129), (1024, 4096)]


@pytest.mark.parametrize("fn", ["transpose_naive", "transpose_tiled"])
@pytest.mark.parametrize("shape", SHAPES)
def test_transpose(fn, shape):
    m = load("c3_transpose")
    x = torch.randn(shape, device="cuda")
    y = getattr(m, fn)(x)
    assert y.shape == (shape[1], shape[0])
    torch.testing.assert_close(y, x.t().contiguous(), rtol=0, atol=0)   # a copy must be exact
