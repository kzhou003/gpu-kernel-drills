import pytest
import torch

from common.env import DEVICE
from drill import load


@pytest.mark.parametrize("shape", [(1, 1), (4, 7), (33, 128), (17, 1000), (8, 4097)])
def test_softmax_fp32(shape):
    m = load("t2_softmax")
    x = torch.randn(shape, device=DEVICE) * 5
    torch.testing.assert_close(m.softmax(x), torch.softmax(x, dim=-1), rtol=1e-5, atol=1e-6)


def test_softmax_large_values_are_stable():
    m = load("t2_softmax")
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1e4, 0.0, 1e4]], device=DEVICE)
    out = m.softmax(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


def test_softmax_fp16_and_row_stride():
    m = load("t2_softmax")
    big = torch.randn(10, 300, device=DEVICE).half()
    x = big[:, :257]                                     # rows are strided (stride 300)
    out = m.softmax(x)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), torch.softmax(x.float(), dim=-1), rtol=1e-2, atol=1e-3)
