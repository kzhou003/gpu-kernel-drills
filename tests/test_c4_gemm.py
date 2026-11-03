import pytest
import torch

from drill import load
from tests._cuda import requires_cuda

pytestmark = requires_cuda

SHAPES = [(1, 1, 1), (17, 33, 9), (64, 64, 64), (100, 70, 130), (128, 512, 256), (513, 257, 1025)]


@pytest.mark.parametrize("fn", ["matmul_naive", "matmul_smem", "matmul_regtile"])
@pytest.mark.parametrize("mnk", SHAPES)
def test_matmul(fn, mnk):
    M, N, K = mnk
    m = load("c4_gemm")
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    ref = (A.double() @ B.double()).float()    # fp64 reference: no TF32 surprises
    torch.testing.assert_close(getattr(m, fn)(A, B), ref, rtol=1e-3, atol=1e-3)
