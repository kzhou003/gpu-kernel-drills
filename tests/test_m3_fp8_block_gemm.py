import pytest
import torch

from common.env import HAS_CUDA
from drill import load

FP8_OK = HAS_CUDA and torch.cuda.get_device_capability() >= (8, 9)
pytestmark = pytest.mark.skipif(not FP8_OK, reason="FP8 tensor cores need sm_89+ (Ada / Hopper / Blackwell)")


def dequant_tokens(q, s, group=128):
    M, K = q.shape
    return (q.float().view(M, -1, group) * s[:, :, None]).view(M, K)


@pytest.mark.parametrize("M,K", [(1, 128), (33, 512), (256, 7168)])
def test_per_token_group_quant(M, K):
    m = load("m3_fp8_block_gemm")
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 3
    q, s = m.per_token_group_quant_fp8(x)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (M, K // 128) and s.dtype == torch.float32
    # each group's largest |value| maps to (almost exactly) 448
    assert torch.allclose(q.float().view(M, -1, 128).abs().amax(-1), torch.full_like(s, 448.0), rtol=1e-2)
    torch.testing.assert_close(dequant_tokens(q, s), x.float(), rtol=7e-2, atol=7e-2 * s.max().item() * 32)


@pytest.mark.parametrize("M,N,K", [(16, 128, 256), (100, 384, 1024), (512, 1024, 2048)])
def test_fp8_block_gemm(M, N, K):
    m = load("m3_fp8_block_gemm")
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K ** 0.5
    aq, as_ = m.per_token_group_quant_fp8(a)
    bq, bs = m.block_quant_fp8(b)
    c = m.fp8_block_gemm(aq, as_, bq, bs)
    # exact reference for the quantised operands: dequantise, then matmul in fp32
    b_deq = (bq.float().view(N // 128, 128, K // 128, 128) * bs[:, None, :, None]).view(N, K)
    ref_q = dequant_tokens(aq, as_) @ b_deq.t()
    torch.testing.assert_close(c.float(), ref_q, rtol=2e-2, atol=2e-2)
    # and the quantisation error against the bf16 matmul stays small
    rel = (c.float() - a.float() @ b.float().t()).norm() / (a.float() @ b.float().t()).norm()
    assert rel < 0.05
