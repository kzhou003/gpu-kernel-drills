import pytest
import torch

from common.env import BF16_OK, DEVICE, HAS_CUDA
from drill import load

FIXED = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}


@pytest.mark.parametrize("mnk", [(16, 16, 16), (64, 64, 64), (100, 70, 130), (257, 129, 96)])
def test_matmul_fixed_config(mnk):
    M, N, K = mnk
    m = load("t4_matmul")
    a = torch.randn(M, K, device=DEVICE).half()
    b = torch.randn(K, N, device=DEVICE).half()
    out = m.matmul(a, b, config=FIXED)
    assert out.dtype == torch.float16 and out.shape == (M, N)
    torch.testing.assert_close(out.float(), a.float() @ b.float(), rtol=1e-2, atol=5e-2)


def test_matmul_transposed_input():
    m = load("t4_matmul")
    a = torch.randn(96, 80, device=DEVICE).half().t()     # [80, 96], strides (1, 80)
    b = torch.randn(96, 72, device=DEVICE).half()
    torch.testing.assert_close(m.matmul(a, b, config=FIXED).float(), a.float() @ b.float(), rtol=1e-2, atol=5e-2)


def test_grouped_order_covers_every_tile_once():
    m = load("t4_matmul")
    # 5 x 3 tiles with GROUP_M=2: if the pid -> (pid_m, pid_n) mapping skipped or repeated a
    # tile, some output tile would be left unwritten (garbage from torch.empty) and the check fails.
    cfg = {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16, "GROUP_M": 2}
    a = torch.randn(80, 32, device=DEVICE).half()
    b = torch.randn(32, 48, device=DEVICE).half()
    out = m.matmul(a, b, config=cfg)
    torch.testing.assert_close(out.float(), a.float() @ b.float(), rtol=1e-2, atol=5e-2)


@pytest.mark.skipif(not HAS_CUDA, reason="autotuning needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_autotuned(dtype):
    if dtype == torch.bfloat16 and not BF16_OK:
        pytest.skip("bf16 needs sm_80+")
    m = load("t4_matmul")
    a = torch.randn(1000, 777, device=DEVICE, dtype=dtype)
    b = torch.randn(777, 1500, device=DEVICE, dtype=dtype)
    torch.testing.assert_close(m.matmul(a, b).float(), (a.float() @ b.float()), rtol=2e-2, atol=1e-1)
