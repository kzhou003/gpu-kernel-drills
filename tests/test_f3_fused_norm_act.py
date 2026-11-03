import pytest
import torch
import torch.nn.functional as F

from common.env import DEVICE, HAS_FLASHINFER
from drill import load


def ref_fused_add_rmsnorm(x, r, w, eps):
    h = x.float() + r.float()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype), h.to(r.dtype)


@pytest.mark.parametrize("shape", [(1, 64), (7, 333), (4, 3, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fused_add_rmsnorm_in_place(shape, dtype):
    m = load("f3_fused_norm_act")
    x = torch.randn(shape, device=DEVICE).to(dtype)
    r = torch.randn(shape, device=DEVICE).to(dtype)
    w = torch.randn(shape[-1], device=DEVICE).to(dtype)
    ref_x, ref_r = ref_fused_add_rmsnorm(x, r, w, 1e-6)
    x_ptr, r_ptr = x.data_ptr(), r.data_ptr()
    assert m.fused_add_rmsnorm(x, r, w, 1e-6) is None
    assert x.data_ptr() == x_ptr and r.data_ptr() == r_ptr, "must write in place"
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(r, ref_r, rtol=tol, atol=tol)
    torch.testing.assert_close(x, ref_x, rtol=tol, atol=tol)


@pytest.mark.parametrize("shape", [(1, 2), (5, 2 * 333), (2, 3, 2 * 4096)])
def test_silu_and_mul(shape):
    m = load("f3_fused_norm_act")
    x = torch.randn(shape, device=DEVICE)
    d = shape[-1] // 2
    out = m.silu_and_mul(x)
    assert out.shape == (*shape[:-1], d)
    torch.testing.assert_close(out, F.silu(x[..., :d]) * x[..., d:], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAS_FLASHINFER, reason="needs a GPU with flashinfer installed")
def test_matches_flashinfer():
    import flashinfer
    m = load("f3_fused_norm_act")
    x, r = (torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    x1, r1, x2, r2 = x.clone(), r.clone(), x.clone(), r.clone()
    m.fused_add_rmsnorm(x1, r1, w)
    flashinfer.fused_add_rmsnorm(x2, r2, w)
    torch.testing.assert_close(r1, r2)
    torch.testing.assert_close(x1.float(), x2.float(), rtol=2e-2, atol=2e-2)
    y = torch.randn(64, 2 * 11008, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(m.silu_and_mul(y).float(), flashinfer.silu_and_mul(y).float(), rtol=2e-2, atol=2e-2)
