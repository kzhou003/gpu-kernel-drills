import pytest

mx = pytest.importorskip("mlx.core", reason="X track needs MLX on an Apple-silicon Mac")
from drill import load  # noqa: E402

SHAPES = [(1, 1, 1), (17, 33, 9), (32, 32, 32), (64, 64, 64), (100, 70, 130), (257, 129, 1025)]


@pytest.mark.parametrize("fn", ["matmul_tiled", "matmul_simd"])
@pytest.mark.parametrize("mnk", SHAPES)
def test_matmul(fn, mnk):
    M, N, K = mnk
    m = load("x3_metal_matmul")
    a, b = mx.random.normal((M, K)), mx.random.normal((K, N))
    out = getattr(m, fn)(a, b)
    assert out.shape == (M, N)
    assert mx.allclose(out, a @ b, atol=1e-3, rtol=1e-3).item()


def test_matmul_fp16():
    m = load("x3_metal_matmul")
    a, b = mx.random.normal((96, 64)).astype(mx.float16), mx.random.normal((64, 80)).astype(mx.float16)
    for fn in ("matmul_tiled", "matmul_simd"):
        out = getattr(m, fn)(a, b)
        assert out.dtype == mx.float16
        assert mx.allclose(out.astype(mx.float32), a.astype(mx.float32) @ b.astype(mx.float32), atol=5e-2, rtol=1e-2).item()
