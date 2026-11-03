import pytest

mx = pytest.importorskip("mlx.core", reason="X track needs MLX on an Apple-silicon Mac")
from drill import load  # noqa: E402


@pytest.mark.parametrize("shape", [(1, 1), (4, 7), (33, 128), (17, 1000), (8, 4097), (2, 32000)])
def test_softmax(shape):
    m = load("x2_metal_softmax")
    x = mx.random.normal(shape) * 5
    assert mx.allclose(m.softmax(x), mx.softmax(x, axis=-1), atol=1e-6, rtol=1e-5).item()


def test_softmax_large_values():
    m = load("x2_metal_softmax")
    x = mx.array([[1000.0, 1001.0, 1002.0], [-1e4, 0.0, 1e4]])
    out = m.softmax(x)
    assert mx.all(mx.isfinite(out)).item()
    assert mx.allclose(out, mx.softmax(x, axis=-1)).item()


def test_softmax_fp16():
    m = load("x2_metal_softmax")
    x = mx.random.normal((10, 257)).astype(mx.float16)
    out = m.softmax(x)
    assert out.dtype == mx.float16
    assert mx.allclose(out.astype(mx.float32), mx.softmax(x.astype(mx.float32), axis=-1), atol=1e-3).item()


@pytest.mark.parametrize("shape", [(1, 16), (7, 333), (32, 4096)])
def test_rms_norm(shape):
    m = load("x2_metal_softmax")
    x, w = mx.random.normal(shape), mx.random.normal((shape[1],))
    ref = mx.fast.rms_norm(x, w, 1e-6)
    assert mx.allclose(m.rms_norm(x, w, 1e-6), ref, atol=1e-4, rtol=1e-4).item()
