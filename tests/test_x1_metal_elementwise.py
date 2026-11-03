import math

import pytest

mx = pytest.importorskip("mlx.core", reason="X track needs MLX on an Apple-silicon Mac")
from drill import load  # noqa: E402


def ref(x, y):
    z = (x + y).astype(mx.float32)
    return 0.5 * z * (1 + mx.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))


@pytest.mark.parametrize("n", [1, 255, 256, 257, 1_000_003])
def test_add_gelu_fp32(n):
    m = load("x1_metal_elementwise")
    x, y = mx.random.normal((n,)) * 3, mx.random.normal((n,))
    assert mx.allclose(m.add_gelu(x, y), ref(x, y), atol=1e-5, rtol=1e-5).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_add_gelu_half_2d(dtype):
    m = load("x1_metal_elementwise")
    x, y = mx.random.normal((7, 333)).astype(dtype), mx.random.normal((7, 333)).astype(dtype)
    out = m.add_gelu(x, y)
    assert out.dtype == dtype and out.shape == (7, 333)
    assert mx.allclose(out.astype(mx.float32), ref(x, y), atol=2e-2, rtol=2e-2).item()


def test_transposed_input():
    m = load("x1_metal_elementwise")
    x, y = mx.random.normal((48, 64)).T, mx.random.normal((64, 48))
    assert mx.allclose(m.add_gelu(x, y), ref(x, y), atol=1e-5).item()


@pytest.mark.parametrize("n", [1, 5, 1023, 4096, 100_001])
def test_add_gelu_vec(n):
    m = load("x1_metal_elementwise")
    x, y = mx.random.normal((n,)), mx.random.normal((n,))
    assert mx.allclose(m.add_gelu_vec(x, y, n_per=4), ref(x, y), atol=1e-5).item()
