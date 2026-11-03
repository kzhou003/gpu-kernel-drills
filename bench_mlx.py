"""MLX benchmarks for the X track (Apple silicon):  .venv-mlx/bin/python bench_mlx.py [x1|x2|x3|all]"""
import sys
import time

import mlx.core as mx

from drill import load


def bench(fn, iters=50):
    for _ in range(5):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t) / iters * 1e3   # ms (includes per-op dispatch overhead)


def row(name, ms, gbytes=None, gflop=None):
    extra = f"{gbytes / (ms * 1e-3):8.1f} GB/s" if gbytes else (f"{gflop / (ms * 1e-3) / 1e3:8.2f} TFLOP/s" if gflop else "")
    print(f"  {name:<28} {ms * 1e3:9.1f} us  {extra}")


def b_x1():
    m = load("x1_metal_elementwise")
    n = 1 << 24
    x, y = mx.random.normal((n,)).astype(mx.float16), mx.random.normal((n,)).astype(mx.float16)
    gb = 3 * n * 2 / 1e9
    print(f"add_gelu fp16 n={n}")
    row("mlx ops (nn.gelu_approx)", bench(lambda: mx.compile(lambda a, b: 0.5 * (a + b) * (1 + mx.tanh(0.7978845608 * ((a + b) + 0.044715 * (a + b) ** 3))))(x, y)), gb)
    row("yours (1 elem/thread)", bench(lambda: m.add_gelu(x, y)), gb)
    row("yours (4 elems/thread)", bench(lambda: m.add_gelu_vec(x, y)), gb)


def b_x2():
    m = load("x2_metal_softmax")
    for N in (1024, 4096, 32000):
        x = mx.random.normal((2048, N)).astype(mx.float16)
        gb = 2 * x.size * 2 / 1e9
        print(f"softmax fp16 2048x{N}")
        row("mx.softmax", bench(lambda: mx.softmax(x, axis=-1)), gb)
        row("yours", bench(lambda: m.softmax(x)), gb)


def b_x3():
    m = load("x3_metal_matmul")
    for n in (512, 1024, 2048):
        a, b = mx.random.normal((n, n)), mx.random.normal((n, n))
        gf = 2 * n ** 3 / 1e9
        print(f"fp32 matmul {n}^3")
        row("mx.matmul (steel)", bench(lambda: a @ b), gflop=gf)
        row("yours tiled", bench(lambda: m.matmul_tiled(a, b)), gflop=gf)
        row("yours simdgroup_matrix", bench(lambda: m.matmul_simd(a, b)), gflop=gf)


B = {"x1": b_x1, "x2": b_x2, "x3": b_x3}
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(mx.device_info()["device_name"], "| mlx", mx.__version__)
    for k in B if which == "all" else [which]:
        print("==", k)
        B[k]()
