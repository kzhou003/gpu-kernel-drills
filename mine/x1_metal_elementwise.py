"""X1 · fused add + GELU as an MLX custom Metal kernel — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
import mlx.core as mx

_HEADER = """
    // TODO: write `inline float gelu_tanh(float z)` (metal::precise::tanh)
inline float gelu_tanh(float z) { return z; }
"""

# One element per thread. T is a template parameter, so one source serves fp32 / fp16 / bf16.
_SRC = """
    // TODO: i = thread_position_in_grid.x; bounds check against n; out[i] = T(gelu_tanh(float(x[i]) + float(y[i])))
"""

# N_PER elements per thread, contiguous: fewer threads, each issuing wider loads.
_SRC_VEC = """
    // TODO: each thread handles N_PER consecutive elements starting at thread_position_in_grid.x * N_PER
"""

_kernel = mx.fast.metal_kernel(name="add_gelu", input_names=["x", "y", "n"], output_names=["out"],
                               source=_SRC, header=_HEADER)
_kernel_vec = mx.fast.metal_kernel(name="add_gelu_vec", input_names=["x", "y", "n"], output_names=["out"],
                                   source=_SRC_VEC, header=_HEADER)


def add_gelu(x: mx.array, y: mx.array, threadgroup: int = 256) -> mx.array:
    assert x.shape == y.shape and x.dtype == y.dtype
    n = x.size
    # ensure_row_contiguous (the default) makes MLX hand the kernel contiguous copies when needed.
    return _kernel(inputs=[x, y, n], template=[("T", x.dtype)],
                   grid=(max(n, 1), 1, 1), threadgroup=(threadgroup, 1, 1),
                   output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def add_gelu_vec(x: mx.array, y: mx.array, n_per: int = 4, threadgroup: int = 256) -> mx.array:
    assert x.shape == y.shape and x.dtype == y.dtype
    n = x.size
    threads = max((n + n_per - 1) // n_per, 1)
    return _kernel_vec(inputs=[x, y, n], template=[("T", x.dtype), ("N_PER", n_per)],
                       grid=(threads, 1, 1), threadgroup=(threadgroup, 1, 1),
                       output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
