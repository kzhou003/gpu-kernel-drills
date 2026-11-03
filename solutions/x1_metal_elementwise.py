"""X1 · First custom Metal kernel in MLX: fused add + GELU — reference solution.

mx.fast.metal_kernel(name, input_names, output_names, source, header) compiles the BODY of a Metal
kernel: MLX writes the signature for you. Inputs arrive as `const device T* x` (or `constant`
scalars), outputs as `device T* out`, and the Metal attributes you mention in the body
(thread_position_in_grid, threadgroup_position_in_grid, ...) are added as parameters.

Grid semantics differ from CUDA: `grid` is the TOTAL number of threads per dimension (Metal's
dispatchThreads), not the number of threadgroups, and it need not be a multiple of `threadgroup`.
"""
import mlx.core as mx

_HEADER = """
inline float gelu_tanh(float z) {
    // 0.5 z (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3)))
    return 0.5f * z * (1.0f + metal::precise::tanh(0.7978845608028654f * (z + 0.044715f * z * z * z)));
}
"""

# One element per thread. T is a template parameter, so one source serves fp32 / fp16 / bf16.
_SRC = """
    uint i = thread_position_in_grid.x;
    if (i < n) {
        out[i] = T(gelu_tanh(float(x[i]) + float(y[i])));   // math in fp32, store in T
    }
"""

# N_PER elements per thread, contiguous: fewer threads, each issuing wider loads.
_SRC_VEC = """
    uint base = thread_position_in_grid.x * N_PER;
    for (uint j = 0; j < N_PER; ++j) {
        uint i = base + j;
        if (i < n) out[i] = T(gelu_tanh(float(x[i]) + float(y[i])));
    }
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
