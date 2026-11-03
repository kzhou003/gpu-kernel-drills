"""X2 · row softmax + RMSNorm with simdgroup reductions — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
import mlx.core as mx

# Block-wide reduction helper: every thread returns the reduced value. `scratch` holds one
# slot per simdgroup; the trailing barrier lets the caller reuse `scratch` right away.
_HEADER = """
    // TODO: template <bool IS_MAX> inline float tg_reduce(float v, threadgroup float* scratch, uint lane, uint sg, uint nsg)
    // TODO: simd_max/simd_sum -> lane 0 writes scratch[sg] -> barrier -> reduce the nsg partials -> barrier
"""

_SOFTMAX = """
    // TODO: row = threadgroup_position_in_grid.x; threads stride over the row; use `auto xr = x + row * n` (address space!)
    // TODO: pass 1 max, pass 2 sum exp(x - max), pass 3 write
"""

_RMSNORM = """
    // TODO: sum of squares -> tg_reduce -> rstd = rsqrt(ss / n + eps) -> out = x * rstd * w
"""

_softmax = mx.fast.metal_kernel(name="row_softmax", input_names=["x", "ncols"], output_names=["out"],
                                source=_SOFTMAX, header=_HEADER)
_rmsnorm = mx.fast.metal_kernel(name="row_rmsnorm", input_names=["x", "w", "ncols", "eps"], output_names=["out"],
                                source=_RMSNORM, header=_HEADER)


def _threads(ncols: int) -> int:
    # one simdgroup per 32..1024 columns, at most 1024 threads per threadgroup
    return min(1024, max(32, 1 << (max(ncols, 1) - 1).bit_length())) if ncols < 1024 else 1024


def softmax(x: mx.array) -> mx.array:
    assert x.ndim == 2
    rows, ncols = x.shape
    t = _threads(ncols)
    return _softmax(inputs=[x, ncols], template=[("T", x.dtype)], grid=(rows * t, 1, 1), threadgroup=(t, 1, 1),
                    output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def rms_norm(x: mx.array, w: mx.array, eps: float = 1e-6) -> mx.array:
    assert x.ndim == 2 and w.shape == (x.shape[1],)
    rows, ncols = x.shape
    t = _threads(ncols)
    return _rmsnorm(inputs=[x, w, ncols, eps], template=[("T", x.dtype)], grid=(rows * t, 1, 1),
                    threadgroup=(t, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
