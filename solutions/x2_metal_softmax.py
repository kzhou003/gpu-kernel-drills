"""X2 · Row softmax and RMSNorm in Metal: simdgroup reductions + threadgroup memory — reference solution.

Apple GPU vocabulary <-> CUDA:
    threadgroup            <-> block           threadgroup memory     <-> shared memory
    simdgroup (32 threads) <-> warp            simd_sum / simd_max    <-> warp shuffle reductions
    threadgroup_barrier(mem_flags::mem_threadgroup)                   <-> __syncthreads()
One threadgroup per row: each thread strides over the row, simd_* reduces inside a simdgroup,
then lane 0 of each simdgroup posts its partial to threadgroup memory for the final reduction.
"""
import mlx.core as mx

# Block-wide reduction helper: every thread returns the reduced value. `scratch` holds one
# slot per simdgroup; the trailing barrier lets the caller reuse `scratch` right away.
_HEADER = """
template <bool IS_MAX>
inline float tg_reduce(float v, threadgroup float* scratch, uint lane, uint sg, uint nsg) {
    v = IS_MAX ? simd_max(v) : simd_sum(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float init = IS_MAX ? -INFINITY : 0.0f;
    v = lane < nsg ? scratch[lane] : init;
    v = IS_MAX ? simd_max(v) : simd_sum(v);       // every simdgroup reduces the partials itself
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}
"""

_SOFTMAX = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint nthr = threads_per_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint nsg = (nthr + 31) / 32;
    threadgroup float scratch[32];
    uint n = uint(ncols);
    // `auto`, not `const device T*`: MLX may pass small inputs in the `constant` address space.
    auto xr = x + row * n;

    float m = -INFINITY;                                   // pass 1: row max
    for (uint c = tid; c < n; c += nthr) m = max(m, float(xr[c]));
    m = tg_reduce<true>(m, scratch, lane, sg, nsg);

    float s = 0.0f;                                        // pass 2: sum of exp(x - max)
    for (uint c = tid; c < n; c += nthr) s += metal::exp(float(xr[c]) - m);
    s = tg_reduce<false>(s, scratch, lane, sg, nsg);

    float inv = 1.0f / s;                                  // pass 3: normalise and write
    for (uint c = tid; c < n; c += nthr) out[row * n + c] = T(metal::exp(float(xr[c]) - m) * inv);
"""

_RMSNORM = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint nthr = threads_per_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint nsg = (nthr + 31) / 32;
    threadgroup float scratch[32];
    uint n = uint(ncols);
    // `auto`, not `const device T*`: MLX may pass small inputs in the `constant` address space.
    auto xr = x + row * n;

    float ss = 0.0f;
    for (uint c = tid; c < n; c += nthr) { float v = float(xr[c]); ss += v * v; }
    ss = tg_reduce<false>(ss, scratch, lane, sg, nsg);
    float rstd = metal::precise::rsqrt(ss / float(n) + eps);
    for (uint c = tid; c < n; c += nthr) out[row * n + c] = T(float(xr[c]) * rstd * float(w[c]));
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
