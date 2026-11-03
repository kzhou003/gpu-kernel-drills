"""X3 · Metal matmul: threadgroup tiling + simdgroup_matrix — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
import mlx.core as mx

_TILED = """
    // TODO: 16x16 threadgroup, one C element per thread; As/Bs [16][16] in threadgroup memory, zero-filled edges
    // TODO: K loop: load tiles, barrier, 16 FMAs, barrier; masked store
"""

_SIMD = """
    // TODO: 128 threads = 4 simdgroups (2x2) on a 32x32 C tile, BK = 16
    // TODO: stage A [32x16] and B [16x32] in threadgroup memory; each simdgroup: 2x2 simdgroup_float8x8 accumulators
    // TODO: simdgroup_load / simdgroup_multiply_accumulate; simdgroup_store into Cs, barrier, masked copy to C
"""

_tiled = mx.fast.metal_kernel(name="matmul_tiled", input_names=["A", "B", "M", "N", "K"], output_names=["C"], source=_TILED)
_simd = mx.fast.metal_kernel(name="matmul_simd", input_names=["A", "B", "M", "N", "K"], output_names=["C"], source=_SIMD,
                             header="#include <metal_simdgroup_matrix>\n")


def _check(a, b):
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0] and a.dtype == b.dtype
    return a.shape[0], b.shape[1], a.shape[1]


def matmul_tiled(a: mx.array, b: mx.array) -> mx.array:
    M, N, K = _check(a, b)
    grid = (((N + 15) // 16) * 16, ((M + 15) // 16) * 16, 1)       # threads, not threadgroups
    return _tiled(inputs=[a, b, M, N, K], template=[("T", a.dtype)], grid=grid, threadgroup=(16, 16, 1),
                  output_shapes=[(M, N)], output_dtypes=[a.dtype])[0]


def matmul_simd(a: mx.array, b: mx.array) -> mx.array:
    M, N, K = _check(a, b)
    # one 128-thread threadgroup per 32x32 tile: grid x = 128 * tiles_n, grid y = tiles_m
    grid = (((N + 31) // 32) * 128, (M + 31) // 32, 1)
    return _simd(inputs=[a, b, M, N, K], template=[("T", a.dtype)], grid=grid, threadgroup=(128, 1, 1),
                 output_shapes=[(M, N)], output_dtypes=[a.dtype])[0]
