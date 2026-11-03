"""X3 · Matmul in Metal: threadgroup-memory tiling, then simdgroup_matrix (8x8 hardware MMA) — reference solution.

C[M, N] = A[M, K] @ B[K, N], row-major, fp32 accumulation.

matmul_tiled : C4's v1 on Apple GPUs. 16x16 threadgroup computes a 16x16 tile of C; A and B tiles go
               through threadgroup memory, every value is reused 16 times.
matmul_simd  : 128 threads = 4 simdgroups compute a 32x32 tile; each simdgroup owns a 16x16 quarter
               as 2x2 simdgroup_float8x8 accumulators and uses simdgroup_multiply_accumulate,
               the Apple-GPU matrix instruction (what MLX's own "steel" GEMM kernels are built on).
"""
import mlx.core as mx

_TILED = """
    const uint TS = 16;
    uint2 t = thread_position_in_threadgroup.xy;
    uint2 g = threadgroup_position_in_grid.xy;
    uint row = g.y * TS + t.y;
    uint col = g.x * TS + t.x;
    threadgroup float As[16][16];
    threadgroup float Bs[16][16];
    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += TS) {
        As[t.y][t.x] = (row < M && k0 + t.x < K) ? float(A[row * K + k0 + t.x]) : 0.0f;   // zero-fill edges
        Bs[t.y][t.x] = (k0 + t.y < K && col < N) ? float(B[(k0 + t.y) * N + col]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TS; ++k) acc += As[t.y][k] * Bs[k][t.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = T(acc);
"""

_SIMD = """
    // Tile: BM = BN = 32, BK = 16. 128 threads = 4 simdgroups arranged 2 x 2.
    const uint BM = 32, BN = 32, BK = 16;
    uint tid = thread_index_in_threadgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint sg_r = sg / 2, sg_c = sg % 2;                 // this simdgroup's 16x16 quarter
    uint2 g = threadgroup_position_in_grid.xy;
    uint row0 = g.y * BM, col0 = g.x * BN;

    threadgroup float As[32 * 16];                     // [BM][BK]
    threadgroup float Bs[16 * 32];                     // [BK][BN]
    threadgroup float Cs[32 * 32];                     // staging for the masked store

    simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 a0, a1, b0, b1;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += 128) {    // 512 values each, 4 per thread
            uint r = e / BK, k = e % BK;
            As[e] = (row0 + r < M && k0 + k < K) ? float(A[(row0 + r) * K + k0 + k]) : 0.0f;
            uint kb = e / BN, c = e % BN;
            Bs[e] = (k0 + kb < K && col0 + c < N) ? float(B[(k0 + kb) * N + col0 + c]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            // simdgroup_load(matrix, pointer to the 8x8 block's first element, row stride in elements)
            simdgroup_load(a0, As + (sg_r * 16 + 0) * BK + kk, BK);
            simdgroup_load(a1, As + (sg_r * 16 + 8) * BK + kk, BK);
            simdgroup_load(b0, Bs + kk * BN + sg_c * 16 + 0, BN);
            simdgroup_load(b1, Bs + kk * BN + sg_c * 16 + 8, BN);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);   // c += a @ b, 8x8x8
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, Cs + (sg_r * 16 + 0) * BN + sg_c * 16 + 0, BN);
    simdgroup_store(c01, Cs + (sg_r * 16 + 0) * BN + sg_c * 16 + 8, BN);
    simdgroup_store(c10, Cs + (sg_r * 16 + 8) * BN + sg_c * 16 + 0, BN);
    simdgroup_store(c11, Cs + (sg_r * 16 + 8) * BN + sg_c * 16 + 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128) {        // masked write-back for partial edge tiles
        uint r = e / BN, c = e % BN;
        if (row0 + r < M && col0 + c < N) C[(row0 + r) * N + col0 + c] = T(Cs[e]);
    }
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
