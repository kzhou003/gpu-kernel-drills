"""Line-by-line NumPy simulation of the index math in c3 transpose_tiled and c4 matmul_regtile/smem."""
import numpy as np
rng = np.random.default_rng(0)

def transpose_tiled(x):
    M, N = x.shape; TILE, ROWS = 32, 8
    y = np.full((N, M), np.nan, np.float32)
    for by in range((M + TILE - 1) // TILE):
        for bx in range((N + TILE - 1) // TILE):
            tile = np.zeros((TILE, TILE + 1), np.float32)
            for ty in range(ROWS):
                for tx in range(TILE):
                    c = bx * TILE + tx; r0 = by * TILE + ty
                    for j in range(0, TILE, ROWS):
                        r = r0 + j
                        if r < M and c < N: tile[ty + j][tx] = x[r, c]
            for ty in range(ROWS):
                for tx in range(TILE):
                    yc = by * TILE + tx; yr0 = bx * TILE + ty
                    for j in range(0, TILE, ROWS):
                        yr = yr0 + j
                        if yr < N and yc < M: y[yr, yc] = tile[tx][ty + j]
    return y

def regtile(A, B):
    BM, BN, BK, TM, TN = 64, 64, 16, 4, 4; THREADS = (BM // TM) * (BN // TN)
    M, K = A.shape; N = B.shape[1]
    C = np.full((M, N), np.nan, np.float64)
    for by in range((M + BM - 1) // BM):
        for bx in range((N + BN - 1) // BN):
            acc = np.zeros((THREADS, TM, TN))
            block_row, block_col = by * BM, bx * BN
            for k0 in range(0, K, BK):
                As = np.zeros((BK, BM)); Bs = np.zeros((BK, BN))
                for tid in range(THREADS):
                    for e in range(tid, BM * BK, THREADS):
                        m, k = e // BK, e % BK; gm, gk = block_row + m, k0 + k
                        As[k][m] = A[gm, gk] if (gm < M and gk < K) else 0
                    for e in range(tid, BK * BN, THREADS):
                        k, n = e // BN, e % BN; gk, gn = k0 + k, block_col + n
                        Bs[k][n] = B[gk, gn] if (gk < K and gn < N) else 0
                for tid in range(THREADS):
                    tx, ty = tid % (BN // TN), tid // (BN // TN)
                    for k in range(BK):
                        a = As[k][ty * TM: ty * TM + TM]; b = Bs[k][tx * TN: tx * TN + TN]
                        acc[tid] += np.outer(a, b)
            for tid in range(THREADS):
                tx, ty = tid % (BN // TN), tid // (BN // TN)
                for i in range(TM):
                    gm = block_row + ty * TM + i
                    if gm >= M: continue
                    for j in range(TN):
                        gn = block_col + tx * TN + j
                        if gn < N: C[gm, gn] = acc[tid][i][j]
    return C

def smem(A, B):
    T = 32; M, K = A.shape; N = B.shape[1]; C = np.full((M, N), np.nan)
    for by in range((M + T - 1) // T):
        for bx in range((N + T - 1) // T):
            acc = np.zeros((T, T))
            for k0 in range(0, K, T):
                As = np.zeros((T, T)); Bs = np.zeros((T, T))
                for ty in range(T):
                    for tx in range(T):
                        row, col = by * T + ty, bx * T + tx
                        As[ty][tx] = A[row, k0 + tx] if (row < M and k0 + tx < K) else 0
                        Bs[ty][tx] = B[k0 + ty, col] if (k0 + ty < K and col < N) else 0
                acc += As @ Bs   # == per-thread sum_k As[ty][k] * Bs[k][tx]
            for ty in range(T):
                for tx in range(T):
                    row, col = by * T + ty, bx * T + tx
                    if row < M and col < N: C[row, col] = acc[ty][tx]
    return C

for shape in [(1, 1), (1, 70), (70, 1), (33, 31), (65, 97)]:
    x = rng.standard_normal(shape).astype(np.float32)
    assert np.array_equal(transpose_tiled(x), x.T), shape
print("transpose_tiled index math OK")
for M, N, K in [(1, 1, 1), (17, 33, 9), (65, 70, 33), (130, 64, 17)]:
    A = rng.standard_normal((M, K)); B = rng.standard_normal((K, N))
    assert np.allclose(regtile(A, B), A @ B), (M, N, K)
    assert np.allclose(smem(A, B), A @ B), (M, N, K)
print("matmul_regtile / matmul_smem index math OK")
