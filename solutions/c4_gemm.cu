// C4 · GEMM: naive -> shared-memory tiling -> register tiling — reference solution.
//
// C[M, N] = A[M, K] @ B[K, N], all row-major float32, fp32 accumulation.
//
// Arithmetic intensity is what changes between versions:
//   naive   : every multiply-add loads one A and one B value from global memory (2 loads / FMA).
//   smem    : a 32x32 block loads a tile once into shared memory and reuses each value 32 times.
//   regtile : each thread computes a 4x4 patch of C from registers; every value read from
//             shared memory is used 4 times, so shared-memory traffic drops by 4x again.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// ---------------------------------------------------------------- v0: naive
__global__ void matmul_naive_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // threadIdx.x -> column: B and C reads coalesce
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[(int64_t)row * K + k] * B[(int64_t)k * N + col];
  C[(int64_t)row * N + col] = acc;
}

// ---------------------------------------------------------------- v1: shared-memory tiling
constexpr int T = 32;  // block = T x T threads = one T x T tile of C

__global__ void matmul_smem_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * T + ty;
  const int col = blockIdx.x * T + tx;

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += T) {
    // Every thread loads one element of each tile. Out-of-range elements become 0, which
    // contributes nothing to the dot product, so the inner loop needs no bounds checks.
    As[ty][tx] = (row < M && k0 + tx < K) ? A[(int64_t)row * K + k0 + tx] : 0.0f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(int64_t)(k0 + ty) * N + col] : 0.0f;
    __syncthreads();  // tiles fully loaded before anyone uses them

#pragma unroll
    for (int k = 0; k < T; ++k) {
      // As[ty][k]: same address for the whole warp (same ty) -> broadcast, no conflict.
      // Bs[k][tx]: 32 consecutive floats -> 32 different banks, no conflict.
      acc += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();  // everyone done with this tile before the next iteration overwrites it
  }
  if (row < M && col < N) C[(int64_t)row * N + col] = acc;
}

// ---------------------------------------------------------------- v2: register tiling
// Block tile BM x BN of C, K in steps of BK, each thread owns a TM x TN patch.
constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 16 * 16 = 256

__global__ void __launch_bounds__(THREADS)
matmul_regtile_kernel(const float* __restrict__ A, const float* __restrict__ B,
                      float* __restrict__ C, int M, int N, int K) {
  // A's tile is stored transposed (As[k][m]) so the inner loop reads TM consecutive m values
  // for a fixed k; B's tile keeps its natural layout Bs[k][n].
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int tx = tid % (BN / TN);  // which TN-wide column strip of the block tile
  const int ty = tid / (BN / TN);  // which TM-tall row strip
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  float acc[TM][TN] = {};  // the thread's TM x TN results live in registers
  float a_frag[TM], b_frag[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Cooperative load of A[block_row : +BM, k0 : +BK] (BM*BK = 1024 floats, 4 per thread).
    // e -> (m, k) with k fastest, so consecutive threads read consecutive addresses of A.
    for (int e = tid; e < BM * BK; e += THREADS) {
      const int m = e / BK, k = e % BK;
      const int gm = block_row + m, gk = k0 + k;
      As[k][m] = (gm < M && gk < K) ? A[(int64_t)gm * K + gk] : 0.0f;
    }
    // Cooperative load of B[k0 : +BK, block_col : +BN], n fastest -> coalesced.
    for (int e = tid; e < BK * BN; e += THREADS) {
      const int k = e / BN, n = e % BN;
      const int gk = k0 + k, gn = block_col + n;
      Bs[k][n] = (gk < K && gn < N) ? B[(int64_t)gk * N + gn] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // Load a column of A's tile and a row of B's tile into registers ...
#pragma unroll
      for (int i = 0; i < TM; ++i) a_frag[i] = As[k][ty * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) b_frag[j] = Bs[k][tx * TN + j];
      // ... and take their outer product: TM*TN FMAs for TM+TN shared-memory reads.
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a_frag[i] * b_frag[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int gm = block_row + ty * TM + i;
    if (gm >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gn = block_col + tx * TN + j;
      if (gn < N) C[(int64_t)gm * N + gn] = acc[i][j];
    }
  }
}

// ---------------------------------------------------------------- host wrappers
static void check_inputs(const torch::Tensor& A, const torch::Tensor& B) {
  for (const auto* t : {&A, &B}) {
    TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kFloat,
                "inputs must be contiguous float32 CUDA tensors");
    TORCH_CHECK(t->dim() == 2, "inputs must be 2-D");
  }
  TORCH_CHECK(A.size(1) == B.size(0), "inner dimensions must match");
}

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
  check_inputs(A, B);
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  auto C = torch::empty({M, N}, A.options());
  if (M == 0 || N == 0) return C;
  dim3 block(32, 8);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  matmul_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}

torch::Tensor matmul_smem(torch::Tensor A, torch::Tensor B) {
  check_inputs(A, B);
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  auto C = torch::empty({M, N}, A.options());
  if (M == 0 || N == 0) return C;
  dim3 block(T, T);  // 1024 threads: the per-block maximum
  dim3 grid((N + T - 1) / T, (M + T - 1) / T);
  matmul_smem_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}

torch::Tensor matmul_regtile(torch::Tensor A, torch::Tensor B) {
  check_inputs(A, B);
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  auto C = torch::empty({M, N}, A.options());
  if (M == 0 || N == 0) return C;
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  matmul_regtile_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}
