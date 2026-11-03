// C4 · GEMM: naive -> shared-memory tiling -> register tiling — YOUR IMPLEMENTATION.
// matmul_naive is given as a worked example. Implement the two TODO kernels.
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
  // TODO: loop over K in steps of T: load one A tile and one B tile (zero-fill out of range),
  //       __syncthreads(), accumulate T products, __syncthreads(). Store C if in range.
}

// ---------------------------------------------------------------- v2: register tiling
// Block tile BM x BN of C, K in steps of BK, each thread owns a TM x TN patch.
constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 16 * 16 = 256

__global__ void __launch_bounds__(THREADS)
matmul_regtile_kernel(const float* __restrict__ A, const float* __restrict__ B,
                      float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BK][BM];  // A tile stored transposed: As[k][m]
  __shared__ float Bs[BK][BN];
  // TODO: tid -> (tx, ty); float acc[TM][TN] = {};
  //       for each K tile: cooperative loads (each thread loads BM*BK/THREADS values of A and of B),
  //       __syncthreads(), for k in BK: load a_frag[TM], b_frag[TN] from shared, outer product into acc,
  //       __syncthreads(). Finally store the TM x TN patch with bounds checks.
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
