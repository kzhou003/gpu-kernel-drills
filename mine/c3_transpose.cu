// C3 · Matrix transpose — YOUR IMPLEMENTATION.  x: [M, N] -> y: [N, M], y[c][r] = x[r][c]
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

constexpr int TILE = 32;
constexpr int ROWS = 8;

// block (32, 8), one thread per element of x.
__global__ void transpose_naive_kernel(const float* __restrict__ x, float* __restrict__ y, int M, int N) {
  // TODO
}

// block (TILE, ROWS) = 256 threads per 32x32 tile; each thread copies TILE / ROWS rows.
// Stage the tile in shared memory so both the global read and the global write are coalesced.
__global__ void transpose_tiled_kernel(const float* __restrict__ x, float* __restrict__ y, int M, int N) {
  __shared__ float tile[TILE][TILE + 1];  // why + 1? (see hints)
  // TODO: phase 1 read x tile -> shared, __syncthreads(), phase 2 write shared -> y tile
}

static void check_input(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat,
              "x must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "x must be 2-D");
}

torch::Tensor transpose_naive(torch::Tensor x) {
  check_input(x);
  const int M = x.size(0), N = x.size(1);
  auto y = torch::empty({N, M}, x.options());
  if (M == 0 || N == 0) return y;
  dim3 block(32, 8);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  transpose_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), M, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor transpose_tiled(torch::Tensor x) {
  check_input(x);
  const int M = x.size(0), N = x.size(1);
  auto y = torch::empty({N, M}, x.options());
  if (M == 0 || N == 0) return y;
  dim3 block(TILE, ROWS);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  transpose_tiled_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), M, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
