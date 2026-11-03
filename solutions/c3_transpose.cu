// C3 · Matrix transpose — reference solution.
//
// y[c][r] = x[r][c] for x: [M, N] -> y: [N, M], both row-major float32.
//
// naive : each thread reads x[r][c] (coalesced: consecutive threads, consecutive c) and
//         writes y[c][r] (strided by M floats: every lane of the warp hits a different
//         memory segment). Writes become ~32 separate transactions per warp.
// tiled : a block stages a 32x32 tile in shared memory, then reads it back "sideways"
//         so that the global write is also coalesced. Shared memory is padded to 33
//         columns so the sideways read does not hit the same bank 32 times.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

constexpr int TILE = 32;  // tile edge = warp size: one warp covers one tile row
constexpr int ROWS = 8;   // threads per block in y; each thread copies TILE / ROWS = 4 rows

__global__ void transpose_naive_kernel(const float* __restrict__ x, float* __restrict__ y, int M, int N) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;  // column in x (fast index)
  const int r = blockIdx.y * blockDim.y + threadIdx.y;  // row in x
  if (r < M && c < N) {
    y[(int64_t)c * M + r] = x[(int64_t)r * N + c];
  }
}

__global__ void transpose_tiled_kernel(const float* __restrict__ x, float* __restrict__ y, int M, int N) {
  // +1 column of padding: element tile[i][j] lives in bank (i * 33 + j) % 32 = (i + j) % 32,
  // so a warp reading a column tile[0..31][j] touches 32 different banks.
  // Without padding, (i * 32 + j) % 32 = j: all 32 lanes hit one bank -> 32-way conflict.
  __shared__ float tile[TILE][TILE + 1];

  // Phase 1: coalesced read of x's tile (blockIdx.y, blockIdx.x) into shared memory.
  const int c = blockIdx.x * TILE + threadIdx.x;   // column in x
  const int r0 = blockIdx.y * TILE + threadIdx.y;  // first row in x handled by this thread
#pragma unroll
  for (int j = 0; j < TILE; j += ROWS) {
    const int r = r0 + j;
    if (r < M && c < N) tile[threadIdx.y + j][threadIdx.x] = x[(int64_t)r * N + c];
  }

  __syncthreads();  // all of the tile must be in shared memory before anyone reads it back

  // Phase 2: x's tile (by, bx) becomes y's tile (bx, by). Swap the block coordinates and
  // let threadIdx.x run along y's rows again, so the global write is coalesced too.
  const int yc = blockIdx.y * TILE + threadIdx.x;   // column in y  (= row in x)
  const int yr0 = blockIdx.x * TILE + threadIdx.y;  // row in y     (= column in x)
#pragma unroll
  for (int j = 0; j < TILE; j += ROWS) {
    const int yr = yr0 + j;
    // y[yr][yc] = x[yc][yr] = tile[yc - by*TILE][yr - bx*TILE] = tile[threadIdx.x][threadIdx.y + j]
    if (yr < N && yc < M) y[(int64_t)yr * M + yc] = tile[threadIdx.x][threadIdx.y + j];
  }
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
  dim3 block(TILE, ROWS);                                      // 256 threads
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);     // one block per 32x32 tile of x
  transpose_tiled_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), M, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
