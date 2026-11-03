// C2 · Reduction sum — YOUR IMPLEMENTATION.
// thread (grid-stride loop) -> warp (__shfl_down_sync) -> block (shared memory) -> grid (atomicAdd)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

// Return the sum of v over the 32 lanes of the warp (only lane 0 needs the right answer).
__device__ __forceinline__ float warp_reduce_sum(float v) {
  // TODO: 5 rounds of __shfl_down_sync(0xffffffff, v, offset), offset = 16, 8, 4, 2, 1
  return v;
}

template <int BLOCK>
__global__ void reduce_sum_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
  static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "BLOCK must be a multiple of 32, at most 1024");
  constexpr int WARPS = BLOCK / 32;
  __shared__ float warp_sums[WARPS];
  // TODO:
  //   1. each thread: grid-stride loop, private running sum
  //   2. warp_reduce_sum
  //   3. lane 0 of each warp -> warp_sums[warp]; __syncthreads(); warp 0 reduces warp_sums
  //   4. lane 0 of warp 0: atomicAdd(out, block_sum)
}

torch::Tensor reduce_sum(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat,
              "x must be a contiguous float32 CUDA tensor");
  auto out = torch::zeros({}, x.options());
  const int64_t n = x.numel();
  if (n == 0) return out;
  constexpr int BLOCK = 256;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t blocks = std::min<int64_t>((n + BLOCK - 1) / BLOCK, (int64_t)sms * 4);
  reduce_sum_kernel<BLOCK><<<blocks, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
