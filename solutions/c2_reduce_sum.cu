// C2 · Reduction sum — reference solution.
//
// Three levels, each using the cheapest mechanism available at that level:
//   1. thread : a grid-stride loop keeps a private running sum in a register;
//   2. warp   : 5 rounds of __shfl_down_sync combine 32 lanes without touching memory;
//   3. block  : each warp's lane 0 writes one partial to shared memory, warp 0 reduces those;
//   4. grid   : one atomicAdd per block onto the single output float.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

// Sum of v across the 32 lanes of a warp; the full result ends up in lane 0.
// Round 1: lane i adds lane i+16; round 2: lane i adds lane i+8; ... 16, 8, 4, 2, 1.
// 0xffffffff = all 32 lanes take part (they must all reach this call).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

template <int BLOCK>
__global__ void reduce_sum_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
  static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "BLOCK must be a multiple of 32, at most 1024");
  constexpr int WARPS = BLOCK / 32;
  __shared__ float warp_sums[WARPS];  // one slot per warp in this block

  // 1. Thread level. Consecutive threads read consecutive elements on every iteration,
  //    so each warp-wide load is coalesced. Out-of-range threads simply add nothing.
  float v = 0.0f;
  const int64_t stride = (int64_t)BLOCK * gridDim.x;
  for (int64_t i = (int64_t)blockIdx.x * BLOCK + threadIdx.x; i < n; i += stride) {
    v += x[i];
  }

  // 2. Warp level.
  v = warp_reduce_sum(v);

  // 3. Block level: lane 0 of every warp publishes its warp's sum ...
  const int lane = threadIdx.x & 31;  // threadIdx.x % 32
  const int warp = threadIdx.x >> 5;  // threadIdx.x / 32
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();  // ... and every warp must have written before warp 0 reads

  // ... then the first warp reduces the WARPS partials (lanes >= WARPS contribute 0).
  if (warp == 0) {
    v = (lane < WARPS) ? warp_sums[lane] : 0.0f;
    v = warp_reduce_sum(v);
    // 4. Grid level: one atomic per block. Float addition is not associative, so the
    //    order in which blocks arrive can change the last bits of the result.
    if (lane == 0) atomicAdd(out, v);
  }
}

torch::Tensor reduce_sum(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat,
              "x must be a contiguous float32 CUDA tensor");
  auto out = torch::zeros({}, x.options());  // atomicAdd accumulates into this, so start at 0
  const int64_t n = x.numel();
  if (n == 0) return out;

  constexpr int BLOCK = 256;
  // A few blocks per SM is enough to saturate bandwidth; more blocks only add atomics.
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t blocks = std::min<int64_t>((n + BLOCK - 1) / BLOCK, (int64_t)sms * 4);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  reduce_sum_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
