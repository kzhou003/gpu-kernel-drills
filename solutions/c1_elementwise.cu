// C1 · Vector add + fused bias-GELU — reference solution.
//
// Two kernels, two indexing styles:
//   vector_add_kernel : one thread per element, grid sized to cover n exactly (+ bounds check).
//   bias_gelu_kernel  : grid-stride loop, grid capped at ~32 blocks per SM, so one launch
//                       works for any size and each thread does several elements.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

#define CHECK_F32_CUDA(t)                                                   \
  TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor");                  \
  TORCH_CHECK((t).is_contiguous(), #t " must be contiguous");               \
  TORCH_CHECK((t).scalar_type() == at::kFloat, #t " must be float32")

// ---------------------------------------------------------------- vector add
// __restrict__ promises the compiler that a, b, c never alias, so it may keep values in
// registers and use the read-only data path for a and b.
__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int64_t n) {
  // Global index of this thread. Cast before multiplying: blockIdx.x * blockDim.x is
  // computed in 32-bit unsigned arithmetic and overflows once n > 2^32.
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {            // the last block is usually only partly full
    c[i] = a[i] + b[i];   // consecutive threads touch consecutive floats -> coalesced
  }
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  CHECK_F32_CUDA(a);
  CHECK_F32_CUDA(b);
  TORCH_CHECK(a.numel() == b.numel(), "a and b must have the same number of elements");
  auto c = torch::empty_like(a);
  const int64_t n = a.numel();
  if (n == 0) return c;  // a launch with 0 blocks is an error

  const int threads = 256;                               // a multiple of 32 (warp size)
  const int64_t blocks = (n + threads - 1) / threads;    // ceil(n / threads)
  // Launch on PyTorch's current stream so the kernel is ordered with other torch ops.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vector_add_kernel<<<blocks, threads, 0, stream>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();  // surfaces bad launch configs immediately
  return c;
}

// ---------------------------------------------------------------- fused bias + GELU
// GELU, tanh approximation (what torch.nn.functional.gelu(approximate="tanh") computes):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
__device__ __forceinline__ float gelu_tanh(float x) {
  const float k0 = 0.7978845608028654f;  // sqrt(2 / pi)
  const float k1 = 0.044715f;
  return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}

// x: [rows, cols] row-major, bias: [cols], y: [rows, cols].
// Unfused PyTorch does `t = x + bias` (read x, write t) then `gelu(t)` (read t, write y):
// 4 passes over memory. Fused: read x once, write y once -> about half the traffic.
__global__ void bias_gelu_kernel(const float* __restrict__ x,
                                 const float* __restrict__ bias,
                                 float* __restrict__ y,
                                 int64_t total,
                                 int64_t cols) {
  const int64_t stride = (int64_t)blockDim.x * gridDim.x;  // threads in the whole grid
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    // Row-major: element i sits in column i % cols. bias is tiny and stays in L1/L2.
    y[i] = gelu_tanh(x[i] + bias[i % cols]);
  }
}

torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor bias) {
  CHECK_F32_CUDA(x);
  CHECK_F32_CUDA(bias);
  TORCH_CHECK(x.dim() == 2, "x must be 2-D [rows, cols]");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == x.size(1), "bias must be [cols]");
  auto y = torch::empty_like(x);
  const int64_t total = x.numel();
  if (total == 0) return y;

  const int threads = 256;
  // Enough blocks to fill every SM several times over, but no more: the grid-stride loop
  // covers the rest, and fewer blocks means less scheduling overhead for huge tensors.
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t blocks = std::min<int64_t>((total + threads - 1) / threads, (int64_t)sms * 32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bias_gelu_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), total, x.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
