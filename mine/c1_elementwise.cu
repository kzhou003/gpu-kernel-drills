// C1 · Vector add + fused bias-GELU — YOUR IMPLEMENTATION.
// Fill in the two kernels and the two launch configurations marked TODO.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

#define CHECK_F32_CUDA(t)                                                   \
  TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor");                  \
  TORCH_CHECK((t).is_contiguous(), #t " must be contiguous");               \
  TORCH_CHECK((t).scalar_type() == at::kFloat, #t " must be float32")

// c[i] = a[i] + b[i] for i < n. One thread per element.
__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int64_t n) {
  // TODO: compute this thread's global index (as int64_t), bounds-check, add.
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  CHECK_F32_CUDA(a);
  CHECK_F32_CUDA(b);
  TORCH_CHECK(a.numel() == b.numel(), "a and b must have the same number of elements");
  auto c = torch::empty_like(a);
  const int64_t n = a.numel();
  if (n == 0) return c;

  const int threads = 256;
  const int64_t blocks = 0;  // TODO: enough blocks of `threads` threads to cover n elements
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vector_add_kernel<<<blocks, threads, 0, stream>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

// GELU, tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
__device__ __forceinline__ float gelu_tanh(float x) {
  // TODO
  return x;
}

// y[r][c] = gelu_tanh(x[r][c] + bias[c]); x, y: [rows, cols] row-major; total = rows * cols.
// Use a grid-stride loop so any grid size covers all `total` elements.
__global__ void bias_gelu_kernel(const float* __restrict__ x,
                                 const float* __restrict__ bias,
                                 float* __restrict__ y,
                                 int64_t total,
                                 int64_t cols) {
  // TODO
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
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t blocks = 0;  // TODO: ceil(total / threads), but at most sms * 32
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bias_gelu_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), total, x.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
