lede: CUDA C++ 里写 kernel 只需要十几个关键字。这里按“定义 → 启动 → 索引 → 同步 → warp 原语 → 接进 PyTorch”的顺序过一遍。
# CUDA C++ 语法速查

## 定义和启动 kernel

```cpp
// __global__：在 GPU 上执行、从 CPU 启动的函数，返回值必须是 void
__global__ void scale_kernel(const float* __restrict__ x, float* __restrict__ y, float a, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程编号
    if (i < n) y[i] = a * x[i];                                     // 边界检查
}

// __device__：只能在 GPU 上被其他 GPU 函数调用的辅助函数
__device__ __forceinline__ float square(float v) { return v * v; }

// 启动：<<<grid, block, 动态 shared memory 字节数, stream>>>
int threads = 256;
int64_t blocks = (n + threads - 1) / threads;          // 向上取整，覆盖所有元素
scale_kernel<<<blocks, threads, 0, stream>>>(x, y, 2.0f, n);
```

| 内建变量 | 含义 |
|---|---|
| `threadIdx.{x,y,z}` | 线程在 block 内的坐标 |
| `blockIdx.{x,y,z}` | block 在 grid 内的坐标 |
| `blockDim.{x,y,z}` | block 的形状（每维线程数） |
| `gridDim.{x,y,z}` | grid 的形状（每维 block 数） |

二维 launch 用 `dim3`：`dim3 block(32, 8); dim3 grid(ceil(N/32), ceil(M/8));`。约定 **`threadIdx.x` 对应内存里最快变化的维度**（行主序矩阵的列），这样一个 warp 的 32 个线程访问连续地址。

::: callout warn 索引溢出
`blockIdx.x * blockDim.x` 是 32-bit 无符号乘法，元素数超过 2³² 时会溢出。先转成 `int64_t` 再乘。行主序下标 `row * N + col` 同理。
:::

### grid-stride loop

```cpp
for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)blockDim.x * gridDim.x) {
    ...
}
```

grid 大小不再需要覆盖 n：可以固定成“每个 SM 若干个 block”，每个线程循环处理多个元素。好处是 launch 配置与数据大小无关，block 数少了调度开销也小，每个线程的累加（如 reduction）还可以留在寄存器里。

## shared memory 与同步

```cpp
__global__ void k(...) {
    __shared__ float tile[32][33];   // 静态大小，block 内所有线程共享
    tile[threadIdx.y][threadIdx.x] = ...;
    __syncthreads();                 // block 内所有线程到达这里之后才继续
    ... = tile[threadIdx.x][threadIdx.y];
}
```

- `__syncthreads()` 必须被 block 内**所有**线程执行到。把它放进只有部分线程会进入的 `if` 里会死锁或出现未定义行为。
- 典型模式：**load → sync → compute → sync**。第二个 sync 防止下一轮迭代的 load 覆盖别人还在读的数据。

## warp 级原语

warp 内 32 个线程可以直接交换寄存器里的值，不经过 shared memory：

```cpp
float v = ...;
// 从 lane (laneid + offset) 读 v；超出范围的 lane 读到自己的值
v += __shfl_down_sync(0xffffffff, v, 16);   // 0xffffffff：参与的 lane 掩码（全部 32 个）
v += __shfl_down_sync(0xffffffff, v, 8);
...                                          // 5 轮后 lane 0 持有 32 个值的和
float s = __shfl_xor_sync(0xffffffff, v, 1); // butterfly：与 lane ^ 1 交换，所有 lane 都得到结果
```

`lane = threadIdx.x % 32`，`warp = threadIdx.x / 32`（一维 block 时）。

## 原子操作

`atomicAdd(float* addr, float v)`：多个线程 / block 安全地加到同一地址。代价是同一地址上的原子操作会串行化，所以一般“先在 block 内归约，最后每个 block 只做一次 atomic”。浮点加法不满足结合律，**原子加的结果每次运行可能差最后几位**（非确定性）。

## 接进 PyTorch：`load_inline`

所有 CUDA 题都用这个模式，你只需要写 `.cu` 文件；`mine/<题>.py` 负责编译和导出：

```cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>       // at::cuda::getCurrentCUDAStream()
#include <c10/cuda/CUDAException.h>      // C10_CUDA_KERNEL_LAUNCH_CHECK()

__global__ void my_kernel(const float* x, float* y, int64_t n) { ... }

torch::Tensor my_op(torch::Tensor x) {                 // host 函数：参数检查 + 分配 + launch
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "...");
    auto y = torch::empty_like(x);
    my_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();                    // launch 配置错误会在这里报出来
    return y;
}
```

```python
from torch.utils.cpp_extension import load_inline
mod = load_inline(name="my_ext", cpp_sources="torch::Tensor my_op(torch::Tensor x);",
                  cuda_sources=open("my.cu").read(), functions=["my_op"])
y = mod.my_op(x)
```

- 用 PyTorch 当前的 stream 启动，kernel 才会和前后的 torch 操作正确排序。
- 第一次编译约 30–60 秒，之后命中缓存（`~/.cache/torch_extensions`）。
- kernel 内部的越界访问通常不会立刻报错，而是在**下一个**同步点报 `illegal memory access`。调试时设 `CUDA_LAUNCH_BLOCKING=1`，或者用 `compute-sanitizer python -m pytest ...`。

## 图解与类比

::: flow load_inline：一个 .cu 文件怎么变成 torch 算子
n cu 0 0 "你的 .cu" "kernel + host 函数" k=acc
n decl 0 1 "C++ 声明" "cpp_sources" k=acc
n nvcc 1 0 "nvcc" "→ cuda.o" k=vio
n cxx 1 1 "c++" "→ main.o（pybind）" k=vio
n so 2 0.5 "ninja 链接" "→ 扩展 .so" k=warn
n py 3 0.5 "Python 模块" "mod.my_op(x)" k=pass
n cache 2 1.65 "~/.cache/torch_extensions" "按名字 + 源码缓存" k=ghost w=1.35
e cu -> nvcc
e decl -> cxx "生成绑定"
e nvcc -> so
e cxx -> so
e so -> py "import"
e so -> cache dash
cap: 第一次调用要编译（30–60 秒），之后直接加载缓存。本仓库的 `common/cuda_ext.py` 把源码哈希放进模块名，所以改完 .cu 会自动重新编译。
:::
::: flow kernel launch 是异步的
n py 0 0 "Python 调用" "立即返回" k=acc
n q 1 0 "CUDA stream" "kernel 排队" k=warn
n gpu 2 0 "GPU 执行" "grid 分发到 SM" k=pass
n sync 3 0 "同步点" ".item() / synchronize()" k=todo w=1.1
e py -> q "enqueue" anim
e q -> gpu "按顺序执行"
e gpu -> sync "完成后才拿到结果"
cap: 所以用 `time.time()` 包住一次调用，测到的只是“下单”的时间；越界访问的报错也常常在后面某个同步点才出现。
:::
::: callout analogy launch 就是给工厂下订单
`kernel<<<...>>>` 像在网上下单：点完“提交”就返回了，货（结果）还在路上。要拿到货（读结果、打印、`.item()`），就得等快递到门口（同步）。
所以计时要在“收货”时按秒表（CUDA event + synchronize），而不是在“下单”时。
:::
::: callout analogy __syncthreads 是集体合影
摄影师要等所有人都站好才按快门。如果有人被安排在 `if` 分支里永远不会来，摄影师就一直等下去（死锁）。所以 `__syncthreads()` 必须放在所有线程都会执行到的地方。
:::
::: callout analogy grid-stride loop 是固定人数的流水线
不是“来多少箱货就雇多少工人”，而是雇固定数量的工人（SM 数 × k 个 block），每人做完一箱就去传送带上拿下一箱（`i += 总线程数`）。货多货少都用同一套班底，工人的私人账本（寄存器累加）还能一直记下去。
:::

## 源码与资料

**文档**：[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) · [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) · [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) · [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

**源码**：[[pytorch:torch/utils/cpp_extension.py@load_inline]] · [[cudasamples:cpp/0_Introduction/vectorAdd/vectorAdd.cu]] · [[cudasamples:cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu]] · [[flashinfer:include/flashinfer/vec_dtypes.cuh]]（FlashInfer 的向量化加载工具）

