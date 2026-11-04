title: Metal / MLX 与 CUDA / Triton：从硬件到 kernel 的对比
lede: 同一个 kernel 写在 NVIDIA GPU 和 Apple GPU 上，为什么写法、框架和性能都不一样？这一页按“硬件 → 软件设计 → 硬件差异怎么塑造了软件 → 框架与性能”的顺序对比，配上本仓库在 RTX 5090 和 M2 上同一类 kernel 的实测数字，以及 C1/T1/X1、C2/X2、C4/T4/X3 这几组题目的代码对照。

::: callout info 怎么读这一页
你在这套题里已经在两边都写过 kernel：Track C（CUDA C++）、Track T（Triton）和 Track X（MLX 的 Metal kernel）。这一页把它们放在一起比较。硬件参数是公开的规格，标了“约”；性能数字只要本仓库测过，就用实测（RTX 5090 + FlashInfer 0.7.2、Apple M2 24 GB + MLX 0.32），并注明来自哪一题。
:::

## ① 一张图看懂最大的区别

::: flow 两种机器的整体结构
cfg cw=210 nw=170
n cpu1 0 0 "CPU + 主内存" "DDR5，几百 GB" k=acc
n pcie 1 0 "PCIe / NVLink" "约 64 GB/s（PCIe 5 x16）" k=todo
n gpu1 2 0 "独立 GPU" "170 个 SM（RTX 5090）" k=vio
n hbm 3 0 "显存" "32 GB · 约 1.8 TB/s" k=warn
n soc 0.5 1.5 "Apple SoC：CPU + GPU + NPU" "同一块芯片" k=pass w=1.3
n um 2 1.5 "统一内存（LPDDR5）" "M2：24 GB · 约 100 GB/s" k=warn w=1.2
g 0 0 3 0 "NVIDIA：独立显卡" k=vio
g 0.5 1.5 2 1.5 "Apple silicon：一块芯片、一块内存" k=pass
e cpu1 -> pcie
e pcie -> gpu1
e gpu1 -> hbm
e soc -> um "共用"
cap: NVIDIA 是“CPU 和 GPU 各有各的内存，中间隔一条 PCIe”；Apple 是“CPU 和 GPU 在同一块芯片上，读同一块内存”。几乎所有后面的差异都能追溯到这一点，以及带宽和算力差了一个数量级以上。
:::

::: callout analogy 独立厨房 vs 开放式厨房
NVIDIA 的机器像一家餐厅：前厅（CPU）和后厨（GPU）是两个房间，后厨有自己超大的冷库（显存）和极快的传菜通道（HBM / GDDR7），但前厅和后厨之间只有一个小窗口（PCIe），东西递来递去很费时间。Apple 的机器像一个开放式厨房：做饭的和点菜的在同一个房间、共用一个冰箱（统一内存），拿东西不用经过窗口，冰箱也可以很大（最多几百 GB），但冰箱门比餐厅冷库的门窄得多（带宽低一个数量级），厨师也少得多（算力低）。
:::

## ② 硬件对比

### 规格

| | NVIDIA RTX 5090 | NVIDIA H100 SXM | Apple M2（本仓库实测机） | Apple M4 Max / M3 Ultra |
|---|---|---|---|---|
| 内存 | 32 GB GDDR7（独立） | 80 GB HBM3（独立） | 24 GB 统一内存 | 最多 128 GB / 512 GB 统一内存 |
| 带宽 | 约 1.8 TB/s | 约 3.35 TB/s | 约 100 GB/s | 约 546 GB/s / 约 819 GB/s |
| 矩阵单元 | Tensor Core（FP16/BF16/FP8/FP4） | Tensor Core（+ TMA、wgmma） | simdgroup_matrix（8×8，在普通 ALU 上） | 同左；M5 起 GPU 核心里加入了 neural accelerator |
| 本仓库实测的 GEMM | fp16 约 218 TFLOP/s（T4） | — | fp32 约 2.6 TFLOP/s（X3，MLX 自带） | — |
| 功耗量级 | 约 575 W（整卡） | 约 700 W | 整机几十瓦 | 整机一百多瓦 |

### 编程模型里能看到的硬件

| 概念 | NVIDIA | Apple |
|---|---|---|
| 一组同步执行的线程 | warp，32 个 | simdgroup，32 个（Apple GPU 上） |
| 一个 block 能用的片上共享内存 | `__shared__`，几十到两百多 KB（H100 每 SM 228 KB） | `threadgroup` 内存，**最多 32 KB** |
| warp 内交换数据 | `__shfl_*_sync`、`__reduce_*_sync` | `simd_shuffle_*`、`simd_sum`、`simd_max` |
| 矩阵乘指令 | `mma.sync`（16×8×16 等）、Hopper `wgmma`（64 行起） | `simdgroup_multiply_accumulate`（8×8×8） |
| 异步拷贝 | `cp.async`（Ampere 起）、TMA（Hopper 起） | 没有对应的公开指令 |
| 低精度 | FP8（Hopper / Ada 起）、FP4（Blackwell） | half / bfloat；没有 FP8 矩阵单元 |
| 多卡 | NVLink、NCCL | 单芯片；多台 Mac 之间走网络（MLX 有 distributed） |

::: flow 编程模型的层次对照
cfg cw=210 nw=170
n g1 0 0 "grid" "多少个 block" k=acc
n b1 0 1.1 "block（≤ 1024）" "__shared__" k=acc
n w1 0 2.2 "warp（32）" "shuffle · mma" k=acc
n t1 0 3.3 "thread" "寄存器" k=acc
n g2 1.6 0 "grid" "多少个线程（dispatchThreads）" k=pass w=1.1
n b2 1.6 1.1 "threadgroup（≤ 1024）" "threadgroup 内存 ≤ 32 KB" k=pass w=1.1
n w2 1.6 2.2 "simdgroup（32）" "simd_sum · simdgroup_matrix" k=pass w=1.1
n t2 1.6 3.3 "thread" "寄存器" k=pass w=1.1
e g1 -> b1
e b1 -> w1
e w1 -> t1
e g2 -> b2
e b2 -> w2
e w2 -> t2
e g1 -> g2 "单位不同" dash
cap: 四层结构几乎一一对应，所以 CUDA 的思路可以直接搬到 Metal。真正的区别在细节：grid 的单位、片上内存的大小、矩阵指令的大小，以及有没有异步拷贝。X1 讲解第 ③ 节专门讲了 grid 单位的坑。
:::

## ③ 软件栈对比

::: flow 两边的软件栈
cfg cw=410 nw=370
n py1 0 0 "PyTorch / JAX" "vLLM · SGLang · TensorRT-LLM" k=acc
n lib1 0 1.1 "库" "cuBLAS · CUTLASS · FlashAttention · FlashInfer" k=acc
n dsl1 0 2.2 "写 kernel 的方式" "CUDA C++ · Triton · CuTe DSL" k=acc
n rt1 0 3.3 "运行时 / 编译器" "CUDA runtime · nvcc / NVRTC · PTX" k=acc
n py2 1 0 "MLX / PyTorch MPS" "mlx-lm · llama.cpp" k=pass
n lib2 1 1.1 "库" "MLX 内置 Metal kernel · MPS" k=pass
n dsl2 1 2.2 "写 kernel 的方式" "MSL · mx.fast.metal_kernel" k=pass
n rt2 1 3.3 "运行时 / 编译器" "Metal · AIR · 运行时编译" k=pass
e py1 -> lib1
e lib1 -> dsl1
e dsl1 -> rt1
e py2 -> lib2
e lib2 -> dsl2
e dsl2 -> rt2
cap: CUDA 这边每一层都有好几个成熟的选择；Apple 这边更集中，LLM 方向基本围绕 MLX（以及 llama.cpp 的 Metal 后端）。Triton 目前没有官方的 Metal 后端，所以在 Mac 上写自定义 kernel 的主要方式就是 Metal Shading Language。
:::

| | CUDA C++ | Triton | Metal / MLX |
|---|---|---|---|
| 你写的是 | 一个线程的代码 | 一个 program（block）的代码 | 一个线程的代码（和 CUDA 一样） |
| 线程 / 共享内存 / 流水线 | 全部手动 | 编译器负责 | 全部手动 |
| 语言 | C++ 方言 | Python 子集 | C++14 方言（MSL） |
| 编译 | nvcc 提前编译，或 NVRTC 运行时编译 | 首次调用时 JIT | MLX 内置 kernel 预编译成 metallib；`metal_kernel` 首次调用时 JIT |
| 函数签名 | 自己写 | 自己写 | **MLX 自动生成**，你只写函数体（X1） |
| 启动 | `<<<blocks, threads, smem, stream>>>` | `kernel[grid](...)` | 由 MLX 编码进 command buffer |
| 调试与 profile | Nsight Systems / Compute | Triton 解释器（TRITON_INTERPRET）+ Nsight | Xcode Metal debugger、Instruments |

## ④ 硬件差异怎么塑造了软件设计

这是这一页的重点：两边的软件长得不一样，大多不是“谁的设计更好”，而是**硬件不一样，最优的设计就不一样**。

::: flow 硬件差异 → 软件设计
cfg cw=300 nw=220
n h1 0 0 "统一内存" "CPU、GPU 读同一块" k=vio
n h2 0 1.2 "带宽低一个数量级" "100–800 GB/s" k=vio
n h3 0 2.4 "threadgroup 内存小（32 KB）" "没有 cp.async / TMA" k=vio w=1.05
n h4 0 3.6 "矩阵单元小（8×8），没有 FP8" "" k=vio
n h5 0 4.8 "一个芯片 · command queue" "" k=vio w=1.05
n s1 1 0 "没有 .to(device)，数组不分设备" "MLX：选 stream，不搬数据" k=pass w=1.05
n s2 1 1.2 "LLM 的主战场是 decode + 量化" "4-bit 分组量化是一等公民" k=pass w=1.05
n s3 1 2.4 "更小的 tile、更简单的流水线" "不用 warp specialization" k=pass w=1.05
n s4 1 3.6 "simdgroup_matrix 拼 GEMM" "反量化后在 half 里算" k=pass w=1.05
n s5 1 4.8 "惰性求值 + 批量提交" "很多 op 进同一个 command buffer" k=pass w=1.05
e h1 -> s1
e h2 -> s2
e h3 -> s3
e h4 -> s4
e h5 -> s5
cap: 左边是 Apple GPU 和 NVIDIA GPU 的硬件差异，右边是 MLX / Metal kernel 因此做出的设计选择。下面逐条展开。
:::

### 1. 统一内存 → 数组不属于任何设备

CUDA 程序的一大块工作是管理两块内存：`cudaMalloc`、`cudaMemcpy`、`tensor.to("cuda")`，以及想办法让拷贝和计算重叠。Apple silicon 上 CPU 和 GPU 读的是同一块物理内存，所以 MLX 的设计是：**数组不分设备，你选的是在哪个 stream 上计算**。

```python
a = mx.random.normal((4096,))           # 不用说放在哪
mx.add(a, a, stream=mx.cpu)             # 在 CPU 上算
mx.add(a, a, stream=mx.gpu)             # 在 GPU 上算，同一块数据，没有拷贝
```

源码：MLX 分配的每一块内存都是 [[mlx:mlx/backend/metal/allocator.cpp@L16|MTL::ResourceStorageModeShared]]（CPU 和 GPU 都能直接访问）；文档 [[mlx:docs/src/usage/unified_memory.rst|unified_memory.rst]]。

代价：没有独立显存，“模型能不能放下”由整机内存决定；好处也在这里：一台 128 GB 的 Mac 能放下一张 32 GB 显卡放不下的模型（第 ⑥ 节）。

::: callout analogy 搬家 vs 换房间
CUDA 里，数据要从 CPU 用到 GPU，像把家具从一栋楼搬到另一栋楼（PCIe 拷贝），所以大家花很多心思安排什么时候搬、边搬边干活。统一内存像一套大房子，家具一直放在客厅，谁要用就走过去用（选 stream）。房子的大门不宽（带宽低），但至少不用搬家。
:::

### 2. 带宽低一个数量级 → 量化是一等公民

LLM decode 每生成一个 token 都要把全部权重读一遍，速度 ≈ 带宽 ÷ 模型字节数（L2 讲过）。M2 的 100 GB/s 只有 RTX 5090 的约 1/18。在 Mac 上提升 decode 速度最直接的办法，就是**让每个权重更小**。

所以 MLX 把 4-bit / 8-bit 分组量化做成了一等公民：`mx.quantize`、`QuantizedLinear`，以及专门的 Metal kernel：decode 用的矩阵乘向量 [[mlx:mlx/backend/metal/kernels/quantized.h@L851|qmv_fast_impl]]、prefill 用的 [[mlx:mlx/backend/metal/kernels/quantized.h@L1341|qmm_t_impl]]，反量化在 [[mlx:mlx/backend/metal/kernels/quantized.h@L558|dequantize]] 里和乘加融合。CUDA 这边当然也有量化（M3 的 FP8 block scaling），但数据中心 GPU 上的主要动机是算力（FP8 Tensor Core 快一倍），而 Mac 上的主要动机是带宽。

### 3. threadgroup 内存小、没有异步拷贝 → 更小的 tile、更简单的流水线

CUDA 上高性能 GEMM / attention 的套路（C4、T4、FlashAttention-3）是：大 tile（128×128 以上）放进上百 KB 的 shared memory，`cp.async` 或 TMA 异步预取下一块，Hopper 上再用 warp specialization 让一部分 warp 专门搬数据。

Apple GPU 上：

- threadgroup 内存最多 32 KB，tile 只能小得多。X3 的 simdgroup 版本用的是 32×32 的输出 tile 和 16 宽的 K 块。
- 没有公开的异步拷贝指令，“边搬边算”只能靠多个 simdgroup / threadgroup 交错执行来掩盖延迟。
- 所以 MLX 的 GEMM（steel）和 attention（[[mlx:mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h|steel_attention.h]]、decode 用的 [[mlx:mlx/backend/metal/kernels/sdpa_vector.h|sdpa_vector.h]]）结构上比 FA3 简单，更接近 FA2 和 C4 的风格。

### 4. 矩阵单元是 8×8 → 用 simdgroup_matrix 拼 GEMM

| | CUDA（T4 / C4 之后） | Metal（X3） |
|---|---|---|
| 一条矩阵指令 | `mma.sync` m16n8k16，Hopper `wgmma` 64×N×16 | `simdgroup_multiply_accumulate` 8×8×8 |
| 在哪执行 | 独立的 Tensor Core | GPU 的普通 ALU（M5 之前） |
| 写法 | Triton 一行 `tl.dot`；CUDA 里 `mma` / CuTe | 手动 `simdgroup_load`、累加、`simdgroup_store` |

X3 讲解里画过：一个 simdgroup 持有 2×2 个 8×8 累加器，做 2×2 的“块外积”。这和 C4 里一个线程持有 4×4 个标量做外积是同一个思想，只是粒度从标量变成了 8×8 的小矩阵。

### 5. 一个芯片、command buffer → 惰性求值 + 批量提交

CUDA 里每次 kernel launch 有几微秒的固定开销，decode 时大量小 kernel 的开销加起来很可观，所以有 CUDA graph（F1、F3 讲过）。Metal 的工作单位是 **command buffer**：把很多 kernel 编码进同一个 buffer，一次提交给 GPU。

MLX 用**惰性求值**配合这一点：你写 `c = a + b` 时什么都没算，只是记下了计算图；直到 `mx.eval(c)`（或需要结果时）才把整张图编码成 command buffer 提交。一个 command buffer 里攒多少个 op 有阈值：[[mlx:mlx/backend/metal/device.cpp@L567-L572|needs_commit：超过 max_ops 或 max_mb 才提交]]。文档：[[mlx:docs/src/usage/lazy_evaluation.rst|lazy_evaluation.rst]]。`mx.compile` 还能把一串逐元素操作融合成一个 kernel（X1 的“测完之后”提到过）。

::: callout analogy 拼单发货
CUDA 的普通 launch 像每做好一样东西就叫一次快递，每单都有起步价（launch 开销）。CUDA graph 是把一整套流程录成模板，以后一键重放。MLX 的惰性求值像网购的购物车：你一直往里加东西（记录计算图），结账（eval）时才打包成一个大包裹（command buffer）一起发。两者解决的是同一个问题：别让固定开销淹没小 kernel。
:::

## ⑤ 同一个 kernel，三种写法

### 逐元素：add + GELU（C1 / T1 / X1）

::: raw
<div style="display:grid; grid-template-columns: 1fr; gap: 8px">
<div class="codebox"><div class="code-head"><span class="path">CUDA（C1，简化）</span></div><pre><code class="language-cpp">__global__ void k(const float* x, float* y,
                  int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x
              + threadIdx.x;
  if (i &lt; n) y[i] = gelu_tanh(x[i]);
}
k&lt;&lt;&lt;(n + 255) / 256, 256, 0, stream&gt;&gt;&gt;(x, y, n);</code></pre></div>
<div class="codebox"><div class="code-head"><span class="path">Triton（T1，简化）</span></div><pre><code class="language-python">@triton.jit
def k(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs &lt; n
    x = tl.load(x_ptr + offs, mask=mask)
    ...
    tl.store(out_ptr + offs, out, mask=mask)
k[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)</code></pre></div>
<div class="codebox"><div class="code-head"><span class="path">Metal / MLX（X1，简化）</span></div><pre><code class="language-cpp">// 只写函数体，签名由 MLX 生成
uint i = thread_position_in_grid.x;
if (i &lt; n) {
  out[i] = T(gelu_tanh(float(x[i]) + float(y[i])));
}
// Python：grid 是线程总数，不是 block 数
// kernel(inputs=[x, y, n], grid=(n,1,1),
//        threadgroup=(256,1,1), ...)</code></pre></div>
</div>
:::

- **CUDA 和 Metal 的 kernel 体几乎一样**：都是“我是第几个线程、算一个元素”。区别在外面：CUDA 的 grid 是 block 数，Metal 的 grid 是线程总数；Metal 的签名由 MLX 生成。
- **Triton 换了一个抽象层**：你写的是一整块数据，线程怎么分由编译器决定。
- 源码：[[here:solutions/c1_elementwise.cu@L20-L31|C1]] · [[here:solutions/t1_add_gelu.py@L16-L34|T1]] · [[here:solutions/x1_metal_elementwise.py@L13-L49|X1]]

### 归约：一行求和（C2 / T2 / X2）

| | CUDA（C2） | Triton（T2） | Metal（X2） |
|---|---|---|---|
| warp 内归约 | `for (o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);` | `tl.sum(x, axis=0)` 一行，编译器生成 | `v = simd_sum(v);` 一行，硬件做完 |
| block 内 | `__shared__` + `__syncthreads()` | 编译器生成 | `threadgroup` + `threadgroup_barrier` |
| 谁拿到结果 | 只有 lane 0（`shfl_down`） | 广播给所有线程 | 每个 lane 都有（`simd_sum`） |

Metal 在这一层其实比 CUDA 更方便：`simd_sum` / `simd_max` 一个函数就是整个 warp 的归约，结果每个 lane 都有，所以 X2 的 `tg_reduce` 让每个 simdgroup 都做第二轮，省掉一次广播（X2 讲解第 ③ 节）。源码：[[here:solutions/c2_reduce_sum.cu@L17-L23|C2 warp_reduce_sum]] · [[here:solutions/x2_metal_softmax.py@L14-L25|X2 tg_reduce]]。

### GEMM 的内层（C4 / T4 / X3）

| | CUDA C4 v2 | Triton T4 | Metal X3 |
|---|---|---|---|
| 内层 | 线程级 4×4 标量外积 | `acc = tl.dot(a, b, acc)` | simdgroup 级 2×2 个 8×8 块外积 |
| 用到的硬件 | FP32 CUDA core | Tensor Core（fp16） | simdgroup_matrix（fp32） |
| 片上 tile | 64×64×16，shared memory | autotune，128×256×64 也可以 | 32×32×16，threadgroup 内存 |
| 预取 | 没有 | `num_stages` 软件流水线 | 没有 |

源码：[[here:solutions/c4_gemm.cu@L93-L105|C4 外积]] · [[here:solutions/t4_matmul.py@L42-L50|T4 主循环]] · [[here:solutions/x3_metal_matmul.py@L59-L69|X3 simdgroup MMA]]。

## ⑥ 性能对比

### 本仓库的实测：同一类 kernel

| kernel | RTX 5090 | Apple M2 | 比例 |
|---|---|---|---|
| 逐元素（fp16，带宽） | T1：**1548 GB/s** | X1：46 GB/s（MLX 自带；同一时刻 `x + 1` 实测 59 GB/s，机器负载较高） | 约 26–34 倍 |
| 行 softmax（N = 4096，fp16） | T2：43.6 us · **1540 GB/s**（M = 4096） | X2：mx.softmax 716 us（M = 2048），约 47 GB/s | 约 33 倍 |
| GEMM FP32（大矩阵） | C4：cuBLAS 65.4 TFLOP/s | X3：MLX steel 2.62 TFLOP/s | 约 25 倍 |
| GEMM FP16 | T4：**217.8 TFLOP/s**（8192³） | — | — |

两点值得注意：

- **都打到了各自的上限**。T1、T2 都到了 5090 峰值的 85% 左右；X1 的 MLX 自带操作也接近那台 M2 当时实测的带宽。所以“慢”不是 kernel 写得差，而是硬件上限低。
- **比例和硬件规格的比例差不多**：带宽差约 18 倍（标称），GEMM 差约 25 倍。memory-bound 的 kernel 差多少，基本就是带宽差多少。

### roofline 的拐点

| | 计算峰值（实测） | 带宽（实测） | 拐点（FLOP / 字节） |
|---|---|---|---|
| RTX 5090，FP32 | 65.4 TFLOP/s（cuBLAS） | 约 1.57 TB/s（C1） | 约 42 |
| RTX 5090，FP16 Tensor Core | 217.8 TFLOP/s（T4） | 约 1.57 TB/s | 约 139 |
| Apple M2，FP32 | 2.62 TFLOP/s（MLX steel） | 约 59 GB/s（X1，负载较高） | 约 44（按标称 100 GB/s 算约 26） |

一个有意思的结论：FP32 下两台机器的拐点差不多，所以**同一个算法在两边是 compute-bound 还是 memory-bound，结论基本一样**，只是绝对速度差了一个数量级。真正拉开差距的是 Tensor Core：它把 5090 的拐点推高了三倍多，而 M2 没有对应的东西。

### LLM 推理：速度和容量的取舍

decode 速度的上限 ≈ 带宽 ÷ 每个 token 要读的字节数（约等于模型权重大小）。用这个公式估一下（只是上限，不含 KV cache 和其他开销）：

::: raw
<style>
.cmpbars { margin: 10px 0; font-size: 12.5px; }
.cmpbars .row { display: grid; grid-template-columns: 210px 1fr 130px; gap: 8px; align-items: center; margin: 3px 0; }
.cmpbars .bar { height: 18px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
.cmpbars .fill { height: 100%; }
.cmpbars .nv { background: var(--f-vio-bg); border-right: 2px solid var(--f-vio); }
.cmpbars .ap { background: var(--f-pass-bg); border-right: 2px solid var(--f-pass); }
.cmpbars .no { background: repeating-linear-gradient(135deg, var(--surface), var(--surface) 4px, var(--f-todo-bg) 4px, var(--f-todo-bg) 8px); }
.cmpbars .v { font-family: var(--mono); text-align: right; }
.cmpbars .h { font-weight: 600; margin: 10px 0 4px; }
</style>
<div class="cmpbars">
<div class="h">Llama-3 8B，4-bit（约 4.5 GB）：decode 速度上限（token/s）</div>
<div class="row"><span>RTX 5090（1.8 TB/s）</span><div class="bar"><div class="fill nv" style="width:100%"></div></div><span class="v">约 400</span></div>
<div class="row"><span>M3 Ultra（819 GB/s）</span><div class="bar"><div class="fill ap" style="width:45.5%"></div></div><span class="v">约 182</span></div>
<div class="row"><span>M4 Max（546 GB/s）</span><div class="bar"><div class="fill ap" style="width:30.3%"></div></div><span class="v">约 121</span></div>
<div class="row"><span>M2（100 GB/s）</span><div class="bar"><div class="fill ap" style="width:5.6%"></div></div><span class="v">约 22</span></div>
<div class="h">Llama-3 70B，4-bit（约 40 GB）</div>
<div class="row"><span>RTX 5090（32 GB）</span><div class="bar"><div class="fill no" style="width:100%"></div></div><span class="v">放不下</span></div>
<div class="row"><span>M3 Ultra（819 GB/s）</span><div class="bar"><div class="fill ap" style="width:100%"></div></div><span class="v">约 20</span></div>
<div class="row"><span>M4 Max 128 GB（546 GB/s）</span><div class="bar"><div class="fill ap" style="width:66.7%"></div></div><span class="v">约 14</span></div>
<div class="row"><span>M2 24 GB</span><div class="bar"><div class="fill no" style="width:100%"></div></div><span class="v">放不下</span></div>
</div>
<p class="legend">按“带宽 ÷ 权重字节数”估算的上限，实际会低一些。70B 那组的条长是相对 819 GB/s 画的。单张 5090 放不下 40 GB 的权重，需要多卡或把一部分放到 CPU 内存（那就受 PCIe 带宽限制了）。</p>
:::

- **decode（生成）**：memory-bound。NVIDIA 显卡带宽高，同一个模型快得多；但只要模型放得下，Mac 也能跑出可用的速度。
- **容量**：Mac 的统一内存可以到 128–512 GB，一台机器就能放下单张消费级显卡放不下的大模型。这是 Mac 跑本地大模型最大的优势。
- **prefill（处理 prompt）**：compute-bound，靠矩阵算力。这里差距最大（5090 的 FP16 Tensor Core 实测 218 TFLOP/s，M2 的 FP32 GEMM 2.6 TFLOP/s），所以长 prompt 的首 token 延迟在 Mac 上明显更长。
- **并发服务**：CUDA 生态有 paged KV cache、continuous batching、FlashInfer、vLLM / SGLang 这一整套（Track L、F）。Mac 上的工具更偏向单用户本地推理。

::: callout analogy 跑车和房车
RTX 5090 像跑车：极快（带宽、算力），但后备箱只有 32 GB，装不下大件行李（大模型）。高配 Mac 像房车：速度慢得多，但能把整个家（70B 模型）装进去开走。一张 H100 像大货车：又快又能装，但不是家用的价钱和功耗。选哪个，取决于你要拉什么、拉多远。
:::

## ⑦ 框架和生态对比

| | CUDA 生态 | Apple 生态 |
|---|---|---|
| 主力框架 | PyTorch（+ torch.compile / Inductor 生成 Triton） | MLX；PyTorch 也有 MPS 后端 |
| 写自定义 kernel | CUDA C++、Triton、CuTe DSL、cuTile 等，多种选择 | Metal Shading Language（`mx.fast.metal_kernel` 或 C++ 扩展） |
| attention / GEMM 库 | FlashAttention、FlashInfer、cuBLAS、CUTLASS | MLX 内置：`mx.fast.scaled_dot_product_attention`、steel GEMM、量化 matmul |
| LLM 推理引擎 | vLLM、SGLang、TensorRT-LLM | mlx-lm、llama.cpp（Metal 后端） |
| 编译器自动生成 kernel | Triton（Inductor）、XLA | `mx.compile` 融合逐元素操作 |
| 多设备 | NCCL、NVLink、张量并行 / 专家并行 | `mx.distributed`，多台 Mac 之间 |
| 调试 / profile | Nsight Systems、Nsight Compute、compute-sanitizer | Xcode Metal debugger、Instruments、`mx.metal.start_capture()` |

MLX 本身也在往跨平台走：它已经有一个 CUDA 后端（[[mlx:mlx/backend/cuda/device.cpp|mlx/backend/cuda/]]，GEMM 走 [[mlx:mlx/backend/cuda/gemms/cublas_gemm.cpp|cuBLAS]]），同一份 MLX 代码可以跑在 NVIDIA GPU 上。MLX 的“快速”算子清单在 [[mlx:mlx/fast.h@L14-L90|mlx/fast.h]]：rms_norm、layer_norm、rope、scaled_dot_product_attention，以及你在 X1–X3 用的 metal_kernel。

## ⑧ 一句话总结与选择建议

| 如果你要…… | 更合适的选择 | 原因 |
|---|---|---|
| 训练、大规模在线服务、极致吞吐 | NVIDIA + CUDA / Triton / FlashInfer | Tensor Core 算力、HBM 带宽、NVLink 多卡、成熟的 serving 生态 |
| 本地跑大模型、单用户、对容量敏感 | 高配 Mac + MLX | 统一内存容量大，量化 decode 速度可用，功耗低 |
| 学 GPU 编程的通用概念 | 两边都学 | 线程层次、合并访存、片上内存、归约、tiling、量化，概念完全通用；这套题就是这么设计的 |
| 写一次到处跑 | Triton（NVIDIA / AMD）或 MLX（Apple / CUDA） | 目前还没有同时原生覆盖两边的 kernel DSL |

**核心结论**：两边的编程模型几乎一一对应（grid / block / warp / thread），所以你学的 CUDA 思路可以直接用在 Metal 上。真正的区别来自硬件：Apple 用统一内存换来了容量和省去拷贝，但带宽和矩阵算力低一个数量级，所以软件把重点放在量化、惰性求值和简单的 kernel 结构上；NVIDIA 有 Tensor Core、大片上内存和异步拷贝，所以软件把重点放在复杂的流水线、tile 调度和多卡扩展上。

## 源码与资料索引

**本仓库的对照题**：[C1](c1_elementwise.html) / [T1](t1_add_gelu.html) / [X1](x1_metal_elementwise.html)（逐元素）· [C2](c2_reduce_sum.html) / [T2](t2_softmax.html) / [X2](x2_metal_softmax.html)（归约）· [C4](c4_gemm.html) / [T4](t4_matmul.html) / [X3](x3_metal_matmul.html)（GEMM）· [M3](m3_fp8_block_gemm.html)（FP8 量化）· [L2](l2_paged_decode.html)（decode 是 memory-bound）

**MLX**：[[mlx:docs/src/usage/unified_memory.rst|unified_memory.rst]] · [[mlx:docs/src/usage/lazy_evaluation.rst|lazy_evaluation.rst]] · [[mlx:docs/src/usage/compile.rst|compile.rst]] · [[mlx:mlx/backend/metal/device.cpp@L567|command buffer 提交策略]] · [[mlx:mlx/backend/metal/allocator.cpp@L16|共享内存分配]] · [[mlx:mlx/backend/metal/kernels/quantized.h@L851|量化 qmv]] · [[mlx:mlx/backend/metal/kernels/steel/gemm/mma.h@L453|steel BlockMMA]] · [[mlx:mlx/backend/metal/kernels/sdpa_vector.h|sdpa_vector]] · [[mlx:mlx/backend/cuda/device.cpp|MLX CUDA 后端]]

**文档**：[MLX 文档](https://ml-explore.github.io/mlx/build/html/index.html) · [MLX · Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) · [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) · [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) · [Triton 文档](https://triton-lang.org/main/index.html)
