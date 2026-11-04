title: C2 讲解：reduction 的四层结构
lede: 一百万个数怎么加成一个数：线程内寄存器累加、warp 内 shuffle、block 内 shared memory、block 之间 atomicAdd。逐段读参考解，配上每一轮 shuffle 的数值、一次 barrier 的时间线和浮点原子加为什么每次结果不同。

::: callout info 怎么读这一页
先读题目页 C2 的背景，再对照 [[here:solutions/c2_reduce_sum.cu|c2_reduce_sum.cu]] 读这一页。灰色小标签是带行号的源码链接，图里的节点可以悬停。没做完题之前，第 ② 节的代码就是答案，自己决定要不要往下看。
:::

## ① 先看全貌：为什么 reduction 难

逐元素运算（C1）里每个线程各干各的，互不通信。reduction 正好相反：**所有线程的结果最后要汇合到同一个数**。难点在于不同层级之间通信的代价差了几个数量级：

| 层级 | 谁和谁通信 | 用什么 | 代价 |
|---|---|---|---|
| 线程内 | 自己和自己 | 寄存器 `v += x[i]` | 1 条加法指令 |
| warp 内 | 32 个 lane | `__shfl_down_sync`：直接读别的 lane 的寄存器 | 每轮几个周期，不用同步 |
| block 内 | 8 个 warp | shared memory + `__syncthreads()` | 一次全 block 的 barrier |
| block 之间 | 几百个 block | global memory 上的 `atomicAdd` | 最贵，同一地址要排队 |

设计原则只有一条：**在便宜的层级把数据压得越少越好，只把极少量数据交给更贵的层级。**

::: flow 一次 reduce_sum 里，每一层还剩多少个数（RTX 5090，n = 2²⁶）
cfg cw=230 nw=150
n x 0 0 "输入 x" "67,108,864 个 float" k=todo
n t 1 0 "线程内累加" "174,080 个部分和" k=pass
n w 2 0 "warp 内 shuffle" "5,440 个（每 warp 1 个）" k=teal w=1.1
n b 3 0 "block 内" "680 个（每 block 1 个）" k=warn w=1.1
n o 3 1.3 "out" "1 个数" k=acc
e x -> t "~386/线程"
e t -> w "÷32"
e w -> b "÷8"
e b -> o "680 次 atomicAdd" k=todo
cap: grid = min(⌈n/256⌉, 170 × 4) = 680 个 block，共 680 × 256 = 174,080 个线程。数据量每过一层缩小一个数量级，最贵的 atomic 只剩 680 次。
:::

::: callout analogy 公司年终盘点
全公司要数清仓库里一共有多少件货。每个员工先把自己负责的货架数完，记在自己的本子上（寄存器）；同一个小组的 32 个人隔着桌子互相报数，几轮就汇总到组长手里（warp shuffle）；每个部门的 8 个组长到会议室集合，大家都到齐了才开始算（`__syncthreads`），由部门经理汇总；最后每个部门经理给财务打一个电话报数（atomicAdd）。财务只有一部电话，所以电话越少越好：这就是为什么要让每个员工多数几个货架（grid-stride），而不是让每个货架都单独打电话。
:::

## ② 参考解逐段讲

### 第 1 层：线程内，grid-stride 累加

[[here:solutions/c2_reduce_sum.cu@L33-L37|c2_reduce_sum.cu:33–37]]

```cpp
float v = 0.0f;
const int64_t stride = (int64_t)BLOCK * gridDim.x;
for (int64_t i = (int64_t)blockIdx.x * BLOCK + threadIdx.x; i < n; i += stride) {
  v += x[i];
}
```

和 C1 的 bias_gelu 一样是 grid-stride loop：同一轮里相邻线程读相邻地址（合并访存），每个线程跨步读很多个元素，**累加结果一直留在寄存器 `v` 里**。越界的线程一次都不进循环，`v` 保持 0，对总和没有影响，所以后面不需要任何特判。

这里用的是 `BLOCK`（编译期常量）而不是 `blockDim.x`，两者值相同，但常量能让编译器把乘法算好或优化掉。

### 第 2 层：warp 内，5 轮 shuffle

[[here:solutions/c2_reduce_sum.cu@L17-L23|warp_reduce_sum（17–23 行）]]

```cpp
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}
```

`__shfl_down_sync(mask, v, offset)` 的意思是：**“把 lane `i + offset` 的 v 给我”**。数据在寄存器之间直接传，不经过任何内存。三个参数：

- `mask = 0xffffffff`：32 位全 1，表示 32 个 lane 都参与。调用时这些 lane 必须都执行到这一行。
- `v`：每个 lane 拿出来“交换”的值。
- `offset`：往下数几个 lane 去取。如果 `i + offset ≥ 32`（越界），返回自己的 v。

用 8 个 lane 演示（真实的 warp 是 32 个 lane、5 轮，原理一样）。初始值 3, 1, 4, 1, 5, 9, 2, 6，总和 31：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 120px repeat(8, minmax(46px, 1fr)); min-width: 560px">
<div class="lbl"></div><div class="lbl" style="text-align:center">lane 0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div><div class="lbl" style="text-align:center">3</div><div class="lbl" style="text-align:center">4</div><div class="lbl" style="text-align:center">5</div><div class="lbl" style="text-align:center">6</div><div class="lbl" style="text-align:center">7</div>
<div class="lbl">初始</div><div class="c0">3</div><div class="c0">1</div><div class="c0">4</div><div class="c0">1</div><div class="c0">5</div><div class="c0">9</div><div class="c0">2</div><div class="c0">6</div>
<div class="lbl">offset 4</div><div class="c2">8<small>3+5</small></div><div class="c2">10<small>1+9</small></div><div class="c2">6<small>4+2</small></div><div class="c2">7<small>1+6</small></div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div>
<div class="lbl">offset 2</div><div class="c2">14<small>8+6</small></div><div class="c2">17<small>10+7</small></div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div>
<div class="lbl">offset 1</div><div class="c3">31<small>14+17</small></div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div>
</div></div>
<p class="legend">每一轮有效的 lane 减半（绿色）；灰色 lane 也在算，但算出来的值没人用。最后只有 <b>lane 0</b> 手里是完整的和。</p>
:::

为什么只有 lane 0 是对的？因为每一轮都是“下标小的 lane 去拿下标大的 lane 的值”，有效数据一路往 lane 0 方向汇集。函数返回后**只有 lane 0 的 v 有意义**，后面的代码都是 `if (lane == 0)`。

`#pragma unroll` 让编译器把 5 次循环展开成 5 条直线代码，省掉循环计数和跳转。

::: callout analogy 一排同学往前传纸条
一排 32 个同学，每人手里有一个数。老师说“每人回头，把后面第 16 个同学的数加到自己的数上”，再说“后面第 8 个”“第 4 个”“第 2 个”“第 1 个”。五句话之后，第一个同学手里的数就是全排的和。没人离开座位，也不用写黑板（shared memory），所以非常快。条件是所有同学都得听到老师的口令（mask 里的 lane 都必须执行到这一行）。
:::

### 第 3 层：block 内，shared memory + 一次 barrier

[[here:solutions/c2_reduce_sum.cu@L43-L51|c2_reduce_sum.cu:43–51]]

```cpp
const int lane = threadIdx.x & 31;   // 在 warp 里排第几（= % 32）
const int warp = threadIdx.x >> 5;   // 第几个 warp（= / 32）
if (lane == 0) warp_sums[warp] = v;  // 每个 warp 的 lane 0 交一份
__syncthreads();                     // 等 8 个 warp 都交完
if (warp == 0) {
  v = (lane < WARPS) ? warp_sums[lane] : 0.0f;
  v = warp_reduce_sum(v);
  ...
}
```

- `& 31` 和 `>> 5` 就是 `% 32` 和 `/ 32` 的位运算写法（C1 讲过为什么没有 warpIdx）。
- `__shared__ float warp_sums[WARPS]`（第 29 行）：block 内所有线程共享的一小块片上内存，`WARPS = 256 / 32 = 8` 个 float，只有 32 字节。
- **为什么必须 `__syncthreads()`**：8 个 warp 的执行进度不同。warp 0 可能已经跑到了读 `warp_sums` 那一行，而 warp 7 还没写。barrier 保证“所有线程都到达这里”之后才放行，读到的一定是写好的值。
- **第二轮归约**：8 个部分和交给 warp 0 的 lane 0–7，lane 8–31 填 0（不能去读 `warp_sums[8..31]`，那是越界）。然后复用同一个 `warp_reduce_sum`。

::: flow block 内：8 个 warp → 8 个槽位 → warp 0 再归约一次
cfg cw=200 nw=160
n w0 0 0 "warp 0" "shuffle 后 lane 0 = s0" k=teal
n w1 0 1 "warp 1" "lane 0 = s1" k=teal
n w7 0 2 "warp 7" "lane 0 = s7" k=teal
n sm 1.2 1 "warp_sums[8]" "shared memory · 32 字节" k=warn
n bar 2.4 1 "__syncthreads()" "8 个 warp 全部到达" k=todo
n r 3.6 1 "warp 0 读回" "lane 0–7 = s0..s7，其余 0" k=teal w=1.15
n fin 3.6 2.3 "warp_reduce_sum" "lane 0 = 整个 block 的和" k=pass w=1.15
e w0 -> sm "写 [0]"
e w1 -> sm "写 [1]"
e w7 -> sm "写 [7]"
e sm -> bar
e bar -> r "放行"
e r -> fin
cap: 中间的 warp 2–6 省略没画。整个 block 只有一次 barrier。教科书式的 shared memory 树形归约要 log₂256 = 8 次 barrier，这就是 warp shuffle 带来的主要节省。
:::

::: callout analogy 旅行团的集合点
导游（warp 0）要统计 8 个小组买了多少纪念品。每个小组长把数字写在集合点的白板上（shared memory）。导游不能一到集合点就看白板，有的小组还在路上，白板上可能还是旧数字。所以规定：所有人到齐才开始（`__syncthreads`）。一个常见错误是只让小组长去集合点（把 `__syncthreads()` 写在 `if (lane == 0)` 里），其他人不来，导游就会永远等下去：barrier 要求 block 里**每个线程**都执行到它。
:::

### 第 4 层：block 之间，一次 atomicAdd

[[here:solutions/c2_reduce_sum.cu@L54|c2_reduce_sum.cu:54]]

```cpp
if (lane == 0) atomicAdd(out, v);
```

`atomicAdd(addr, v)` 保证“读出 `*addr`、加上 v、写回”这三步不被其他线程打断。几百个 block 同时往同一个 `out` 上加，硬件让它们排队一个一个来，不会丢失更新。所以 host 端必须先把 out 置 0：

[[here:solutions/c2_reduce_sum.cu@L61|torch::zeros({}, x.options())（61 行）]]：`{}` 是 0 维张量（标量），`x.options()` 复制 x 的设备和 dtype。

**如果不用 atomic，直接写 `*out += v` 会怎样？** 两个 block 可能同时读到旧值 10，各自加上自己的部分和，再先后写回，其中一个的结果被覆盖。这叫**丢失更新**，结果偏小，而且每次不一样。

### host 端：为什么是 sms × 4，而 C1 是 sms × 32

[[here:solutions/c2_reduce_sum.cu@L65-L70|c2_reduce_sum.cu:65–70]]

```cpp
constexpr int BLOCK = 256;
const int64_t blocks = std::min<int64_t>((n + BLOCK - 1) / BLOCK, (int64_t)sms * 4);
reduce_sum_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(...);
```

- 每个 block 最后要做一次 atomic，block 越多，同一地址上排队越长。而读带宽只要每个 SM 有几个 block 就能跑满。所以这里比 C1 更激进地封顶：每个 SM 只给 4 个 block，剩下的工作都交给 grid-stride 循环在寄存器里完成。
- `template <int BLOCK>` + `reduce_sum_kernel<BLOCK>`：block 大小在**编译期**已知。shared memory 数组 `warp_sums[WARPS]` 的长度必须是编译期常量，所以要用模板参数，不能用运行时的 `blockDim.x`。`static_assert`（第 27 行）在编译期就把非法配置（不是 32 的倍数、超过 1024）挡掉。

## ③ 浮点原子加：为什么每次结果末位不同

浮点加法**不满足结合律**。用 float32 算：

| 顺序 | 计算 | 结果 |
|---|---|---|
| (1e8 + 1) − 1e8 | 1e8 附近 float 的间隔是 8，+1 被舍掉 | **0** |
| (1e8 − 1e8) + 1 | | **1** |

680 个 block 的 atomicAdd 谁先谁后由硬件调度决定，每次运行顺序不同，所以最后几位可能不同。测试只要求相对误差 1e-4，这不是 bug。

需要逐位可复现时（例如训练的可复现性），把 atomic 换成两趟：

::: flow 确定性的两趟归约
cfg cw=250 nw=160
n k1 0 0 "kernel 1" "每个 block 写 partial[b]" k=acc w=1.1
n p 1 0 "partial[680]" "global memory" k=warn
n k2 2 0 "kernel 2（1 个 block）" "按固定顺序归约" k=acc
n o 3 0 "out" "每次逐位相同" k=pass
e k1 -> p "无 atomic"
e p -> k2 "读"
e k2 -> o
cap: 多一次 kernel launch 和一点点显存读写，换来确定的加法顺序。PyTorch 的通用 reduction（Reduce.cuh）在需要跨 block 时也是先把部分和写进缓冲区，再由最后到达的 block 统一归约，浮点加法本身不用原子操作。
:::

## ④ 常见坑

- **`__syncthreads()` 放进了分支**：`if (lane == 0) { warp_sums[warp] = v; __syncthreads(); }` 只有 8 个线程到达 barrier，其余 248 个永远不来：死锁或未定义行为。
- **第二轮读越界**：`v = warp_sums[lane]` 不加 `lane < WARPS` 判断，lane 8–31 会读到 shared memory 里的垃圾值，并且被加进结果。
- **用了部分 mask**：`__shfl_down_sync` 的 mask 必须包含所有调用它的 lane。`if (warp == 0)` 是整个 warp 一起进去，所以 `0xffffffff` 正确；如果写成 `if (lane < 8) warp_reduce_sum(...)`，mask 就不对了。
- **忘了 out 置 0**：用 `torch::empty` 分配 out，atomicAdd 会加在一个随机值上。
- **以为 lane 1–31 的结果也是对的**：shuffle 之后只有 lane 0 是完整的和。

::: answer 自测 1：BLOCK 改成 1024，warp_sums 有多大？第二轮还用一个 warp 能做完吗？
WARPS = 1024 / 32 = 32，warp_sums 是 32 个 float（128 字节）。第二轮正好 32 个部分和，warp 0 的 32 个 lane 每人一个，一次 `warp_reduce_sum` 做完。这就是为什么 block 上限 1024 = 32 × 32 时，两轮 warp 归约总是够用。
:::

::: answer 自测 2：n = 1000，会启动几个 block？每个线程循环几次？
⌈1000/256⌉ = 4，远小于 sms × 4，所以 4 个 block、1024 个线程。前 1000 个线程各读 1 个元素，最后 24 个线程不进循环，v = 0。最终 4 次 atomicAdd。
:::

## ⑤ 对照真实实现

- **PyTorch 的 BlockReduceSum 和参考解几乎一模一样**：先 `WarpReduceSum`，lane 0 写 `shared[wid]`，`__syncthreads()`，再让第一个 warp 归约一次。见 [[pytorch:aten/src/ATen/native/cuda/block_reduce.cuh@L77-L93|block_reduce.cuh · BlockReduceSum]]，warp 内归约 [[pytorch:aten/src/ATen/native/cuda/block_reduce.cuh@L30-L35|WarpReduceSum]]。PyTorch 里 LayerNorm、RMSNorm 等 kernel 都用它。
- **cuda-samples 的 reduce0 → reduce7**：从最朴素的 shared memory 树形归约，一步步优化到 warp shuffle 和 cooperative groups。适合做对比实验：[[cudasamples:cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu@L104|reduce0（最朴素）]] · [[cudasamples:cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu@L78|__shfl_down_sync 版]] · [[cudasamples:cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu@L516|cg::reduce]]。配套讲义：[Mark Harris · Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)。
- **PyTorch 通用 reduction 框架**：`torch.sum` 背后的 [[pytorch:aten/src/ATen/native/cuda/Reduce.cuh|Reduce.cuh]]，能处理任意维度、任意归约操作。
- **CUB**：[BlockReduce](https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockReduce.html) 是生产代码里最常用的现成实现。

## 源码与资料索引

**本仓库**：[[here:solutions/c2_reduce_sum.cu|solutions/c2_reduce_sum.cu]] · [[here:mine/c2_reduce_sum.cu|mine/c2_reduce_sum.cu]] · [[here:tests/test_c2_reduce_sum.py|tests/test_c2_reduce_sum.py]] · [[here:content/c2.md|content/c2.md]]

**文档**

- [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions) · [Warp Reduce Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-reduce-functions)（`__reduce_add_sync`，整数专用）
- [Thread Block Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#thread-block-synchronization) · [Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory)
- [atomicAdd](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomicadd) · [Atomics 概览](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#atomics)
- [#pragma unroll](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#pragma-unroll)
- [NVIDIA blog · Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
