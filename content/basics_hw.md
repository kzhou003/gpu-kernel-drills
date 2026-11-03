lede: 写 kernel 之前必须在脑子里有的那张图：线程怎么组织、数据住在哪、一个 kernel 的速度上限由什么决定。
# GPU 执行模型与内存层次

## 线程的层级：grid → block → warp → thread

| 层级 | 是什么 | 关键数字（NVIDIA） |
|---|---|---|
| **thread** | 最小执行单位，有自己的寄存器 | 每线程最多 255 个 32-bit 寄存器 |
| **warp** | 32 个线程，**同一条指令同时执行**（SIMT） | 永远是 32 |
| **block**（CUDA）/ **program**（Triton） | 一组 warp，被调度到**同一个 SM** 上，可以共享 shared memory、用 `__syncthreads()` 同步 | 最多 1024 线程 |
| **grid** | 一次 kernel launch 的所有 block，彼此之间**没有**同步手段（除了结束） | `gridDim.x` 最大 2³¹−1 |

**SM**（Streaming Multiprocessor）是硬件上的“核”：A100 有 108 个，H100 SXM 有 132 个，T4 有 40 个，L4 有 58 个。一个 SM 同时驻留多个 block，每个时钟周期从就绪的 warp 里挑一个发射指令。某个 warp 在等内存时，其他 warp 顶上，这就是 GPU 隐藏延迟的方式。所以需要**足够多的 warp 在飞**：这个比例叫 **occupancy**（驻留 warp 数 / SM 上限）。寄存器和 shared memory 用得越多，能同时驻留的 block 越少。

::: callout info warp divergence
同一个 warp 里的线程走了 `if` 的不同分支时，两个分支会**串行**执行，各自屏蔽掉另一半线程。边界检查（`if (i < n)`）只影响最后一个 warp，代价很小；但依赖数据的分支（比如按 token 类型走不同代码）可能让效率减半。
:::

## 内存层次：越近越快、越小

| 存储 | 作用域 | 容量（量级） | 带宽 / 延迟（量级） |
|---|---|---|---|
| 寄存器 | 线程私有 | 每 SM 64K × 32-bit = 256 KB | 最快，约 1 cycle |
| shared memory / L1 | block 内共享 | 每 SM 约 100–228 KB（A100 164 KB，H100 228 KB） | 约 20–30 cycle |
| L2 cache | 全 GPU | 几十 MB（A100 40 MB，H100 50 MB） | |
| HBM / GDDR（global memory） | 全 GPU，kernel 之间持久 | 16–80 GB | 约 400–800 cycle；带宽见下表 |

### 合并访存（coalescing）

global memory 以 **32 字节 sector** 为单位传输。一个 warp 发出一条 load 指令时，硬件把 32 个线程的地址合并成尽量少的 sector：

- 32 个线程读 32 个**连续**的 float（128 字节）：4 个 sector，100% 有效。
- 32 个线程各读相隔 4 KB 的 float：32 个 sector，每个只用 4 字节，有效率 12.5%。

所以写 kernel 的第一条规则：**让相邻线程（`threadIdx.x` 相邻）访问相邻地址**。C3（转置）整道题都在处理这件事。

### shared memory 的 bank

shared memory 分成 **32 个 bank**，每个 bank 每周期服务一个 4 字节字。地址 `a`（按 4 字节计）落在 bank `a % 32`。同一 warp 里多个线程访问**同一 bank 的不同地址**时会串行化（bank conflict）；访问**同一个地址**则是广播，没有冲突。经典的规避手段是给二维数组加一列 padding：`tile[32][33]`。

## 性能上限：roofline

一个 kernel 要么被**内存带宽**卡住，要么被**算力**卡住。判断看**算术强度**（arithmetic intensity）：

```
AI = FLOPs / 从 global memory 读写的字节数
```

和硬件的“拐点” `峰值 FLOP/s ÷ 峰值带宽` 比较：AI 低于拐点就是 memory-bound，优化方向是**少读写字节**（融合、复用、低精度）；高于拐点是 compute-bound，优化方向是**用好 Tensor Core**、减少非矩阵运算。

| GPU | 显存带宽 | FP16/BF16 Tensor Core（稠密） | FP32（非 Tensor Core） | 拐点（FP16） |
|---|---|---|---|---|
| T4 | ~320 GB/s | ~65 TFLOP/s | ~8 TFLOP/s | ~200 FLOP/B |
| L4 | ~300 GB/s | ~121 TFLOP/s | ~30 TFLOP/s | ~400 FLOP/B |
| A100 80GB SXM | ~2.0 TB/s | ~312 TFLOP/s | ~19.5 TFLOP/s | ~156 FLOP/B |
| H100 SXM | ~3.35 TB/s | ~990 TFLOP/s | ~67 TFLOP/s | ~295 FLOP/B |
| RTX 5090 | ~1.79 TB/s | ~210 TFLOP/s（FP32 累加；实测 cuBLAS ≈ 218） | ~105 TFLOP/s | ~117 FLOP/B |

（厂商公开峰值的约数，实际能达到的带宽一般是峰值的 80–90%。）

几个常见 kernel 的 AI，记住量级就够了：

| kernel | AI（FLOP / byte） | 结论 |
|---|---|---|
| vector add（fp32） | 1 / 12 | 永远 memory-bound：衡量标准是 **GB/s** |
| softmax、LayerNorm | ~1 | memory-bound：理想是“读一次、写一次” |
| GEMM（n×n×n，fp16） | ~n / 3 | n 较大时 compute-bound：衡量标准是 **TFLOP/s** |
| decode attention（每 token） | ~1（GQA 时 ×group） | memory-bound：瓶颈是读 KV cache |
| prefill attention | ~S | compute-bound |

::: callout ok 衡量一个 kernel 好不好
memory-bound 的 kernel 算 **有效带宽 = 必须读写的字节 / 时间**，和峰值带宽比；compute-bound 的算 **TFLOP/s**，和峰值算力比。只说“比 PyTorch 快 1.3 倍”是不够的：要能说出离硬件上限还差多少。`bench.py` 对每题都打印这两种数字之一。
:::
