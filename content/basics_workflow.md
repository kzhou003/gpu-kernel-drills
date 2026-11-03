lede: 每个 kernel 都走同一个循环：reference → 最简单的正确实现 → 测试 → benchmark → profile → 一次只改一个因素。
# 测试、计时与 profiling

## 正确性：和 PyTorch 对拍

```python
torch.testing.assert_close(out, ref, rtol=..., atol=...)
```

- **reference 用更高精度算**：fp16 kernel 和 fp32（甚至 fp64）的 reference 比，而不是和另一个 fp16 实现比。
- **容差要和数值实际误差对应**：fp32 逐元素运算 `1e-5`；fp16 输出 `1e-2` 量级（fp16 只有约 3 位有效数字）；纯拷贝（转置、写 cache）要求**完全相等**。
- **边界 shape 一定要测**：1、block 大小 ±1、非 2 的幂、远大于 block、空张量、非连续（转置 / 切片后的 view）。大多数 bug 出在最后一个不完整的 block。
- CUDA 的越界读写不一定当场崩溃：`compute-sanitizer --tool memcheck python -m pytest tests/test_c3_transpose.py` 能抓出来。

## 计时：别用 `time.time()` 包一次调用

GPU 调用是**异步**的：Python 只是把 kernel 放进队列就返回了。正确的做法：

```python
import triton.testing
ms = triton.testing.do_bench(lambda: my_kernel(x), warmup=25, rep=100, return_mode="median")  # 毫秒
```

`do_bench` 会先 warmup（编译、autotune、缓存预热），每次测量前清空 L2，用 CUDA event 计时；默认返回平均值，传 `return_mode="median"` 取中位数，不容易被偶发的慢样本带偏。自己写的话要点相同：

```python
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for _ in range(10): fn()                 # warmup
start.record()
for _ in range(100): fn()
end.record(); torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
```

报告结果时要附上 GPU 型号、dtype、shape，并换算成 **GB/s 或 TFLOP/s**。小 shape 下几微秒的时间主要是 launch 开销，此时比的是 Python / launch 路径，不是 kernel 本身。

## Profiling：找到真正的瓶颈

| 工具 | 回答的问题 | 命令 |
|---|---|---|
| **Nsight Compute**（`ncu`） | 单个 kernel 为什么慢：带宽用了多少、occupancy、warp 在等什么、bank conflict、寄存器数 | `ncu --set full -k regex:softmax -o prof python bench.py t2` |
| **Nsight Systems**（`nsys`） | 时间线：kernel 之间有没有空隙、CPU 是不是瓶颈、launch 了多少个 kernel | `nsys profile -o timeline python bench.py t2` |
| **PyTorch Profiler** | 框架层哪个算子花时间 | `torch.profiler.profile(...)` |
| **compute-sanitizer** | 越界、竞争条件、未初始化 shared memory | `compute-sanitizer --tool racecheck ...` |

`ncu` 报告里优先看三个数字：

1. **Memory Throughput（% of peak）**：memory-bound 的 kernel 应该在 80% 以上。
2. **Compute (SM) Throughput**：compute-bound 的 kernel 看这里，以及 Tensor Core 利用率。
3. **Warp State Statistics**：warp 大部分时间停在哪。`Long Scoreboard` 表示在等 global memory，`Barrier` 表示在等 `__syncthreads`，`MIO Throttle` 表示 shared memory 指令排队。

::: callout ok 一次只改一个因素
每次优化只改一件事（block 大小、num_warps、加 padding、加 tiling…），测一次，记下数字和原因。做完一题，用三五行写下“瓶颈是什么、改了什么、提升多少、离上限还差多少”。这份记录就是原计划里每题的 README.md。
:::

## 图解与类比

::: flow 每个 kernel 都走的开发循环
n ref 0 0 "PyTorch reference" "高精度" k=pass
n naive 1 0 "最简实现" "先求正确" k=acc
n test 2 0 "测试" "边界 shape / dtype" k=acc
n bench 3 0 "benchmark" "GB/s · TFLOP/s" k=warn
n prof 3 1 "profile" "ncu / nsys" k=warn
n opt 2 1 "只改一个因素" "tile / warps / 融合" k=vio
n doc 1 1 "记录" "数字 + 原因" k=teal
e ref -> naive
e naive -> test
e test -> bench
e bench -> prof
e prof -> opt
e opt -> test "回归测试" anim
e opt -> doc dash
:::
::: callout analogy 调音台一次只推一个推子
同时动三个推子，声音变好了也说不清是哪个起了作用。kernel 调优也一样：每次只改一件事（block 大小、num_warps、padding……），测一次，记下来。
:::

## 源码与资料

**文档**：[Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) · [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) · [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/testing.html)

**源码**：[[triton:python/triton/testing.py@do_bench]] · [[pytorch:torch/testing/_comparison.py@assert_close]] · [[flashinfer:flashinfer/testing/utils.py@bench_gpu_time]]（FlashInfer 的 CUPTI 计时）

