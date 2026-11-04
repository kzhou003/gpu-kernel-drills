title: X3 讲解：Metal GEMM，从 threadgroup tiling 到 simdgroup_matrix
lede: 在 Apple GPU 上把 C4 再做一遍。第一个 kernel 是 C4 v1 的直译；第二个用 Apple GPU 的矩阵指令 simdgroup_multiply_accumulate，每次算一个 8×8×8 的小矩阵乘。这一页画出 32×32 的 tile 怎么分给 4 个 simdgroup、每个 simdgroup 怎么持有 2×2 个 8×8 累加器，逐行讲 simdgroup_load 的指针和 stride，以及为什么写回时要先经过 threadgroup 内存。

::: callout info 怎么读这一页
对照 [[here:solutions/x3_metal_matmul.py|x3_metal_matmul.py]] 读。前置：[C4 讲解](c4_gemm.html)（v1 的两次 barrier、v2 的外积），[X1 讲解](x1_metal_elementwise.html)（grid 的语义）。第二个 kernel 的思路和 C4 的 v2 完全一样，只是“一个线程的 4×4 外积”变成了“一个 simdgroup 的 2×2 个 8×8 块的外积”。
:::

## ① 先看全貌：两个版本

| | matmul_tiled | matmul_simd |
|---|---|---|
| 对应 | C4 v1（shared memory tiling） | C4 v2 的思路 + 硬件矩阵指令 |
| threadgroup | 16 × 16 = 256 个线程 | 128 个线程 = 4 个 simdgroup |
| 负责 C 的 tile | 16 × 16 | 32 × 32 |
| 内层计算 | 每个线程标量 FMA | 每个 simdgroup 调用 `simdgroup_multiply_accumulate`，8×8×8 |
| 题目页实测（M2，fp32，n = 2048） | 0.39 TFLOP/s | 0.79 TFLOP/s（MLX steel 是 2.62） |

## ② matmul_tiled：C4 v1 的直译

[[here:solutions/x3_metal_matmul.py@L13-L30|_TILED（13–30 行）]]

```cpp
uint2 t = thread_position_in_threadgroup.xy;
uint2 g = threadgroup_position_in_grid.xy;
uint row = g.y * TS + t.y, col = g.x * TS + t.x;
threadgroup float As[16][16], Bs[16][16];
for (uint k0 = 0; k0 < K; k0 += TS) {
    As[t.y][t.x] = (...) ? float(A[row * K + k0 + t.x]) : 0.0f;    // 越界填 0
    Bs[t.y][t.x] = (...) ? float(B[(k0 + t.y) * N + col]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);                // ① 都写完了才能读
    for (uint k = 0; k < TS; ++k) acc += As[t.y][k] * Bs[k][t.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);                // ② 都读完了才能覆盖
}
```

和 C4 的 [matmul_smem_kernel](c4_gemm.html) 逐行对应：`__shared__` → `threadgroup`，`__syncthreads()` → `threadgroup_barrier`，`threadIdx` → `thread_position_in_threadgroup`。两次 barrier 的理由也完全一样。

**grid 是线程数**（[[here:solutions/x3_metal_matmul.py@L95|95 行]]）：`grid = (⌈N/16⌉·16, ⌈M/16⌉·16, 1)`，不是 C4 里的 `(⌈N/32⌉, ⌈M/32⌉)` 个 block。这里特意把线程数补齐到 16 的倍数，让每个 threadgroup 都是完整的 16×16：barrier 要求 threadgroup 里的所有线程都执行到，加载 tile 时也需要每个位置都有线程。

## ③ matmul_simd：4 个 simdgroup，各管四分之一

[[here:solutions/x3_metal_matmul.py@L32-L81|_SIMD（32–81 行）]]

32 × 32 的 C tile 被分成 4 个 16 × 16 的象限，每个 simdgroup 负责一个；每个象限又是 2 × 2 个 8 × 8 的块，对应 4 个累加器 c00、c01、c10、c11：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 70px repeat(4, minmax(70px, 1fr)); max-width: 460px">
<div class="lbl"></div><div class="lbl" style="text-align:center">列 0–7</div><div class="lbl" style="text-align:center">8–15</div><div class="lbl" style="text-align:center">16–23</div><div class="lbl" style="text-align:center">24–31</div>
<div class="lbl">行 0–7</div><div class="c0">sg 0<small>c00</small></div><div class="c0">sg 0<small>c01</small></div><div class="c1">sg 1<small>c00</small></div><div class="c1">sg 1<small>c01</small></div>
<div class="lbl">8–15</div><div class="c0">sg 0<small>c10</small></div><div class="c0">sg 0<small>c11</small></div><div class="c1">sg 1<small>c10</small></div><div class="c1">sg 1<small>c11</small></div>
<div class="lbl">16–23</div><div class="c2">sg 2<small>c00</small></div><div class="c2">sg 2<small>c01</small></div><div class="c3">sg 3<small>c00</small></div><div class="c3">sg 3<small>c01</small></div>
<div class="lbl">24–31</div><div class="c2">sg 2<small>c10</small></div><div class="c2">sg 2<small>c11</small></div><div class="c3">sg 3<small>c10</small></div><div class="c3">sg 3<small>c11</small></div>
</div></div>
<p class="legend">sg_r = sg / 2，sg_c = sg % 2（37 行）：simdgroup 0 在左上，1 在右上，2 在左下，3 在右下。每个格子是一个 8×8 的 simdgroup_float8x8 矩阵。</p>
:::

```cpp
simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);   // c01、c10、c11 同理
simdgroup_float8x8 a0, a1, b0, b1;
```

**simdgroup_float8x8 是什么**：一个 8 × 8 的 float 矩阵，它的 64 个元素**分散存放在 simdgroup 的 32 个线程的寄存器里**（每个线程 2 个）。具体哪个线程拿哪个元素由硬件决定，你不需要知道：只能通过 `simdgroup_load`、`simdgroup_store` 和 `simdgroup_multiply_accumulate` 整体操作它。这和 G1 讲的 layout 是同一个概念，只是这里的 layout 是硬件规定的。

::: callout analogy 四个小组拼一面墙
32×32 的 tile 像一面墙，4 个施工小组（simdgroup）各包一个角（16×16）。每个小组手里有 4 块 8×8 的预制板（c00…c11），整块整块地往上加料（multiply_accumulate），从不一块砖一块砖地砌（标量 FMA）。预制板在组员之间怎么分着扛，由工厂规定（硬件 layout）；你只能整块搬进来（load）、整块加工、整块搬出去（store）。
:::

## ④ 主循环：加载、四次 MMA

### 协作加载

[[here:solutions/x3_metal_matmul.py@L51-L58|51–58 行]]

```cpp
for (uint e = tid; e < BM * BK; e += 128) {        // 32 × 16 = 512 个数，每线程 4 个
    uint r = e / BK, k = e % BK;
    As[e] = (...) ? float(A[(row0 + r) * K + k0 + k]) : 0.0f;
    uint kb = e / BN, c = e % BN;
    Bs[e] = (...) ? float(B[(k0 + kb) * N + col0 + c]) : 0.0f;
}
```

和 C4 v2 的 e 循环一样：128 个线程一起把 A 的 32×16 块和 B 的 16×32 块搬进 threadgroup 内存，快维度连续，越界填 0。这里 As 和 Bs 都是**自然布局**（As 是 `[BM][BK]`，Bs 是 `[BK][BN]`），不像 C4 v2 那样把 A 转置存放，因为 simdgroup_load 可以直接按 stride 读 8×8 块。

### 四次 multiply_accumulate

[[here:solutions/x3_metal_matmul.py@L59-L69|59–69 行]]

```cpp
for (uint kk = 0; kk < BK; kk += 8) {                         // BK = 16，走两步
    simdgroup_load(a0, As + (sg_r * 16 + 0) * BK + kk, BK);   // 我这个象限的上 8 行
    simdgroup_load(a1, As + (sg_r * 16 + 8) * BK + kk, BK);   // 下 8 行
    simdgroup_load(b0, Bs + kk * BN + sg_c * 16 + 0, BN);     // 左 8 列
    simdgroup_load(b1, Bs + kk * BN + sg_c * 16 + 8, BN);     // 右 8 列
    simdgroup_multiply_accumulate(c00, a0, b0, c00);          // c00 += a0 @ b0
    simdgroup_multiply_accumulate(c01, a0, b1, c01);
    simdgroup_multiply_accumulate(c10, a1, b0, c10);
    simdgroup_multiply_accumulate(c11, a1, b1, c11);
}
```

**simdgroup_load(矩阵, 指针, stride)**：指针指向 8×8 块的左上角元素，stride 是“下一行比这一行往后多少个元素”。比如 a1 的左上角在 As 的第 `sg_r·16 + 8` 行第 kk 列，As 每行 BK = 16 个元素，所以指针是 `As + (sg_r·16 + 8)·16 + kk`，stride 是 16。

**这是 C4 v2 外积的块版本**：

::: flow 一个 kk 步：2 + 2 次加载，4 次 MMA
cfg cw=220 nw=170
n a0 0 0 "a0（上 8 行）" "8×8" k=acc
n a1 0 1.3 "a1（下 8 行）" "8×8" k=acc
n b0 1 0 "b0（左 8 列）" "8×8" k=warn
n b1 1 1.3 "b1（右 8 列）" "8×8" k=warn
n c 2.1 0.65 "c00 c01 c10 c11" "2×2 外积，各 += a·b" k=pass w=1.05
e a0 -> c
e a1 -> c
e b0 -> c
e b1 -> c
cap: C4 v2 里一个线程从 shared memory 读 4 + 4 个数，做 4 × 4 = 16 次 FMA。这里一个 simdgroup 从 threadgroup 内存读 2 + 2 个 8×8 块，做 2 × 2 = 4 次 8×8×8 的矩阵乘（每次 512 次 FMA）。每个读进来的块被用 2 次。
:::

题目页提到 MLX 的 steel GEMM 快 3 倍多，主要原因之一就是每个 simdgroup 持有更多累加器（更大的外积，复用度更高），这和 C4 里 TM、TN 从 4 加到 8 是同一个方向。题目页的实验建议正是把 2×2 改成 4×2。

## ⑤ 写回：为什么要经过 Cs

[[here:solutions/x3_metal_matmul.py@L72-L80|72–80 行]]

```cpp
simdgroup_store(c00, Cs + (sg_r * 16 + 0) * BN + sg_c * 16 + 0, BN);   // 4 个累加器先存进 threadgroup 内存
...
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint e = tid; e < BM * BN; e += 128) {
    uint r = e / BN, c = e % BN;
    if (row0 + r < M && col0 + c < N) C[(row0 + r) * N + col0 + c] = T(Cs[e]);   // 逐元素、带 mask 写回
}
```

`simdgroup_store` 一次写一整个 8×8 块，**没有 mask**。在矩阵边缘，一个 8×8 块可能只有一部分落在 C 里面，直接存到 C 就会越界写。所以先把整个 32×32 的结果存进 threadgroup 内存 Cs（这里永远不会越界），再由 128 个线程逐个元素地检查边界、写回 C，顺便做 `T(...)` 的类型转换。

这多用了 32 × 32 × 4 = 4 KB 的 threadgroup 内存和一次 barrier。M、N 都是 32 的倍数时其实可以直接 store 到 C；MLX 的 steel GEMM 就区分了这两种情况：[[mlx:mlx/backend/metal/kernels/steel/gemm/mma.h@L542|store_result]] 和带边界检查的 [[mlx:mlx/backend/metal/kernels/steel/gemm/mma.h@L579|store_result_safe]]。

## ⑥ grid 和 threadgroup

[[here:solutions/x3_metal_matmul.py@L100-L105|matmul_simd host 端（100–105 行）]]

```python
grid = (((N + 31) // 32) * 128, (M + 31) // 32, 1)       # 线程数
threadgroup = (128, 1, 1)
```

x 方向：每个 32 列的 tile 需要 128 个线程，所以 x 方向的线程数是 tile 数 × 128，组成 ⌈N/32⌉ 个 threadgroup。y 方向：threadgroup 的 y 大小是 1，所以 ⌈M/32⌉ 个线程就是 ⌈M/32⌉ 个 threadgroup。kernel 里用 `threadgroup_position_in_grid.xy` 拿到 tile 坐标。

n = 2048 时：x 方向 64 × 128 = 8192 个线程（64 个 threadgroup），y 方向 64 个，一共 4096 个 threadgroup。

## ⑦ 对照 MLX 的 steel GEMM

- [[mlx:mlx/backend/metal/kernels/steel/gemm/mma.h@L453|BlockMMA]]：每个 simdgroup 持有 [[mlx:mlx/backend/metal/kernels/steel/gemm/mma.h@L464-L466|TM × TN 个 8×8 片段]]，相当于这里的 2×2 推广成可配置的大小；底层调用的同样是 [[mlx:mlx/backend/metal/kernels/steel/gemm/mma.h@L205|simdgroup_multiply_accumulate]]，片段类型是 [[mlx:mlx/backend/metal/kernels/steel/gemm/mma.h@L46|metal::simdgroup_matrix]]。
- [[mlx:mlx/backend/metal/kernels/steel/gemm/loader.h@L25|BlockLoader]]：协作加载，每个线程一次读 [[mlx:mlx/backend/metal/kernels/steel/gemm/loader.h@L22-L27|n_reads 个连续元素]]（向量化），对应这里的 e 循环。
- [[mlx:mlx/backend/metal/kernels/steel/gemm/gemm.h@L37|GEMMKernel]]：把 loader 和 mma 组合起来的主循环。
- Mac 上跑 LLM 的主力其实是量化 matmul：decode 时 [[mlx:mlx/backend/metal/kernels/quantized.h@L851|qmv_fast_impl]]（矩阵乘向量，memory-bound），prefill 时 [[mlx:mlx/backend/metal/kernels/quantized.h@L1341|qmm_t_impl]]；反量化在 [[mlx:mlx/backend/metal/kernels/quantized.h@L558|dequantize]] 里和乘加融合。

## ⑧ 常见坑与自测

- **grid 用 threadgroup 数**：X1 讲过的语义问题，GEMM 里尤其容易错，因为 x 和 y 两个方向的换算还不一样（x 要乘 128，y 不用）。
- **simdgroup_load 的 stride 写错**：As 的行宽是 BK，Bs 的行宽是 BN，写反了读到的是错位的块。
- **直接 simdgroup_store 到 C**：边缘 tile 越界写。
- **忘了 `#include <metal_simdgroup_matrix>`**（[[here:solutions/x3_metal_matmul.py@L85|header，85 行]]）：编译找不到 simdgroup_float8x8。

::: answer 自测 1：每个 kk 步，一个 simdgroup 一共做了多少次 FMA？整个 threadgroup 每轮 k0 呢？
每次 multiply_accumulate 是 8 × 8 × 8 = 512 次 FMA，4 次是 2048。每轮 k0 有 2 个 kk 步、4 个 simdgroup：2 × 4 × 2048 = 16384 = 32 × 32 × 16，正好是 BM × BN × BK。
:::

::: answer 自测 2：simdgroup 2 的 a1 和 b0 的指针分别是什么？
sg = 2：sg_r = 1，sg_c = 0。a1 = As + (1·16 + 8)·16 + kk = As + 384 + kk；b0 = Bs + kk·32 + 0。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/x3_metal_matmul.py|solutions/x3_metal_matmul.py]] · [[here:mine/x3_metal_matmul.py|mine/x3_metal_matmul.py]] · [[here:tests/test_x3_metal_matmul.py|tests/test_x3_metal_matmul.py]] · [[here:content/x3.md|content/x3.md]]

**MLX steel**：[[mlx:mlx/backend/metal/kernels/steel/gemm/mma.h@L453|mma.h]] · [[mlx:mlx/backend/metal/kernels/steel/gemm/loader.h@L25|loader.h]] · [[mlx:mlx/backend/metal/kernels/steel/gemm/gemm.h@L37|gemm.h]] · [[mlx:mlx/backend/metal/kernels/quantized.h@L851|quantized.h]] · [[mlx:mlx/backend/metal/kernels/sdpa_vector.h|sdpa_vector.h]]

**文档**：[Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)（SIMD-group matrix functions 一节）· [MLX · Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
