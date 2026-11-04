title: C4 讲解：GEMM 三连，复用度一级级往上提
lede: naive → shared memory tiling → register tiling，三个 kernel 做的是同一件事：每从慢的存储读一个数，就多做几次乘加。逐段读参考解：每个版本的数据复用怎么算、两次 __syncthreads 各自防什么、线程怎么分到 4×4 的输出块、外积为什么省读取，以及小矩阵为什么反而变慢。

::: callout info 怎么读这一页
对照 [[here:solutions/c4_gemm.cu|c4_gemm.cu]] 读。第 ④ 节的 register tiling 是这题的核心，建议拿纸跟着“线程 (ty=2, tx=5)”的例子把下标算一遍。
:::

## ① 先看全貌：一个数能被用几次

`C[M,N] = A[M,K] @ B[K,N]`，`C[i][j] = Σₖ A[i][k] · B[k][j]`。一共 M·N·K 次乘加（FMA），而 A、B 只有 MK + KN 个数。n = 4096 时，每个输入数在数学上被用了 4096 次。**问题是：每用一次，要从哪里把它拿过来？**

| 版本 | 每次 FMA 平均从显存（global）读几个数 | 从 shared memory 读几个数 | 复用发生在哪 |
|---|---|---|---|
| v0 naive | 2 | 0 | 没有（全靠 L1/L2 缓存碰运气） |
| v1 smem，T = 32 | 2 / 32 = 0.0625 | 2 | 32×32 的 tile 里，每个数被 32 个线程用 |
| v2 regtile，64×64×16，4×4 | 2048 / 65536 ≈ 0.031 | 8 / 16 = 0.5 | 再加一层：寄存器里每个数用 4 次 |

（v2 的 global 一栏：每轮 k0 读 (BM + BN)·BK = 2048 个数，做 BM·BN·BK = 65536 次 FMA。shared 一栏：每个 k 读 TM + TN = 8 个数，做 TM·TN = 16 次 FMA。）

::: flow 三级存储，三级复用
cfg cw=290 nw=160
n g 0 0 "global memory（显存）" "最慢 · 几 TB/s" k=todo w=1.15
n s 1 0 "shared memory" "片上 · 每 SM 几百 KB" k=warn
n r 2 0 "寄存器" "最快 · 每线程几十个" k=pass
n v1 1 1.3 "v1：tile 读一次" "block 内复用 32 次" k=warn
n v2 2 1.3 "v2：片段读一次" "线程内复用 4 次" k=pass
e g -> s "协作加载" k=todo anim
e s -> r "a_frag / b_frag" k=warn anim
e s -> v1 dash
e r -> v2 dash
cap: 每一级优化都是在更快的那一级存储上多用几次同一个数。v1 减少了 global 读，v2 在 v1 的基础上再减少 shared memory 读。
:::

::: callout analogy 写论文查资料
naive：每写一句话就跑一趟图书馆借两本书，查完还回去。smem：小组 32 个人一次把一摞资料复印到会议桌上（shared memory），大家围着桌子查，查完一摞再去复印下一摞。regtile：每个人再把自己要反复用的几页抄到手边的便签上（寄存器），一张便签上的数字能连用四次，连会议桌都不用伸手去够。
:::

## ② v0 naive：每个线程一个输出

[[here:solutions/c4_gemm.cu@L15-L23|matmul_naive_kernel（15–23 行）]]

```cpp
const int row = blockIdx.y * blockDim.y + threadIdx.y;
const int col = blockIdx.x * blockDim.x + threadIdx.x;   // threadIdx.x → 列
if (row >= M || col >= N) return;
float acc = 0.0f;
for (int k = 0; k < K; ++k) acc += A[(int64_t)row * K + k] * B[(int64_t)k * N + col];
C[(int64_t)row * N + col] = acc;
```

为什么让 `threadIdx.x` 走**列**：一个 warp 是同一个 row、32 个连续 col。

- 读 `B[k][col]`：32 个连续 float，合并。
- 读 `A[row][k]`：32 个线程读**同一个地址**，硬件广播一次就够。
- 写 `C[row][col]`：连续，合并。

如果反过来让 `threadIdx.x` 走行，读 A 和写 C 都会变成跨步。问题出在**复用**上：A 的第 row 行被同一行的 N 个线程各读一遍，B 的每一列被 M 个线程各读一遍，全靠缓存兜底。

## ③ v1 shared memory tiling：两次 barrier

[[here:solutions/c4_gemm.cu@L28-L53|matmul_smem_kernel（28–53 行）]]

block = 32×32 个线程，负责 C 的一个 32×32 tile。沿 K 方向每次取 A 的一个 32×32 tile 和 B 的一个 32×32 tile：

::: flow v1 的一轮 k0（共 ⌈K/32⌉ 轮）
cfg cw=210 nw=170
n ld 0 0 "每线程搬 2 个数" "As[ty][tx]、Bs[ty][tx]" k=acc
n b1 1 0 "__syncthreads ①" "tile 全部就位" k=todo
n mm 2 0 "32 次 FMA" "As[ty][k] · Bs[k][tx]" k=pass
n b2 2 1.3 "__syncthreads ②" "大家都用完了" k=todo
n nx 0 1.3 "下一轮 k0 += 32" "覆盖同一块 smem" k=acc
e ld -> b1
e b1 -> mm
e mm -> b2
e b2 -> nx
e nx -> ld "循环"
cap: 第一个 barrier 防“还没写完就读”，第二个 barrier 防“还没读完就被下一轮覆盖”。少了第二个，快的 warp 会提前把下一轮的数据写进 As/Bs，慢的 warp 读到的就是混合了两轮的数据。
:::

逐行：

- [[here:solutions/c4_gemm.cu@L40-L41|40–41 行]]：每个线程搬 A 的 `(row, k0 + tx)` 和 B 的 `(k0 + ty, col)`。越界的位置填 **0**：0 乘任何数都是 0，内层循环就不需要任何边界判断。
- [[here:solutions/c4_gemm.cu@L44-L49|44–49 行]]：内层循环 `acc += As[ty][k] * Bs[k][tx]`。同一个 warp（相同 ty，tx = 0..31）读 `As[ty][k]` 是同一个地址（广播），读 `Bs[k][0..31]` 是 32 个连续 float，落在 32 个不同的 bank，都没有 bank conflict。
- 每个从显存读进来的数，在 shared memory 里被 32 个线程各用一次：global 读取减少到 1/32。

block 用了 32×32 = 1024 个线程，正好是上限（[[here:solutions/c4_gemm.cu@L149|149 行]]）。

## ④ v2 register tiling：每个线程算 4×4

[[here:solutions/c4_gemm.cu@L56-L119|matmul_regtile_kernel（56–119 行）]]

参数（[[here:solutions/c4_gemm.cu@L57-L58|57–58 行]]）：block 负责 C 的 BM×BN = 64×64，K 方向每次 BK = 16，每个线程负责 TM×TN = 4×4 = 16 个输出，所以只要 (64/4)·(64/4) = 256 个线程。

### 线程到输出块的映射

```cpp
const int tx = tid % (BN / TN);   // tid % 16：第几个 4 列宽的竖条
const int ty = tid / (BN / TN);   // tid / 16：第几个 4 行高的横条
```

64×64 的 block tile 被切成 16×16 个 4×4 小块，每个线程一块。例如 **tid = 37**：ty = 2，tx = 5，负责 block tile 里的第 8–11 行、第 20–23 列：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 90px repeat(8, minmax(44px, 1fr)); min-width: 520px">
<div class="lbl"></div><div class="lbl" style="text-align:center">tx 0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div><div class="lbl" style="text-align:center">3</div><div class="lbl" style="text-align:center">4</div><div class="lbl" style="text-align:center">5</div><div class="lbl" style="text-align:center">6</div><div class="lbl" style="text-align:center">… 15</div>
<div class="lbl">ty 0</div><div>0</div><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>…</div>
<div class="lbl">ty 1</div><div>16</div><div>17</div><div>18</div><div>19</div><div>20</div><div>21</div><div>22</div><div>…</div>
<div class="lbl">ty 2</div><div>32</div><div>33</div><div>34</div><div>35</div><div>36</div><div class="c3">37<small>行 8–11 · 列 20–23</small></div><div>38</div><div>…</div>
<div class="lbl">…</div><div>…</div><div>…</div><div>…</div><div>…</div><div>…</div><div>…</div><div>…</div><div>…</div>
</div></div>
<p class="legend">每一格是一个线程（格子里是 tid），对应 C 的一个 4×4 小块。第 ty 行第 tx 列的线程负责 C 的行 4ty…4ty+3、列 4tx…4tx+3（都是相对 block 左上角）。</p>
:::

### 协作加载：e 循环和转置存放的 As

[[here:solutions/c4_gemm.cu@L80-L90|80–90 行]]

```cpp
for (int e = tid; e < BM * BK; e += THREADS) {     // A 的 64×16 = 1024 个数，每线程 4 个
  const int m = e / BK, k = e % BK;                // k 是快维度
  As[k][m] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;   // 注意：转置存
}
```

- **为什么 e 循环里 k 是快维度**：A 是行优先，同一行里 k 连续。相邻线程（相邻 e）读相邻的 k，读显存是合并的。B 的循环里 n 是快维度，同理。
- **为什么存成 `As[k][m]`**（转置）：内层循环需要“固定 k，取连续 4 个 m”（`As[k][ty*4 + i]`）。转置之后这 4 个数在 shared memory 里是挨着的，读起来快，也方便以后改成一次 `float4` 读取。
- **代价**：转置写入时相邻线程 m 相同、k 不同，地址相差 64 个 float，正好落在同一个 bank 上（16 路冲突）。题目页的“测完之后”建议用 padding 或 `float4` 加载去优化它。

### 外积：16 次 FMA 只读 8 个数

[[here:solutions/c4_gemm.cu@L93-L105|93–105 行]]

```cpp
for (int k = 0; k < BK; ++k) {
  for (int i = 0; i < TM; ++i) a_frag[i] = As[k][ty * TM + i];   // A 的一列里的 4 个数
  for (int j = 0; j < TN; ++j) b_frag[j] = Bs[k][tx * TN + j];   // B 的一行里的 4 个数
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) acc[i][j] += a_frag[i] * b_frag[j];   // 4×4 外积
}
```

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 70px repeat(4, minmax(64px, 1fr)); max-width: 440px">
<div class="lbl"></div><div class="c1">b0</div><div class="c1">b1</div><div class="c1">b2</div><div class="c1">b3</div>
<div class="c0">a0</div><div class="c2">a0·b0</div><div class="c2">a0·b1</div><div class="c2">a0·b2</div><div class="c2">a0·b3</div>
<div class="c0">a1</div><div class="c2">a1·b0</div><div class="c2">a1·b1</div><div class="c2">a1·b2</div><div class="c2">a1·b3</div>
<div class="c0">a2</div><div class="c2">a2·b0</div><div class="c2">a2·b1</div><div class="c2">a2·b2</div><div class="c2">a2·b3</div>
<div class="c0">a3</div><div class="c2">a3·b0</div><div class="c2">a3·b1</div><div class="c2">a3·b2</div><div class="c2">a3·b3</div>
</div></div>
<p class="legend">从 shared memory 读 4 + 4 = 8 个数进寄存器，做 4 × 4 = 16 次 FMA，每个数被用 4 次。16 个格子分别累加进 acc[i][j]，K 方向全部走完后 acc 就是这 16 个 C 元素的最终值。</p>
:::

`acc[TM][TN]`、`a_frag`、`b_frag` 都是固定大小的局部数组，配合 `#pragma unroll` 全部展开后下标是常量，编译器就能把它们全部放进寄存器（16 + 4 + 4 = 24 个 float）。如果下标不是编译期常量，数组会被放到 local memory（实际在显存里），性能会断崖式下跌。

::: callout analogy 九九乘法表
外积就是填一张乘法表：上面一排写 4 个数，左边一列写 4 个数，表里 16 个格子各是行头乘列头。你只需要记住 8 个数，就能填满 16 个格子。表越大（TM、TN 越大），每记一个数能填的格子越多，但手里要同时记的格子（寄存器）也越多，记不住就得写到纸上（寄存器溢出到 local memory）。
:::

### 写回和 `__launch_bounds__`

- [[here:solutions/c4_gemm.cu@L109-L118|109–118 行]]：每个线程把自己的 16 个结果写回 C，逐个检查 `gm < M`、`gn < N`。
- [[here:solutions/c4_gemm.cu@L60|60 行]] `__launch_bounds__(THREADS)`：告诉编译器“这个 kernel 每个 block 最多 256 个线程”。编译器据此决定每个线程最多能用多少个寄存器，保证 256 个线程放得下。不写的话编译器按最坏情况（1024 个线程）保守估计。
- host 端（[[here:solutions/c4_gemm.cu@L162-L163|162–163 行]]）：block 是**一维**的 256 个线程，二维映射是 kernel 里自己用 `tid % 16`、`tid / 16` 算的。

## ⑤ 为什么 n = 512 时 regtile 反而最慢

题目页的实测（RTX 5090，TFLOP/s）：n = 512 时 naive 5.2、smem 5.2、**regtile 2.5**；n = 4096 时 regtile 33.0，是 smem 的 3.6 倍。

n = 512 时 regtile 的 grid 是 (512/64)² = **64 个 block**，而 5090 有 170 个 SM：106 个 SM 完全闲着。smem 版本是 (512/32)² = 256 个 block，至少每个 SM 都有活干。

::: flow 小矩阵上 block 太少
cfg cw=240 nw=180
n big 0 0 "64×64 tile，n = 512" "64 个 block" k=todo
n sm 1 0 "170 个 SM" "106 个空闲" k=todo
n small 0 1.3 "32×32 tile，n = 512" "256 个 block" k=pass
n sm2 1 1.3 "170 个 SM" "全部有活" k=pass
e big -> sm "铺不满" k=todo
e small -> sm2 "铺满" k=pass
cap: tile 越大复用越高，但 block 数越少。矩阵小的时候，block 数不足以填满 GPU，复用再高也没用。实际的库（cuBLAS、Triton autotune）会按 shape 选不同的 tile，或者用 split-K 把 K 方向也切开来增加并行度。
:::

::: callout analogy 出租车车队
公司有 170 辆出租车（SM）。大订单（大 tile）一单能拉很多人，效率高，但今天只来了 64 个大订单，106 辆车停在车库里。把订单拆小一点（小 tile），每单效率低一些，但所有车都跑起来了，总运量反而更大。
:::

## ⑥ 常见坑与自测

- **只写了第一个 `__syncthreads()`**：大矩阵上偶尔出错，而且很难复现。
- **边界不填 0**：K 不是 32 的倍数时，最后一个 tile 里读进了越界的垃圾值，加进了结果。
- **acc 用了运行时下标**：例如 `acc[i][j]` 的循环没法展开，数组被放到 local memory。用 `ncu` 看 `local load/store` 是否为 0。
- **`threadIdx.x` 走行**：读写都不合并，比 naive 还慢。

::: answer 自测 1：v1 里一个 block 每轮 k0 从显存读多少个数？做多少次 FMA？每个数被用了几次？
每轮读 A、B 各 32 × 32，共 2048 个数；做 32 × 32 × 32 = 32768 次 FMA。每次 FMA 用 2 个操作数，共 65536 次“使用”，除以 2048 次读取，每个读进来的数被用了 **32 次**。换个说法，每次 FMA 平均读 2048 / 32768 = 0.0625 = 2/32 个数，和第 ① 节表里一致。
:::

::: answer 自测 2：TM = TN = 8 会怎样？
每个线程 64 个 acc，加上 16 个 fragment，寄存器至少 80 个 float。每次 k 读 16 个数做 64 次 FMA，shared memory 读取再减半。但 block 里只需要 (64/8)² = 64 个线程，或者把 BM、BN 加大到 128。寄存器占用高会降低 occupancy，太高还会溢出。这是 GEMM 调参的核心权衡，Triton 的 autotune（T4）就是在搜这些参数。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/c4_gemm.cu|solutions/c4_gemm.cu]] · [[here:mine/c4_gemm.cu|mine/c4_gemm.cu]] · [[here:tests/test_c4_gemm.py|tests/test_c4_gemm.py]] · [[here:content/c4.md|content/c4.md]]

**对照实现**

- cuda-samples 的 shared memory 版（和 v1 同构）：[[cudasamples:cpp/0_Introduction/matrixMul/matrixMul.cu@L62|MatrixMulCUDA]]，两次 barrier 在 [[cudasamples:cpp/0_Introduction/matrixMul/matrixMul.cu@L109|109 行]] 和 [[cudasamples:cpp/0_Introduction/matrixMul/matrixMul.cu@L123|123 行]]
- CuTe 版本的同一个算法：[[cutlass:examples/cute/tutorial/sgemm_1.cu@L52|gemm_device]]，循环体是 [[cutlass:examples/cute/tutorial/sgemm_1.cu@L199-L214|copy → __syncthreads → gemm]]，G2 会讲 CuTe 的布局代数
- Tensor Core：[[cudasamples:cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu@L423|wmma::mma_sync]] 一次算 16×16×16；FlashInfer 的 PTX 封装 [[flashinfer:include/flashinfer/mma.cuh|mma.cuh]]

**文档与文章**

- [Simon Boehm · How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)：从 naive 一路到接近 cuBLAS，每一步都有 profile，和这题的 v0 → v2 正好对应
- [Programming Guide · __launch_bounds__](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#launch-bounds) · [Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory) · [Bank Conflicts](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-bank-conflicts) · [WMMA](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#wmma)
