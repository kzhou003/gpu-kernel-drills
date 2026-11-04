title: Tensor Core 与新指令集：从 WMMA 到 wgmma、TMA、tcgen05
lede: 面试题和附录里提到的“用 WMMA 改写 C4”“读 CuTe 的 MMA atom / TiledMMA”“读 FlashAttention-3/4 和 DeepGEMM 的主循环”，背后是同一条主线：每一代 GPU 都把更多的活从线程手里拿走，交给专用硬件。这一页按 Volta → Ampere → Hopper → Blackwell 的顺序讲清楚每一代新增了什么指令、解决了什么问题、代码里怎么用，并把每一步都链接到 cuda-samples、CUTLASS/CuTe、FA3、FA4 和 DeepGEMM 里对应的源码行。

::: callout info 怎么读这一页
前置：[C4 讲解](c4_gemm.html)（三级复用、外积）、[T4 讲解](t4_matmul.html)（tl.dot、num_stages）、[CuTe 专题](cute_overview.html)（Layout、TV layout）。本仓库实测用的 RTX 5090 是 **sm_120**（消费级 Blackwell），它有 Tensor Core 和 warp 级的 mma 指令，但**没有** Hopper 的 wgmma，也没有数据中心 Blackwell（sm_100）的 tcgen05 和 Tensor Memory。所以第 ④⑤ 节的代码要在 H100 / B200 上才能运行，这也是面试题里写“有 H100 / B200 时”的原因。
:::

## ① 先看全貌：每一代都在把活从线程手里拿走

**为什么要专门的矩阵指令**：C4 的 v2 里，一个线程每条 FMA 指令做 1 次乘加。Ampere 的 `mma.sync.m16n8k16` 是一个 warp 一起发的一条指令，一次做 16 × 8 × 16 = **2048 次乘加**。指令数少了两个数量级，取指、调度、寄存器读写的开销都随之摊薄，这就是 Tensor Core 快的根本原因。本仓库的实测：RTX 5090 上 FP32 CUDA core 的 cuBLAS 是 65 TFLOP/s（C4），FP16 Tensor Core 的 Triton GEMM 是 218 TFLOP/s（T4）。

但 Tensor Core 算得越快，**喂数据**就越成问题。每一代新增的指令，几乎都是为了让数据跟上：

::: flow 四代 GPU，每一代把哪件活交给了硬件
cfg cw=250 nw=225
n v 0 0 "Volta / Turing（sm_70/75）" "WMMA：16×16×16，warp 级" k=acc w=1.05
n a 1 0 "Ampere（sm_80）" "mma.sync · ldmatrix · cp.async" k=vio w=1.05
n h 2 0 "Hopper（sm_90a）" "wgmma · TMA · mbarrier · cluster" k=warn w=1.05
n b 3 0 "Blackwell（sm_100a）" "tcgen05 · Tensor Memory · 2-CTA" k=pass w=1.05
n v2 0 1.3 "交给硬件：矩阵乘" "数据怎么搬仍是线程的事" k=acc w=1.05
n a2 1 1.3 "交给硬件：全局 → smem 拷贝" "不经过寄存器、异步" k=vio w=1.05
n h2 2 1.3 "交给硬件：整个 tile 的地址计算" "一个线程发起整块拷贝" k=warn w=1.05
n b2 3 1.3 "交给硬件：累加器的存放" "不占寄存器，单线程发 MMA" k=pass w=1.05
e v -> a
e a -> h
e h -> b
e v -> v2 dash
e a -> a2 dash
e h -> h2 dash
e b -> b2 dash
cap: 读 FA3、FA4、DeepGEMM 之前最需要建立的直觉：越新的架构，线程自己干的活越少，越多的是“发起一个异步操作，然后等一个信号”。kernel 的写法也就从“每个线程一步步算”变成了“几组线程各司其职、用 barrier 互相通知”（warp specialization）。
:::

| | 谁发起矩阵乘 | 操作数从哪来 | 累加器放哪 | 谁搬数据 | 同步方式 |
|---|---|---|---|---|---|
| Volta WMMA | 一个 warp | 寄存器（fragment） | 寄存器 | 线程 ld / st | `__syncthreads` |
| Ampere mma.sync | 一个 warp | 寄存器（`ldmatrix` 从 smem 装进来） | 寄存器 | `cp.async`（异步） | `cp.async.wait_group` |
| Hopper wgmma | 一个 **warpgroup**（4 个 warp） | **shared memory**（描述符），A 也可以在寄存器 | 寄存器 | **TMA**（一个线程发起整块） | **mbarrier**（按字节数完成） |
| Blackwell tcgen05 | **一个线程** | shared memory 或 Tensor Memory | **Tensor Memory** | TMA（可跨两个 CTA） | mbarrier |

::: callout analogy 盖楼的工具在升级
最早是每个工人一块块搬砖、一块块砌（CUDA core 的 FMA）。Volta 给每个小组发了一台小型砌砖机（WMMA），砌得快多了，但砖还得人搬。Ampere 加了传送带（cp.async），砖可以自己从仓库送到工地旁边，工人不用来回跑。Hopper 换成了塔吊（TMA）：一个人按一下按钮，一整托盘砖就吊到指定位置；砌砖机也变大了，四个小组合用一台（wgmma）。Blackwell 的砌砖机自带记录板（Tensor Memory），工人连“砌到哪了”都不用记在自己脑子里（寄存器），只要一个人按启动键就行（单线程发 MMA）。工地上的人越来越少地动手，越来越多地在“调度”。
:::

## ② 第一步：WMMA，用 Tensor Core 改写 C4

`nvcuda::wmma`（`#include <mma.h>`）是 CUDA 9 起的 C++ 接口，最容易上手。它的核心是 **fragment**：一个 warp 共同持有的一小块矩阵，具体每个线程拿哪几个元素是不公开的（opaque）。

::: flow 一个 warp 用 WMMA 算一个 16×16 的 C tile
cfg cw=225 nw=180
n la 0 0 "load_matrix_sync(a_frag)" "A 的 16×16（half）" k=acc w=1.05
n lb 0 1.3 "load_matrix_sync(b_frag)" "B 的 16×16（half）" k=acc w=1.05
n mm 1.15 0.65 "mma_sync(acc, a, b, acc)" "16×16×16，fp32 累加" k=pass w=1.05
n lp 2.3 0.65 "沿 K 循环" "每次 K += 16" k=warn
n st 2.3 1.9 "store_matrix_sync(acc)" "写回 C" k=vio
e la -> mm
e lb -> mm
e mm -> lp
e lp -> st
cap: 和 C4 v1 对照：C4 里一个线程算 C 的 1 个元素（v1）或 4×4 个（v2），这里一个 warp 算 C 的 16×16 = 256 个，乘加全部交给 Tensor Core。
:::

cuda-samples 里最简单的版本 [[cudasamples:cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu@L389-L440|simple_wmma_gemm]]，核心就是这几行：

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
wmma::fill_fragment(acc_frag, 0.0f);
for (int k = 0; k < K; k += 16) {
  wmma::load_matrix_sync(a_frag, a + k + aRow * lda, lda);
  wmma::load_matrix_sync(b_frag, b + k + bCol * ldb, ldb);
  wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
}
wmma::store_matrix_sync(c + cCol + cRow * ldc, acc_frag, ldc, wmma::mem_row_major);
```

**改写 C4 的路线**（面试题里的第一步）：

1. 先照抄上面这个版本：直接从全局内存 load fragment，没有 shared memory。它能工作，但每个 fragment 都从显存读，复用很差。
2. 把 C4 v1 的 shared memory tiling 加回来：一个 block 把 A、B 的 tile 读进 shared memory，多个 warp 各自从 shared memory load fragment。cuda-samples 的完整版 [[cudasamples:cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu@L197|compute_gemm]] 就是这样，[[cudasamples:cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu@L318-L326|每个 warp 持有多个 a、b fragment]]，相当于 C4 v2 的“寄存器 tile”。
3. 和 cuBLAS 的 FP16 GEMM 对比。参考上限：本仓库 T4 在 5090 上实测 cuBLAS FP16 约 218 TFLOP/s。

**WMMA 的局限**：fragment 的寄存器布局不公开，所以你没法自己决定每个线程持有哪些元素，也就没法和 softmax、量化之类的计算精细融合；shared memory 到 fragment 的加载方式也是固定的。这正是下一代写法（直接用 `mma.sync` PTX，或者用 CuTe）要解决的问题。文档：[Programming Guide · WMMA](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#wmma)。

## ③ Ampere：mma.sync、ldmatrix、cp.async，以及 CuTe 的 MMA Atom

### 三条新指令

| 指令 | 做什么 | CuTe 里的封装 |
|---|---|---|
| `mma.sync.aligned.m16n8k16` | 一个 warp 做 16×8×16 的矩阵乘，每个线程持有的元素**有公开的布局** | [[cutlass:include/cute/arch/mma_sm80.hpp@L158-L173|SM80_16x8x16_F32F16F16F32_TN]] |
| `ldmatrix`（sm_75 起） | 一个 warp 从 shared memory 读 8×8 的小块，直接放成 mma 要的寄存器布局 | [[cutlass:include/cute/arch/copy_sm75.hpp@L121|SM75_U32x4_LDSM_N]] |
| `cp.async`（sm_80 起） | 全局内存 → shared memory 的**异步**拷贝，不经过寄存器 | [[cutlass:include/cute/arch/copy_sm80.hpp@L47-L62|SM80_CP_ASYNC_CACHEALWAYS]]，配合 [[cutlass:include/cute/arch/copy_sm80.hpp@L164-L177|cp_async_fence / cp_async_wait]] |

::: flow Ampere 上一个 K 块的数据流
cfg cw=245 nw=170
n g 0 0.65 "全局内存" "A、B 的 tile" k=todo
n s 1.1 0.65 "shared memory" "多 stage 环形缓冲" k=warn
n r 2.2 0.65 "寄存器" "mma 要求的布局" k=acc
n t 3.3 0.65 "Tensor Core" "mma.sync m16n8k16" k=pass
e g -> s "cp.async"
e s -> r "ldmatrix"
e r -> t "mma.sync"
cap: cp.async 让“搬下一块”和“算这一块”可以同时进行，这就是 T4 里 num_stages 做的事（软件流水线）。ldmatrix 省掉了线程自己从 shared memory 一个个读、再拼成 mma 布局的指令。
:::

### MMA Atom：把一条指令的布局写成 layout

`mma.sync` 的难点在于：32 个线程各自持有 A、B、C 的哪几个元素，是硬件规定好的。CuTe 用**TV layout**（(线程号, 值号) → (行, 列)）把这个规定写下来：[[cutlass:include/cute/atom/mma_traits_sm80.hpp@L109|MMA_Traits SM80_16x8x16]] 里的 ALayout、BLayout、CLayout，例如 C 的布局是 [[cutlass:include/cute/atom/mma_traits_sm80.hpp@L53|SM80_16x8_Row]]。这就是 [CuTe 专题](cute_overview.html) 第 ⑦ 节说的“硬件也用 Layout 说话”，也是 G1 里你写的“lane 持有哪几个元素”的计算。

### TiledMMA：用小印章拼大印章

一条 mma 只算 16×8 的 C。`make_tiled_mma` 把这个 Atom 在多个 warp 上、在 M / N 方向上重复，拼成一个大的 tile：[[cutlass:examples/cute/tutorial/sgemm_sm80.cu@L375|make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN, …)]]。之后的代码只调用 `gemm(mma, A, B, C)`（[[cutlass:examples/cute/tutorial/sgemm_sm80.cu@L298|298 行]]），每个线程该算哪些、该持有哪些，全部由 layout 推出来。同一份 kernel 换成 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@L478|UniversalFMA]]（普通 FMA）也能跑，只是换了一个 Atom。

::: callout analogy 印章和拼版
一条 mma 指令像一枚 16×8 的小印章，盖一下出一小块图案，而且印章上每个格子对应哪个工人是出厂就定好的（TV layout）。TiledMMA 是把很多枚小印章按规则拼成一块大印版，一次盖出一整个 tile。你只需要说“用这种印章、横着拼几个、竖着拼几个”，每个工人负责哪个格子由 CuTe 自动推出来。换一代硬件，就是换一种印章，拼版的代码不用改。
:::

### 多级流水线

[[cutlass:examples/cute/tutorial/sgemm_sm80.cu@L135-L144|K_PIPE_MAX 个 stage]]：先发出前几个 stage 的 cp.async，主循环里 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@L262-L298|等最老的一个 stage 到齐（cp_async_wait<K_PIPE_MAX−2>），算它，同时发出下一个 stage 的拷贝]]。

::: raw
<style>
.gantt { margin: 10px 0; font-size: 12.5px; }
.gantt .row { display: grid; grid-template-columns: 120px 1fr; gap: 8px; align-items: center; margin: 3px 0; }
.gantt .bar { position: relative; height: 22px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 4px; }
.gantt .s { position: absolute; top: 2px; height: 16px; border-radius: 3px; font: 10.5px/16px var(--mono); text-align: center; color: var(--text); overflow: hidden; white-space: nowrap; }
.gantt .ld { background: var(--f-warn-bg); border: 1px solid var(--f-warn); }
.gantt .mm { background: var(--f-pass-bg); border: 1px solid var(--f-pass); }
.gantt .sm { background: var(--f-vio-bg); border: 1px solid var(--f-vio); }
.gantt .h { font-weight: 600; margin: 10px 0 4px; }
.gantt .t { text-align: right; color: var(--muted); font-family: var(--mono); }
</style>
<div class="gantt">
<div class="h">没有流水线：搬一块、算一块（C4 的写法）</div>
<div class="row"><span>同一组线程</span><div class="bar">
<div class="s ld" style="left:0%;width:12%">搬 0</div><div class="s mm" style="left:12%;width:10%">算 0</div>
<div class="s ld" style="left:22%;width:12%">搬 1</div><div class="s mm" style="left:34%;width:10%">算 1</div>
<div class="s ld" style="left:44%;width:12%">搬 2</div><div class="s mm" style="left:56%;width:10%">算 2</div>
<div class="s ld" style="left:66%;width:12%">搬 3</div><div class="s mm" style="left:78%;width:10%">算 3</div></div></div>
<div class="h">cp.async 多级流水线（3 个 stage）</div>
<div class="row"><span>拷贝（异步）</span><div class="bar">
<div class="s ld" style="left:0%;width:12%">搬 0</div><div class="s ld" style="left:12%;width:12%">搬 1</div><div class="s ld" style="left:24%;width:12%">搬 2</div><div class="s ld" style="left:36%;width:12%">搬 3</div></div></div>
<div class="row"><span>Tensor Core</span><div class="bar">
<div class="s mm" style="left:12%;width:10%">算 0</div><div class="s mm" style="left:24%;width:10%">算 1</div><div class="s mm" style="left:36%;width:10%">算 2</div><div class="s mm" style="left:48%;width:10%">算 3</div></div></div>
</div>
<p class="legend">示意图，不是实测时间。有了异步拷贝，第 k 块的计算和第 k+1、k+2 块的搬运同时进行，总时间从“搬 + 算”的和变成接近两者中较大的那个。</p>
:::

## ④ Hopper（sm_90a）：TMA、wgmma、mbarrier 与 warp specialization

### TMA：一个线程搬一整个 tile

Ampere 的 cp.async 仍然是**每个线程**算自己那几个元素的地址、各发各的拷贝。Hopper 的 TMA（Tensor Memory Accelerator）是一个专用的拷贝引擎：

- host 端先建一个 **tensor map**（描述整个张量的形状、stride、tile 大小、swizzle 方式）：[[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu@L305-L306|make_tma_atom(SM90_TMA_LOAD, mA, sA, tile_shape)]]。
- kernel 里**一个线程**发一条 `cp.async.bulk.tensor`，整个 tile（比如 128×64）就被搬进 shared memory：[[cutlass:include/cute/arch/copy_sm90_tma.hpp@L103-L117|SM90_TMA_LOAD_2D]]。地址计算、越界处理、swizzle 全部由硬件完成。
- 完成通知走 **mbarrier**，按**字节数**计：[[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu@L174-L176|arrive_and_expect_tx(barrier, 字节数) 之后发 TMA]]，消费者 [[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu@L225|wait(barrier, phase)]]，等到这么多字节都到了才继续。
- **multicast**：一个 cluster 里的几个 CTA 要同一块数据时，TMA 可以一次读、同时写进多个 CTA 的 shared memory：[[cutlass:include/cute/arch/copy_sm90_tma.hpp@L655|SM90_TMA_LOAD_MULTICAST_2D]]。cluster 是 Hopper 新加的一层：几个 CTA 保证同时运行在相邻的 SM 上，还能读写彼此的 shared memory（distributed shared memory）。

::: callout analogy 快递柜和取件码
cp.async 像每个人自己去仓库取几件货。TMA 像下一张整车订单：一个人填好单子（tensor map + 坐标），仓库自动把一整托盘货送到你家门口的快递柜（shared memory）。mbarrier 是快递柜上的计数器：订单上写了“共 16 KB”，计数器数到 16 KB 才亮灯（phase 翻转），等着用货的人（消费者）看灯就知道能不能开柜。multicast 是一张订单同时送到楼里好几户的柜子。
:::

### wgmma：四个 warp 合用一台更大的机器，操作数直接从 shared memory 读

- **warpgroup** = 4 个连续的 warp（128 个线程）一起发一条 `wgmma.mma_async`，M 固定为 64，N 从 8 到 256，FP16 时 K = 16：[[cutlass:include/cute/arch/mma_sm90_gmma.hpp@L139-L158|MMA_64x8x16 … m64n8k16]]（一直到 m64n256k16）。
- **操作数来自 shared memory**（通过一个 64 位的描述符，A 也可以放在寄存器里），不再需要 ldmatrix 把数据先搬进寄存器。
- **异步**：发出之后线程可以继续做别的事，之后再等它完成：[[cutlass:include/cute/arch/mma_sm90_gmma.hpp@L53-L80|wgmma.fence / commit_group / wait_group]]，在 CuTe 里是 [[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu@L228-L233|warpgroup_arrive → gemm → warpgroup_commit_batch → warpgroup_wait]]。TiledMMA 的构造和 Ampere 一样：[[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu@L297|make_tiled_mma(SM90_64x64x16_F16F16F16_SS)]]。

### warp specialization：分工，而不是每个人什么都干

有了异步的 TMA 和异步的 wgmma，最自然的写法是**分工**：

::: flow warp specialization：生产者、消费者和两组 barrier
cfg cw=225 nw=180
n p 0 0.65 "生产者 warpgroup" "只发 TMA，寄存器很少" k=warn w=1.05
n s0 1.15 0 "smem stage 0" "" k=teal
n s1 1.15 1.3 "smem stage 1" "" k=teal
n c 2.3 0.65 "消费者 warpgroup" "wgmma + softmax，寄存器很多" k=pass w=1.05
e p -> s0 "TMA"
e p -> s1 "TMA"
e s0 -> c "full 灯亮"
e s1 -> c "full 灯亮"
e c -> p "empty：用完了" dash
cap: 每个 stage 有两个 mbarrier：full（数据到齐，消费者可以用）和 empty（消费者用完，生产者可以覆盖）。shared memory 是一个环形缓冲区，生产者往前填，消费者在后面追。CuTe 教程里这两组 barrier 是 producer_mbar 和 consumer_mbar（[[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu@L152-L161|152–161 行]]）。
:::

- **setmaxnreg**：生产者几乎不需要寄存器（只是发 TMA），消费者需要大量寄存器存累加器。Hopper 允许 warpgroup 在运行中**交换寄存器配额**：FA3 里生产者 [[fa:hopper/flash_fwd_kernel_sm90.h@L308-L309|warpgroup_reg_dealloc]]、消费者 [[fa:hopper/flash_fwd_kernel_sm90.h@L360-L361|warpgroup_reg_alloc]]；两个消费者 warpgroup、用 TMA 读 K/V 时，生产者降到每线程 24 个寄存器，消费者升到 240 个，其他配置的数字在 [[fa:hopper/flash_fwd_kernel_sm90.h@L82-L83|LoadRegisterRequirement / MmaRegisterRequirement]]。文档：[PTX · setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)。

::: callout analogy 后厨分工
以前后厨每个厨师都要自己去冷库拿菜、自己切、自己炒（每个线程都搬数据、都算）。现在分成采购员和厨师：采购员（生产者 warpgroup）只负责把菜从冷库送到备菜台（TMA → shared memory），自己两手空空，不需要灶台（寄存器很少）；厨师（消费者）守着灶台只炒菜，灶台越大越好（寄存器很多）。备菜台上每个格子有两盏灯：“菜到了”（full）和“盘子空了”（empty），采购员和厨师看灯干活，不用互相喊话。setmaxnreg 就是把采购员用不上的灶台挪给厨师。
:::

### 读 FlashAttention-3：生产者 + 两个消费者的乒乓

FA3 的 forward kernel 把上面这些全部用上：

- **角色**：[[fa:hopper/flash_fwd_kernel_sm90.h@L308|warp_group_idx == 0 是生产者]]（发 Q、K、V 的 TMA），[[fa:hopper/flash_fwd_kernel_sm90.h@L360|其余是消费者]]（做 QKᵀ、softmax、PV）；消费者的个数是 [[fa:hopper/flash_fwd_kernel_sm90.h@L75-L78|NumMmaWarpGroups，1–3 个]]。
- **主循环**：生产者的 [[fa:hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp@L591|load()]] 和消费者的 [[fa:hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp@L955|mma()]]。
- **乒乓调度**：softmax 用的是普通 ALU（exp、max、sum），GEMM 用的是 Tensor Core，两者可以同时进行。两个消费者 warpgroup 用 [[fa:hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp@L915-L922|warp_scheduler_barrier_sync / arrive]] 轮流使用 Tensor Core：一个在做 GEMM 时，另一个在做 softmax。另外还有 warpgroup 内部的重叠（[[fa:hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp@L352|IntraWGOverlap]]）：发出下一块的 QKᵀ 之后，再做上一块的 softmax。

::: raw
<div class="gantt">
<div class="h">FA3 的乒乓：两个消费者 warpgroup 轮流占用 Tensor Core（示意）</div>
<div class="row"><span>消费者 WG 1</span><div class="bar">
<div class="s mm" style="left:0%;width:16%">GEMM</div><div class="s sm" style="left:16%;width:14%">softmax</div>
<div class="s mm" style="left:32%;width:16%">GEMM</div><div class="s sm" style="left:48%;width:14%">softmax</div>
<div class="s mm" style="left:64%;width:16%">GEMM</div><div class="s sm" style="left:80%;width:14%">softmax</div></div></div>
<div class="row"><span>消费者 WG 2</span><div class="bar">
<div class="s sm" style="left:2%;width:14%">softmax</div><div class="s mm" style="left:16%;width:16%">GEMM</div>
<div class="s sm" style="left:34%;width:14%">softmax</div><div class="s mm" style="left:48%;width:16%">GEMM</div>
<div class="s sm" style="left:66%;width:14%">softmax</div><div class="s mm" style="left:80%;width:16%">GEMM</div></div></div>
<div class="row"><span>生产者 WG 0</span><div class="bar">
<div class="s ld" style="left:0%;width:30%">TMA：K、V 的下几块</div><div class="s ld" style="left:32%;width:30%">TMA</div><div class="s ld" style="left:64%;width:30%">TMA</div></div></div>
</div>
<p class="legend">Tensor Core（绿色）几乎一直有活干：总有一个 warpgroup 在做 GEMM。softmax（紫色）的指数运算吞吐比 GEMM 低得多，如果不和 GEMM 重叠，它会成为瓶颈。</p>
:::

### 读 DeepGEMM（SM90）：FP8 GEMM 的精度补偿

DeepGEMM 的 SM90 FP8 GEMM 结构完全一样：[[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh@L34-L50|kNumTMAThreads = 128（一个 warpgroup）+ 若干 math warpgroup]]，[[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh@L171|TMA 侧 reg_dealloc]]、[[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh@L250|math 侧 reg_alloc]]，[[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh@L119-L139|full / empty barrier]]。

它多做了一件事，和 M3 直接相关：每个 K 块做完 [[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh@L300-L305|FP8 的 wgmma]] 之后，把结果乘上这一块的 scale，[[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh@L321|加到 FP32 的 final_accum 里]]（在 CUDA core 上做）。这就是 M3 里 `acc += tl.dot(a, b) * (sa * sb)` 的硬件级版本，同时也补偿了 Hopper FP8 Tensor Core 内部累加精度有限的问题。

## ⑤ Blackwell（sm_100a）：tcgen05、Tensor Memory 与 2-CTA MMA

### tcgen05.mma：一个线程发起，累加器放在 Tensor Memory

| | Hopper wgmma | Blackwell tcgen05.mma |
|---|---|---|
| 谁发起 | 一个 warpgroup（128 线程）一起发 | **一个线程**发 |
| 累加器在哪 | 128 个线程的寄存器里 | **Tensor Memory（TMEM）**：每个 SM 一块专用的 256 KB（128 行 × 512 列 × 32 位） |
| 读结果 | 直接在寄存器里 | 用 `tcgen05.ld` 从 TMEM 读到寄存器 |
| 跨 CTA | 不支持 | `cta_group::2`：两个 CTA（一对 SM）合作做一条更大的 MMA |
| 低精度 | FP8 | FP8 / FP6 / FP4，**硬件 block scaling**（MXFP8、MXFP4、NVFP4） |

源码：[[cutlass:include/cute/arch/mma_sm100_umma.hpp@L86-L111|SM100_MMA_F16BF16_SS → tcgen05.mma.cta_group::1.kind::f16]]；TMEM 的分配 [[cutlass:include/cute/arch/tmem_allocator_sm100.hpp@L82-L87|tcgen05.alloc]]；2-CTA 版本 [[cutlass:include/cute/arch/mma_sm100_umma.hpp@L552-L576|SM100_MMA_F16BF16_2x1SM_SS → cta_group::2]]；block-scaled 版本 [[cutlass:include/cute/arch/mma_sm100_umma.hpp@L1250-L1277|SM100_MMA_MXF8F6F4_SS → kind::mxf8f6f4.block_scale]]。

::: flow Blackwell 上一个 CTA 的分工（对照 DeepGEMM SM100 / FA4）
cfg cw=225 nw=180
n tma 0 0 "TMA warp（1 个线程发）" "全局 → smem" k=warn w=1.05
n mma 1.15 0 "MMA warp（1 个线程发）" "tcgen05.mma" k=vio w=1.05
n tm 2.3 0 "Tensor Memory" "累加器，256 KB / SM" k=teal
n epi 2.3 1.3 "epilogue warp" "tcgen05.ld → 寄存器 → 写回" k=pass w=1.05
n sm 1.15 1.3 "smem stages" "full / empty barrier" k=acc
e tma -> sm
e sm -> mma
e mma -> tm
e tm -> epi
cap: 和 Hopper 相比，累加器从“占满 128 个线程的寄存器”变成放在 TMEM 里，于是发 MMA 的只需要一个线程，大量寄存器空出来给 softmax、反量化、epilogue 这些“非矩阵”工作。
:::

CuTe 的 Blackwell 教程按难度排好了：[[cutlass:examples/cute/tutorial/blackwell/01_mma_sm100.cu@L74-L87|01：tcgen05.mma + tcgen05.ld]]（[[cutlass:examples/cute/tutorial/blackwell/01_mma_sm100.cu@L234-L238|分配 TMEM]]、[[cutlass:examples/cute/tutorial/blackwell/01_mma_sm100.cu@L289-L293|gemm 之后 umma_arrive 通知]]、[[cutlass:examples/cute/tutorial/blackwell/01_mma_sm100.cu@L303|make_tmem_copy 把结果读回寄存器]]）→ 02：加 TMA → 03：TMA multicast → [[cutlass:examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu@L450|04：2-CTA 的 MMA]]（[[cutlass:examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu@L532|SM100_TMA_2SM_LOAD_MULTICAST]]）→ 05：TMA epilogue。

::: callout analogy 公共账本
Hopper 上，每次大合唱（wgmma）的结果要分到 128 个歌手各自的小本子上记着（寄存器），歌手们因此没空干别的。Blackwell 在舞台边上放了一块公共大黑板（Tensor Memory），合唱结果直接写在黑板上，指挥（一个线程）喊一声开始就行。歌手们的小本子空出来了，可以去做和声以外的事（softmax、反量化）。2-CTA MMA 则是两个合唱团并排站，合起来唱一首更大的曲子，共用一块更大的黑板。
:::

### 读 FlashAttention-4：用 CuTe DSL 写成，角色分得更细

FA4（[[fa:flash_attn/cute/flash_fwd_sm100.py|flash_fwd_sm100.py]]，用 Python 的 CuTe DSL 写）的角色分配一目了然：[[fa:flash_attn/cute/flash_fwd_sm100.py@L322-L327|softmax0、softmax1 各 4 个 warp，correction 4 个 warp，MMA 1 个 warp，epilogue 1 个 warp，load 1 个 warp]]。

- **S 和 O 都在 TMEM 里**：[[fa:flash_attn/cute/flash_fwd_sm100.py@L390-L395|tmem_s_offset、tmem_o_offset]]，QKᵀ 的结果 S 和输出累加器 O 各占 TMEM 的一段，两个 Q tile（softmax0 和 softmax1）乒乓。
- **correction warp**：online softmax 里“max 变了就要把 O 乘一个系数”（T5 讲过的 alpha）由专门的 warp 去做，不挡 softmax 和 MMA 的路。
- **exp2 软件模拟**：Blackwell 的 Tensor Core 太快了，softmax 里的指数运算（硬件的 MUFU 单元）反而成了瓶颈，FA4 用多项式在 FMA 单元上模拟一部分 exp2：[[fa:flash_attn/cute/flash_fwd_sm100.py@L249-L251|ex2_emu 的配置]]。这正是第 ① 节那条主线的延续：Tensor Core 越快，非矩阵的部分越要想办法挤。
- 寄存器配额同样在运行中调整：[[fa:flash_attn/cute/flash_fwd_sm100.py@L1256|setmaxregister_decrease]]。

### 读 DeepGEMM（SM100）：一个 warp 一个角色

[[deepgemm:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh@L210-L212|warp 0：TMA load]]，[[deepgemm:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh@L299-L300|warp 1：MMA issue（只在 leader CTA 上）]]，[[deepgemm:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh@L496|其余 warp group：epilogue]]。TMEM 分配按是否 2-CTA 选择 [[deepgemm:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh@L49|Allocator1Sm 或 Allocator2Sm]]，MMA 的形状见 [[deepgemm:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh@L64-L68|UMMA_M / UMMA_N / UMMA_K]]。和 SM90 版本相比，FP8 的 scale 由 Tensor Core 的 block scaling 在硬件里乘，不再需要 CUDA core 上的 promotion。

## ⑥ 哪块 GPU 能跑哪些

| GPU | 架构 | 能用的矩阵指令 | 拷贝 | 本页哪几节能动手 |
|---|---|---|---|---|
| A100 | sm_80 | WMMA、mma.sync | cp.async | ②③ |
| RTX 4090 / L40S | sm_89 | 同上 + FP8 mma | cp.async | ②③ |
| H100 / H200 | **sm_90a** | + **wgmma** | + **TMA**、cluster | ②③④（FA3、DeepGEMM SM90） |
| B200 / GB200 | **sm_100a** | + **tcgen05**、TMEM、2-CTA | TMA（含 2-SM） | ②③⑤（FA4、DeepGEMM SM100） |
| RTX 5090（本仓库） | sm_120 | WMMA、mma.sync（含 FP8 / FP4） | cp.async 等 | ②③；没有 wgmma 和 tcgen05 |

`sm_90a`、`sm_100a` 里的 **a** 表示“架构专用特性”：编译出来的代码只能在这一代上运行，不向后兼容。这也是 FlashAttention、DeepGEMM 为每一代 GPU 单独写一套主循环的原因。

## ⑦ 推荐的阅读和动手路线

::: flow 从 C4 到 FA4
cfg cw=235 nw=205
n c4 0 0 "C4 v2" "寄存器外积" k=acc
n w 1 0 "WMMA 改写 C4" "对比 cuBLAS FP16" k=acc
n s80 2 0 "CuTe sgemm_sm80" "MMA Atom · TiledMMA · cp.async" k=vio w=1.05
n s90 3.1 0 "CuTe wgmma_tma_sm90" "TMA · mbarrier · wgmma" k=warn w=1.05
n fa3 3.1 1.3 "FA3 · DeepGEMM SM90" "warp specialization · 乒乓" k=warn w=1.05
n s100 2 1.3 "CuTe blackwell/01–05" "tcgen05 · TMEM · 2-CTA" k=pass w=1.05
n fa4 1 1.3 "FA4 · DeepGEMM SM100" "角色更细的分工" k=pass
e c4 -> w
e w -> s80
e s80 -> s90
e s90 -> fa3
e fa3 -> s100
e s100 -> fa4
:::

1. **在 5090 上就能做**：用 WMMA 改写 C4（第 ②节），然后读 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@L368-L375|sgemm_sm80]] 和 [[cutlass:media/docs/cpp/cute/0t_mma_atom.md|0t_mma_atom.md]]，把 MMA Atom 的 TV layout 和 G1 / G2 对照起来。
2. **有 H100 时**：跑 [[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu|wgmma_tma_sm90.cu]]，再读 FA3 的 [[fa:hopper/flash_fwd_kernel_sm90.h@L308|生产者 / 消费者分支]] 和 DeepGEMM SM90 的主循环。
3. **有 B200 时**：按顺序跑 [[cutlass:examples/cute/tutorial/blackwell/01_mma_sm100.cu|blackwell/01–05]]，再读 FA4 和 DeepGEMM SM100。

::: answer 自测 1：一条 mma.sync m16n8k16 和一条 FFMA 指令，各做多少次乘加？
mma.sync：一个 warp 一条指令，16 × 8 × 16 = 2048 次乘加。FFMA：每个线程 1 次，一个 warp 一条指令 32 次。前者是后者的 64 倍。
:::

::: answer 自测 2：为什么 Hopper 的 kernel 要把一个 warpgroup 专门用来发 TMA，而不是让所有线程都一起搬一点、算一点？
TMA 只需要一个线程发起，搬运本身由硬件完成，所以“搬数据”几乎不消耗线程的计算和寄存器。让一组线程专门发 TMA、并把寄存器让给计算的 warpgroup（setmaxnreg），计算的 warpgroup 就能用更多寄存器存更大的累加器 tile；两边通过 full / empty barrier 异步配合，搬运和计算完全重叠。所有线程都既搬又算，就要在两件事之间来回切换，寄存器也要按两者中较大的需求分配。
:::

::: answer 自测 3：Blackwell 把累加器放进 Tensor Memory，最大的好处是什么？
累加器不再占用寄存器，发 MMA 也只需要一个线程，大量寄存器和线程可以去做 softmax、反量化、epilogue 等非矩阵工作。FA4 的 correction warp、exp2 软件模拟，都依赖这些空出来的资源。
:::

## 源码与资料索引

**本仓库**：[C4 讲解](c4_gemm.html) · [T4 讲解](t4_matmul.html) · [T5 讲解](t5_flash_attn.html) · [M3 讲解](m3_fp8_block_gemm.html) · [CuTe 专题](cute_overview.html) · [G1 讲解](g1_layouts.html)

**WMMA**：[[cudasamples:cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu@L389|simple_wmma_gemm]] · [[cudasamples:cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu@L197|compute_gemm]] · [Programming Guide · WMMA](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#wmma)

**CuTe（Ampere）**：[[cutlass:include/cute/arch/mma_sm80.hpp@L158|SM80_16x8x16]] · [[cutlass:include/cute/atom/mma_traits_sm80.hpp@L109|MMA_Traits]] · [[cutlass:include/cute/arch/copy_sm75.hpp@L121|ldmatrix]] · [[cutlass:include/cute/arch/copy_sm80.hpp@L47|cp.async]] · [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@L375|sgemm_sm80]] · [[cutlass:media/docs/cpp/cute/0t_mma_atom.md|0t_mma_atom.md]]

**Hopper**：[[cutlass:include/cute/arch/copy_sm90_tma.hpp@L103|TMA]] · [[cutlass:include/cute/arch/mma_sm90_gmma.hpp@L139|wgmma]] · [[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu|wgmma_tma_sm90.cu]] · [[fa:hopper/flash_fwd_kernel_sm90.h@L308|FA3 kernel]] · [[fa:hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp@L591|FA3 主循环]] · [[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh@L34|DeepGEMM SM90]] · [[flashinfer:include/flashinfer/attention/hopper/prefill_sm90.cuh@L148|FlashInfer 的 Hopper prefill（同样的生产者 / 消费者）]]

**Blackwell**：[[cutlass:include/cute/arch/mma_sm100_umma.hpp@L86|tcgen05.mma]] · [[cutlass:include/cute/arch/tmem_allocator_sm100.hpp@L82|TMEM 分配]] · [[cutlass:examples/cute/tutorial/blackwell/01_mma_sm100.cu|blackwell/01–05]] · [[fa:flash_attn/cute/flash_fwd_sm100.py@L322|FA4]] · [[deepgemm:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh@L210|DeepGEMM SM100]]

**PTX 文档**：[mma（m16n8k16 的寄存器布局）](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float) · [ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix) · [cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) · [cp.async.bulk.tensor（TMA）](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) · [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier) · [wgmma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-operation-wgmma-mma-async) · [setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) · [tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-instructions) · [Tensor Memory 分配](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-alloc-manage-instructions) · [CTA pair](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-cta-pair) · [block scaling](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-block-scaling)

**CUDA Programming Guide**：[TMA](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-the-tensor-memory-accelerator-tma) · [Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)

**论文**：[FlashAttention-3](https://arxiv.org/abs/2407.08608) · [DeepSeek-V3（FP8 训练与 DeepGEMM 的背景）](https://arxiv.org/abs/2412.19437)
