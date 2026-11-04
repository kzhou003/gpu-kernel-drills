title: T4 讲解：Triton GEMM，tl.dot、分组顺序与 autotune
lede: C4 里手写的 shared memory 和寄存器分块，在 Triton 里只剩一行 tl.dot。这一页逐段读参考解：program 怎么按分组顺序分到 C 的 tile（附 4×4 网格的完整对照）、二维指针块是怎么广播出来的、主循环和 mask、num_stages 的软件流水线，以及 autotune 在背后做了什么。

::: callout info 怎么读这一页
对照 [[here:solutions/t4_matmul.py|t4_matmul.py]] 读。建议先看 [C4 讲解](c4_gemm.html)，这一页会经常拿 C4 来对比：哪些事 Triton 替你做了，哪些仍然要你自己决定。
:::

## ① 先看全貌：每个 program 一个 C tile

和 C4 的 v2 一样：每个 program 负责 C 的一个 BLOCK_M × BLOCK_N 的 tile，沿 K 方向每次取 A 的 BLOCK_M × BLOCK_K 块和 B 的 BLOCK_K × BLOCK_N 块，累加到 fp32 的 acc 里。

| 事情 | C4（CUDA 手写） | T4（Triton） |
|---|---|---|
| tile 读进 shared memory | 自己写 e 循环、自己 `__syncthreads()` | `tl.load` 一个二维指针块，编译器决定放哪 |
| 线程到输出的映射 | 自己算 `tid % 16`、`tid / 16` | 编译器按 layout 分 |
| 内层乘加 | 4×4 外积，FP32 CUDA core | `tl.dot`，**Tensor Core**（mma / wgmma） |
| 预取下一块 | 没做（C4 的“测完之后”里提到双缓冲） | `num_stages` 一个参数 |
| tile 尺寸 | 写死 64×64×16 | autotune 从几个候选里选 |
| program 的顺序 | 按 blockIdx 行主序 | **分组顺序**，提高 L2 命中（你自己写） |

::: flow 一个 program 的四步
cfg cw=230 nw=175
n id 0 0 "① 我负责哪个 tile" "pid → (pid_m, pid_n)" k=acc
n ptr 1 0 "② 二维指针块" "a_ptrs、b_ptrs" k=acc
n loop 2 0 "③ 沿 K 循环" "load → tl.dot → 指针后移" k=pass w=1.1
n epi 2 1.3 "④ epilogue" "acc 转 dtype，store" k=vio
e id -> ptr
e ptr -> loop
e loop -> epi
cap: 第 ③ 步里 a 和 b 的 load 会被编译器放进 shared memory，并按 num_stages 提前若干轮发出，tl.dot 从 shared memory 取数喂给 Tensor Core。这些都是 C4 里你手写的东西。
:::

## ② 分组顺序：program 和 tile 怎么对应

[[here:solutions/t4_matmul.py@L24-L32|24–32 行]]

```python
num_pid_in_group = GROUP_M * num_pid_n
group_id     = pid // num_pid_in_group
first_pid_m  = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)   # 最后一组可能不满
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

这几行只是把一维的 `pid` 换算成 tile 的 (行, 列)。最直接的换算是行主序（`pid_m = pid // num_pid_n`，`pid_n = pid % num_pid_n`），分组顺序则是**每 GROUP_M 行 tile 为一组，组内先往下走、再往右走**。

C 被切成 4×4 个 tile，GROUP_M = 2，每格里写的是负责它的 pid：

::: raw
<div style="display:flex; gap:28px; flex-wrap:wrap">
<div><p class="legend">行主序</p>
<div class="cells" style="grid-template-columns: repeat(4, 52px)">
<div class="c0">0</div><div class="c0">1</div><div class="c0">2</div><div class="c0">3</div>
<div>4</div><div>5</div><div>6</div><div>7</div>
<div>8</div><div>9</div><div>10</div><div>11</div>
<div>12</div><div>13</div><div>14</div><div>15</div>
</div></div>
<div><p class="legend">分组顺序（GROUP_M = 2）</p>
<div class="cells" style="grid-template-columns: repeat(4, 52px)">
<div class="c2">0</div><div class="c2">2</div><div>4</div><div>6</div>
<div class="c2">1</div><div class="c2">3</div><div>5</div><div>7</div>
<div class="c1">8</div><div class="c1">10</div><div class="c1">12</div><div class="c1">14</div>
<div class="c1">9</div><div class="c1">11</div><div class="c1">13</div><div class="c1">15</div>
</div></div>
</div>
<p class="legend">左图里 pid 0–3 是第一行的 4 个 tile；右图里 pid 0–3 是左上角的 2×2。下半部分橙色是第二组（group_id = 1）。</p>
:::

拿 pid = 5 验算右图：num_pid_in_group = 2 × 4 = 8，group_id = 0，first_pid_m = 0，group_size_m = 2；pid_m = 0 + (5 % 8) % 2 = 1，pid_n = (5 % 8) // 2 = 2。所以 pid 5 负责第 1 行第 2 列，和图里一致。

**为什么这样更快：** 假设 GPU 同时在跑 pid 0–3 这 4 个 program。

| 顺序 | 同时在跑的 tile | 需要 A 的哪几行块 | 需要 B 的哪几列块 | 一共要从显存拉几个 panel |
|---|---|---|---|---|
| 行主序 | (0,0) (0,1) (0,2) (0,3) | 1 个 | 4 个 | 5 |
| 分组 | (0,0) (1,0) (0,1) (1,1) | 2 个 | 2 个 | 4 |

同时运行的 program 数越多，差距越大。真实场景里同时运行的是几百个 program，行主序会让它们需要一整行 B 的 panel，L2 装不下，就会反复从显存读。分组之后，同时运行的 program 集中在一个近似正方形的区域里，A、B 的 panel 都能留在 L2 里被重复使用。题目页建议把 GROUP_M 改成 1 来对比 L2 命中率。

::: callout analogy 印刷厂排版
印刷厂要印一面墙那么大的海报，切成 16 块，每块需要从仓库调一卷“横向颜料”（A 的一行）和一卷“纵向颜料”（B 的一列）。工作台旁边的架子（L2）只放得下几卷。如果按行一块块印，印完第一行就用掉了 4 卷纵向颜料，架子放不下，后面又得回仓库取。按 2×2 的小方块印，同样印 4 块只要 2 卷横向加 2 卷纵向，架子上的颜料被反复用到。
:::

## ③ 二维指针块：广播出来的

[[here:solutions/t4_matmul.py@L35-L39|35–39 行]]

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)       # [BLOCK_M]
offs_k = tl.arange(0, BLOCK_K)                         # [BLOCK_K]
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # [BLOCK_M, BLOCK_K]
```

`offs_m[:, None]` 是一列（形状 `[BLOCK_M, 1]`），`offs_k[None, :]` 是一行（`[1, BLOCK_K]`），两者相加按 NumPy 的规则**广播**成 `[BLOCK_M, BLOCK_K]` 的矩阵。每个格子是 A 里对应元素的地址。

BLOCK_M = BLOCK_K = 4，A 是 `[M, 8]` 的行优先矩阵（stride_am = 8，stride_ak = 1），pid_m = 0 时：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 130px repeat(4, 60px); min-width: 400px">
<div class="lbl"></div><div class="c1">k=0</div><div class="c1">k=1</div><div class="c1">k=2</div><div class="c1">k=3</div>
<div class="c0">m=0 → 0</div><div>0</div><div>1</div><div>2</div><div>3</div>
<div class="c0">m=1 → 8</div><div>8</div><div>9</div><div>10</div><div>11</div>
<div class="c0">m=2 → 16</div><div>16</div><div>17</div><div>18</div><div>19</div>
<div class="c0">m=3 → 24</div><div>24</div><div>25</div><div>26</div><div>27</div>
</div></div>
<p class="legend">格子里是 a_ptrs 相对 a_ptr 的偏移（以元素为单位）。左列是 offs_m × stride_am，上行是 offs_k × stride_ak，每格 = 行头 + 列头。</p>
:::

**为什么要传 stride，而不是假设连续**：传 `b.t()` 这样的转置视图进来，`stride_bk = 1`、`stride_bn = K`，同一套代码照样对，不需要先 `.contiguous()` 拷贝一份。PyTorch 的 `Linear` 存的权重就是 `[N, K]`，乘的时候需要的正是它的转置视图。

## ④ 主循环：load、tl.dot、指针后移

[[here:solutions/t4_matmul.py@L42-L50|42–50 行]]

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    k_remaining = K - k * BLOCK_K
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
    acc = tl.dot(a, b, acc)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

- **mask 也是二维广播**：`(offs_m[:, None] < M)` 是一列 bool，`(offs_k[None, :] < k_remaining)` 是一行 bool，`&` 之后是 `[BLOCK_M, BLOCK_K]` 的 mask。越界填 0，和 C4 的道理一样：0 不影响点积。
- **`k_remaining`**：最后一轮 K 方向可能不满一个 BLOCK_K。用“还剩多少”来比较，指针只需要往后移，offs_k 不用变。
- **`tl.dot(a, b, acc)`** 等于 `acc += a @ b`。fp16 / bf16 输入时编译成 Tensor Core 指令，在 fp32 里累加。第三个参数让编译器把累加融合进 mma 指令。
- **指针整体后移**：A 往右移 BLOCK_K 列，B 往下移 BLOCK_K 行。

### num_stages：软件流水线

你写的循环是“读一块、算一块”，但编译器会把 load 提前发出：

::: flow num_stages = 3 时，第 k 轮在做什么
cfg cw=270 nw=175
n c 0 0 "Tensor Core" "算第 k 块" k=pass
n s1 1 0 "shared memory 槽 1" "第 k+1 块已到" k=warn
n s2 2 0 "shared memory 槽 2" "第 k+2 块在路上" k=todo
e s1 -> c "下一轮用"
e s2 -> s1 dash
cap: 显存读取有几百个周期的延迟。提前发出后面几轮的 load，等真正要用时数据已经在 shared memory 里了，计算和搬运就重叠起来。stage 越多，藏延迟的能力越强，但每个 stage 都要一份 shared memory。
:::

每个 stage 要存一块 A 和一块 B（fp16 每个元素 2 字节）：

| config | 每 stage 的元素 | 每 stage 字节 | stages | 共计 |
|---|---|---|---|---|
| 128 × 256 × 64 | 128·64 + 64·256 = 24,576 | 48 KB | 3 | 144 KB |
| 128 × 128 × 64 | 128·64 + 64·128 = 16,384 | 32 KB | 4 | 128 KB |
| 64 × 64 × 32 | 64·32 + 32·64 = 4,096 | 8 KB | 5 | 40 KB |

超过 GPU 每个 SM 的 shared memory 上限，这个 config 就会编译失败或被跳过；占用太多，每个 SM 能同时驻留的 program 就少。

::: callout analogy 流水线上的备料
厨师（Tensor Core）每做一道菜要用一盘切好的料。如果每次都是做完一道再去冷库取料、切好，厨师大部分时间在等。改成帮厨提前两道菜就开始取料切配，放在灶台旁的备料台上（shared memory 的几个槽），厨师做完一道，下一道的料已经在手边了。备料台越大，能提前准备的菜越多，但厨房的空间是有限的。
:::

## ⑤ epilogue

[[here:solutions/t4_matmul.py@L53-L55|53–55 行]]：acc 从 fp32 转回 fp16/bf16，用 C 的 stride 算出二维指针块，带 mask 写回。题目页的进阶练习是在这里融合 bias 和激活：`c = gelu(acc + bias)`，这是 GEMM 融合最常见的形式。

## ⑥ autotune 在背后做了什么

[[here:solutions/t4_matmul.py@L60-L70|60–70 行]] · host 端 [[here:solutions/t4_matmul.py@L84-L93|84–93 行]]

```python
matmul_kernel_autotuned = triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])(matmul_kernel)
grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
matmul_kernel_autotuned[grid](*args)
```

::: flow 第一次遇到某个 (M, N, K) 时
cfg cw=225 nw=175
n call 0 0 "matmul(a, b)" "key = (M, N, K)" k=acc
n hit 1 0 "缓存里有吗？" "Autotuner.cache" k=warn
n bench 1 1.3 "5 个 config 各跑几次" "编译 + 计时" k=todo
n best 2 1.3 "选最快的" "写入 cache[key]" k=vio
n run 2 0 "用这个 config 启动" "grid(meta)" k=pass
e call -> hit
e hit -> run "有"
e hit -> bench "没有"
e bench -> best
e best -> run
cap: 所以第一次调用某个 shape 很慢（要编译 5 个版本、各跑若干次），之后同一个 shape 就直接用缓存的结果。key 里不包含的参数变了不会重新调优。
:::

- **grid 必须是函数**：autotuner 在尝试不同 config 时 BLOCK_M、BLOCK_N 各不相同，program 数也跟着变，所以 grid 要根据 `meta`（当前 config 的参数）现算。
- **CPU 解释器不调优**（85–86 行）：解释器上计时没有意义，直接用一个固定的小 config。
- 源码：[[triton:python/triton/runtime/autotuner.py@Autotuner|Autotuner]]，查缓存和 benchmark 的逻辑在 [[triton:python/triton/runtime/autotuner.py@L232-L259|run()]]。

## ⑦ 对照与延伸

- **官方 tutorial 03** 和参考解几乎逐行相同：[[triton:python/tutorials/03-matrix-multiplication.py@L233|matmul_kernel]]，L2 分组的讲解在 [[triton:python/tutorials/03-matrix-multiplication.py@L93-L133|“L2 Cache Optimizations” 一节]]（带图），[[triton:python/tutorials/03-matrix-multiplication.py@L304|tl.dot]]。
- **persistent kernel**：program 数等于 SM 数，每个 program 循环处理多个 tile，省掉反复启动 program 的开销：[[triton:python/tutorials/09-persistent-matmul.py@L296|matmul_kernel_persistent]]。题目页的面试追问提到这是 cuBLAS 的优势之一。
- **grouped GEMM**（一次算很多个不同 shape 的小 GEMM，M2 的 MoE 会用到）：[[triton:python/tutorials/08-grouped-gemm.py|08-grouped-gemm]]；**block-scaled GEMM**（M3 的 FP8）：[[triton:python/tutorials/10-block-scaled-matmul.py|10-block-scaled-matmul]]。

## ⑧ 常见坑与自测

- **grid 用了固定的 BLOCK_M**：配合 autotune 时 program 数不对，部分 tile 没人算或者越界。
- **只 mask 了 M、N，没 mask K**：K 不是 BLOCK_K 的倍数时读进垃圾值。
- **acc 用了 fp16**：大 K 时累加误差很大。Tensor Core 本来就支持 fp32 累加，没有性能损失。
- **GROUP_M 的最后一组**：num_pid_m 不是 GROUP_M 的倍数时，最后一组更矮，必须用 `group_size_m` 而不是 `GROUP_M` 取模。

::: answer 自测 1：num_pid_m = 3，num_pid_n = 2，GROUP_M = 2。写出 pid 0–5 对应的 (pid_m, pid_n)。
num_pid_in_group = 4。第 0 组（pid 0–3）：group_size_m = 2，得到 (0,0) (1,0) (0,1) (1,1)。第 1 组（pid 4–5）：first_pid_m = 2，group_size_m = min(3 − 2, 2) = 1，pid 4 → (2, 0)，pid 5 → (2, 1)。如果第 1 组误用 GROUP_M = 2 取模，pid 5 会算出 pid_m = 3，越界。
:::

::: answer 自测 2：M = N = K = 4096，config 是 128 × 128 × 64，一共几个 program？每个 program 的主循环跑几轮？
(4096/128) × (4096/128) = 32 × 32 = 1024 个 program，每个跑 4096 / 64 = 64 轮。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/t4_matmul.py|solutions/t4_matmul.py]] · [[here:mine/t4_matmul.py|mine/t4_matmul.py]] · [[here:tests/test_t4_matmul.py|tests/test_t4_matmul.py]] · [[here:content/t4.md|content/t4.md]]

**Triton**：[[triton:python/tutorials/03-matrix-multiplication.py@L233|tutorial 03]] · [[triton:python/tutorials/09-persistent-matmul.py@L296|09 persistent]] · [[triton:python/tutorials/08-grouped-gemm.py|08 grouped]] · [[triton:python/triton/runtime/autotuner.py@Autotuner|Autotuner]] · API：[tl.dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) · [triton.autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html) · [triton.Config](https://triton-lang.org/main/python-api/generated/triton.Config.html)

**延伸阅读**：[Simon Boehm · How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)（C4 → cuBLAS 的每一步） · [CUTLASS](https://github.com/NVIDIA/cutlass)
