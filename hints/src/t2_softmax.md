title: T2 讲解：一行一个 program 的融合 softmax
lede: 读一次、写一次的数值稳定 softmax。逐行读参考解：为什么要减 max、padding 位置为什么填 -inf、row stride 是干什么的、tl.max 和 tl.sum 背后其实就是 C2 的 reduction，以及行太长时为什么要换成 online softmax。

::: callout info 怎么读这一页
对照 [[here:solutions/t2_softmax.py|t2_softmax.py]] 读。kernel 只有十几行，但每一行背后都有一个“为什么”。第 ⑤ 节的 online softmax 是 T5 FlashAttention 的前置知识。
:::

## ① 先看全貌：融合省在哪

softmax 对每一行 x 做：`y = exp(x − max(x)) / Σ exp(x − max(x))`。不融合的写法（官方 tutorial 里的 `naive_softmax`）要把同一行反复从显存读出来：

::: flow 不融合 vs 融合：一行数据进出显存几次
cfg cw=200 nw=160
n a1 0 0 "求 max" "读 x" k=acc
n a2 1 0 "x − max" "读 x，写 z" k=acc
n a3 2 0 "exp" "读 z，写 num" k=acc
n a4 3 0 "求和、相除" "读 num 两次，写 y" k=acc
n f 0 1.4 "融合 kernel" "读 x 一次" k=pass
n f2 1.5 1.4 "寄存器里" "max · exp · sum · 除" k=pass
n f3 3 1.4 "写 y 一次" "" k=pass
g 0 0 3 0 "不融合：每一步都是一个 kernel，中间结果写回显存" k=todo
g 0 1.4 3 1.4 "融合：整行留在片上" k=pass
e a1 -> a2
e a2 -> a3
e a3 -> a4
e f -> f2
e f2 -> f3
cap: softmax 是 memory-bound 的，时间约等于搬运的字节数除以带宽。融合版每个元素只读一次、写一次，这就是下限。题目页的实测：N = 4096 时融合版 1540 GB/s，torch.softmax 1025 GB/s。
:::

前提是**整行放得进一个 program**。每个 program 负责一行，BLOCK_SIZE 取不小于 N 的 2 的幂。

::: callout analogy 收银台
每个收银员（program）负责一位顾客的整个购物车（一行）。不融合的做法是：先让顾客把东西全部放到传送带上找出最贵的那件，收回去；再放一遍算折扣，收回去；再放一遍结账。融合的做法是：东西放上传送带一次，收银员把所有计算都在脑子里（寄存器）做完，直接打出小票。购物车太大、收银员记不住的时候（N 很大），才需要分批处理（第 ⑤ 节）。
:::

## ② 为什么要减 max

数学上 `softmax(x) = softmax(x − c)` 对任意常数 c 成立（分子分母同时乘以 e⁻ᶜ）。取 c = max(x) 是为了**不溢出**：

| x | 直接算 exp(x) | 减 max 之后 |
|---|---|---|
| [1, 2, 3] | 2.7, 7.4, 20.1 | exp([−2, −1, 0]) = 0.14, 0.37, 1 |
| [1000, 1001] | fp32 的 exp 超过 88.7 就是 inf，inf / inf = **NaN** | exp([−1, 0]) = 0.37, 1 → 结果 [0.27, 0.73] |

减完之后最大的指数是 0，exp 的结果都在 (0, 1] 之间，分母至少是 1，既不会上溢也不会除以 0。

::: callout analogy 量身高用相对值
全班同学比身高，如果都从海平面量起（几千米），数字很大、容易写不下。以全班最高的同学为基准，每个人记“比他矮多少”，数字都在 0 附近，比较起来一样准确。softmax 只关心相对大小，所以减去最大值不影响结果。
:::

## ③ kernel 逐行讲

[[here:solutions/t2_softmax.py@L12-L30|softmax_kernel（12–30 行）]]

### 我负责哪一行

```python
row  = tl.program_id(0).to(tl.int64)
cols = tl.arange(0, BLOCK_SIZE)
mask = cols < n_cols
```

- grid 是 `(M,)`，program `row` 处理第 row 行。
- **`.to(tl.int64)`**：`program_id` 是 int32。下一行要算 `row * in_row_stride`，M × N 超过 2³¹ 时 int32 乘法会溢出。和 C1 里 `(int64_t)blockIdx.x` 是同一个道理。
- `cols` 是 0 到 BLOCK_SIZE − 1，`mask` 屏蔽掉 N 之后的 padding 位置。

### 读：stride 和 -inf

[[here:solutions/t2_softmax.py@L21|21 行]]

```python
x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf")).to(tl.float32)
```

- **`in_row_stride`**：第 row 行的起点是 `row * stride(0)`，不一定是 `row * N`。比如 `x = big[:, :1000]` 是从一个 `[M, 4096]` 张量切出来的视图，每行只用前 1000 个，但行与行之间隔着 4096 个元素。只要**行内**连续（`stride(1) == 1`），kernel 就能直接读，不需要拷贝。
- **`other=-inf`**：padding 位置填负无穷，而不是 0。原因：
  - 求 max 时，-inf 永远不会是最大值；如果填 0，而一行全是负数，max 就会被错误地算成 0。
  - 求和时，exp(-inf) = 0，不贡献任何值。
- **`.to(tl.float32)`**：fp16 输入也在 fp32 里算 exp 和 sum。

N = 5，BLOCK_SIZE = 8 时，一行的处理过程：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 130px repeat(8, minmax(50px, 1fr)); min-width: 600px">
<div class="lbl">cols</div><div>0</div><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>7</div>
<div class="lbl">x（load 之后）</div><div class="c0">−1</div><div class="c0">−3</div><div class="c0">−2</div><div class="c0">−5</div><div class="c0">−4</div><div class="cx">−inf</div><div class="cx">−inf</div><div class="cx">−inf</div>
<div class="lbl">x − max(= −1)</div><div class="c1">0</div><div class="c1">−2</div><div class="c1">−1</div><div class="c1">−4</div><div class="c1">−3</div><div class="cx">−inf</div><div class="cx">−inf</div><div class="cx">−inf</div>
<div class="lbl">exp</div><div class="c2">1</div><div class="c2">.135</div><div class="c2">.368</div><div class="c2">.018</div><div class="c2">.050</div><div class="cx">0</div><div class="cx">0</div><div class="cx">0</div>
<div class="lbl">÷ sum(= 1.571)</div><div class="c3">.636</div><div class="c3">.086</div><div class="c3">.234</div><div class="c3">.012</div><div class="c3">.032</div><div class="cx">不写</div><div class="cx">不写</div><div class="cx">不写</div>
</div></div>
<p class="legend">这一行全是负数。如果 padding 填 0，max 会变成 0 而不是 −1，结果依然正确吗？数学上减任何常数都对，但 padding 的 exp(0) = 1 会被加进分母，结果就错了。</p>
:::

### 算：tl.max 和 tl.sum 就是 C2

[[here:solutions/t2_softmax.py@L25-L28|25–28 行]]

```python
x   = x - tl.max(x, axis=0)    # 向量 → 标量 → 广播回向量
num = tl.exp(x)
den = tl.sum(num, axis=0)
y   = num / den
```

`tl.max(x, axis=0)` 把整个 program 的向量归约成一个标量。你写了一个函数调用，编译器生成的就是 C2 里手写的那一套：

::: flow tl.max / tl.sum 编译出来的东西
cfg cw=215 nw=170
n v 0 0 "每线程持有若干元素" "线程内先归约" k=pass
n w 1 0 "warp 内" "shuffle 归约" k=teal
n b 2 0 "warp 之间" "shared memory + barrier" k=warn
n s 2 1.3 "一个标量" "再广播给所有线程" k=acc
e v -> w
e w -> b
e b -> s
cap: 这正是 C2 的前三层（没有第四层，因为一行只在一个 program 里）。num_warps = 1 时，warp 之间那一层就省掉了。G1 会讲编译器怎么根据 layout 决定这三层各做多少。
:::

### 写

[[here:solutions/t2_softmax.py@L30|30 行]] 用输出的 row stride 算地址，转回输入的 dtype，mask 保证 padding 位置不写。

## ④ host 端：BLOCK_SIZE 和 num_warps

[[here:solutions/t2_softmax.py@L33-L46|softmax（33–46 行）]]

```python
if x.stride(1) != 1: x = x.contiguous()       # 只要求行内连续
BLOCK_SIZE = triton.next_power_of_2(N)        # tl.arange 要 2 的幂
num_warps = 4 if BLOCK_SIZE <= 2048 else (8 if BLOCK_SIZE <= 8192 else 16)
softmax_kernel[(M,)](y, x, x.stride(0), y.stride(0), N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
```

- `next_power_of_2(1000) = 1024`，浪费 24 个位置；`next_power_of_2(1025) = 2048`，几乎浪费一半。N 刚好超过 2 的幂时最吃亏。
- **num_warps 随行长增加**，目的是让每个线程持有的元素数不要太多：

| N | BLOCK_SIZE | num_warps | 线程数 | 每线程持有的 fp32 |
|---|---|---|---|---|
| 1000 | 1024 | 4 | 128 | 8 |
| 4096 | 4096 | 8 | 256 | 16 |
| 16384 | 16384 | 16 | 512 | 32 |

每个线程持有的值都在寄存器里，x、exp 之后的 num 等中间结果还要多占几份。行越长，寄存器压力越大，超出时会溢出到 local memory，题目页 N = 16384 的测试就在观察这个现象。

## ⑤ 行太长怎么办：online softmax

一行放不进一个 program 时，只能分段处理。最直接的办法是两遍：第一遍求 max 和 sum，第二遍归一化，多读一次。**online softmax** 可以一遍完成：每读一段，维护到目前为止的最大值 m 和“以 m 为基准”的指数和 l。

遇到新的一段时：

```
m_new = max(m, max(段))
l     = l · exp(m − m_new) + Σ exp(段 − m_new)      # 旧的和要换算到新的基准
```

**算一个例子**，一行 [1, 2, 3, 0] 分成两段：

| 步骤 | m | l |
|---|---|---|
| 第 1 段 [1, 2] | 2 | e⁻¹ + e⁰ = 1.368 |
| 第 2 段 [3, 0]：m_new = 3 | 3 | 1.368 · e⁻¹ + (e⁰ + e⁻³) = 0.503 + 1.050 = **1.553** |
| 直接算验证 | 3 | e⁻² + e⁻¹ + e⁰ + e⁻³ = 0.135 + 0.368 + 1 + 0.050 = **1.553** ✓ |

`l · exp(m − m_new)` 这一项是关键：之前的和是以旧的最大值为基准算的，基准变了，要乘一个“换算系数”。T5 的 FlashAttention 每处理一块 K、V 就做一次这个更新。

::: callout analogy 换汇
你在旅行，每到一个国家就用当地货币记账（以当前的最大值为基准）。换到一个汇率更高的国家（遇到更大的 max）时，之前记的总额要按汇率换算成新货币（乘 exp(m − m_new)），再加上新国家的花销。走完全程，账本上的总额和你一开始就用同一种货币记账的结果完全相同。
:::

## ⑥ 对照真实实现

- **官方 tutorial 02**：[[triton:python/tutorials/02-fused-softmax.py@L85|softmax_kernel]] 是进阶版：grid 只开固定数量的 program，[[triton:python/tutorials/02-fused-softmax.py@L88-L90|每个 program 用 tl.range 循环处理多行]]，解决了题目页里 N = 256 时“一行一个 program 太碎”的问题。文件开头的 `naive_softmax` 和 [[triton:python/tutorials/02-fused-softmax.py@L68|显存流量估算]] 也值得看。
- **PyTorch 的 CUDA softmax**：[[pytorch:aten/src/ATen/native/cuda/SoftMax.cu|SoftMax.cu]]，里面 [[pytorch:aten/src/ATen/native/cuda/SoftMax.cu@L397|MaxFloat]] 和 [[pytorch:aten/src/ATen/native/cuda/SoftMax.cu@L416-L432|SumExpFloat（减 max 再 exp 求和）]] 是同样的思路，按行长选择不同的 kernel。
- 论文：[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)。

## ⑦ 常见坑与自测

- **padding 填 0**：全负数的行结果错误（见第 ③ 节的例子）。
- **假设 stride(0) == N**：切片视图的结果错位。
- **忘了 int64**：M × N 很大时地址溢出，读到别的行。
- **在 fp16 里直接 exp**：fp16 的 exp 在 x > 11 时就溢出（e¹¹ ≈ 59874，接近 65504）。减 max 之后虽然安全，但求和的精度仍然不够，所以统一转 fp32。

::: answer 自测 1：x = [0, 0, 0, 0]，N = 4，输出是多少？中间每一步的值？
max = 0，x − max = [0, 0, 0, 0]，exp = [1, 1, 1, 1]，sum = 4，y = [0.25, 0.25, 0.25, 0.25]。
:::

::: answer 自测 2：一行里全是 -inf（比如 attention 里整行被 mask 掉），会发生什么？
max = -inf，x − max = -inf − (-inf) = **NaN**，结果全是 NaN。真实的 attention kernel 会对这种情况特判（例如把 max 设成 0，或者最后把 0/0 处理成 0）。T5 和 L2 里会遇到这个边界情况。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/t2_softmax.py|solutions/t2_softmax.py]] · [[here:mine/t2_softmax.py|mine/t2_softmax.py]] · [[here:tests/test_t2_softmax.py|tests/test_t2_softmax.py]] · [[here:content/t2.md|content/t2.md]] · 前置：[C2 reduction 讲解](c2_reduce_sum.html)

**参考实现**：[[triton:python/tutorials/02-fused-softmax.py@L85|Triton tutorial 02]] · [[pytorch:aten/src/ATen/native/cuda/SoftMax.cu|PyTorch SoftMax.cu]] · [[cutlass:examples/python/CuTeDSL/notebooks/3_kernels/01_softmax.ipynb|CuTe DSL softmax notebook]]

**API**：[tl.max](https://triton-lang.org/main/python-api/generated/triton.language.max.html) · [tl.sum](https://triton-lang.org/main/python-api/generated/triton.language.sum.html) · [tl.exp](https://triton-lang.org/main/python-api/generated/triton.language.exp.html) · [triton.jit](https://triton-lang.org/main/python-api/generated/triton.jit.html)
