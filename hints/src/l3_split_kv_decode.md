title: L3 讲解：split-KV decode 与 LSE 合并
lede: L2 在 B = 1、长上下文时只用了 5% 的 GPU。Flash-Decoding 把每个序列的 KV 切成几段并行算，再用 log-sum-exp 精确合并。这一页逐段读参考解：页怎么分给各个分片、空分片为什么要写 −inf、合并公式为什么成立（用 T5 的同一个例子手算验证），以及分几片最合适。

::: callout info 怎么读这一页
对照 [[here:solutions/l3_split_kv_decode.py|l3_split_kv_decode.py]] 读。第一个 kernel 的循环体和 [L2](l2_paged_decode.html) 一字不差，新内容是“分片的范围”和“合并”。第 ③ 节的 LSE 合并在 F2（merge_state）里还会再出现一次。
:::

## ① 先看全貌：两趟 kernel

::: flow Flash-Decoding 的两个 kernel
cfg cw=225 nw=175
n q 0 0.65 "一个序列、一个 KV head" "ctx = 32768 个 token" k=acc w=1.05
n s0 1 0 "分片 0" "页 0–31 → o₀, lse₀" k=teal
n s1 1 1.3 "分片 63" "页 2016–2047 → o₆₃, lse₆₃" k=teal w=1.1
n m 2 0.65 "merge kernel" "按 lse 加权求和" k=warn
n o 3 0.65 "最终输出" "[B, Hq, D]" k=pass
e q -> s0
e q -> s1
e s0 -> m "fp32"
e s1 -> m "fp32"
e m -> o
cap: 第一个 kernel 的 grid 是 (B, Hkv, NUM_SPLITS)，每个 program 只走自己那段页，输出“这一段内部归一化过的 o”和“这一段的 lse”。第二个 kernel 的 grid 是 (B, Hq)，把各段合并。
:::

B = 1、Hkv = 8 时，L2 只有 8 个 program；分 64 片后有 512 个，170 个 SM 都有活干。题目页实测：1838 us → **96 us**，快了 19 倍。

::: callout analogy 合并各班的平均分
年级要算全体学生的平均分。每个班（分片）各自算出本班平均分（o_s，已经在班内归一化过）。直接把各班平均分再平均一次是错的：50 人的班和 10 人的班不能一样算。必须同时报上各班的人数（exp(lse_s)，就是这一段的 softmax 分母），按人数加权。lse 就是那个“人数”，只是取了对数，防止数字太大。
:::

## ② 分片：每个 program 走哪几页

[[here:solutions/l3_split_kv_decode.py@L30-L39|30–39 行]]

```python
split = tl.program_id(2)
num_pages = tl.cdiv(ctx, PAGE)
pages_per_split = tl.cdiv(num_pages, NUM_SPLITS)
p_start = split * pages_per_split
p_end   = tl.minimum(p_start + pages_per_split, num_pages)    # 可能 ≤ p_start：空分片
```

每个分片负责一段**连续**的页 `[p_start, p_end)`。因为 batch 里每个序列长度不同，分片范围是在 kernel 里根据各自的 ctx 算的。

两个例子（PAGE = 16，NUM_SPLITS = 4）：

::: raw
<p class="legend">ctx = 1000：63 页，每片 ⌈63/4⌉ = 16 页</p>
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 16fr 16fr 16fr 15fr; min-width: 480px">
<div class="c0">分片 0<small>页 0–15</small></div><div class="c1">分片 1<small>页 16–31</small></div><div class="c2">分片 2<small>页 32–47</small></div><div class="c3">分片 3<small>页 48–62（15 页）</small></div>
</div></div>
<p class="legend">ctx = 37：3 页，每片 ⌈3/4⌉ = 1 页</p>
<div class="strip-scroll"><div class="cells" style="grid-template-columns: repeat(4, 1fr); min-width: 480px">
<div class="c0">分片 0<small>页 0</small></div><div class="c1">分片 1<small>页 1</small></div><div class="c2">分片 2<small>页 2（5 个 token）</small></div><div class="cx">分片 3<small>p_start = 3，p_end = 3：空</small></div>
</div></div>
<p class="legend">短序列会出现空分片：它的循环一次都不执行。grid 是对整个 batch 统一开的，所以 kernel 必须能处理这种情况（第 ④ 节）。</p>
:::

循环体（[[here:solutions/l3_split_kv_decode.py@L53-L68|53–68 行]]）和 L2 完全相同，只是 `range(p_start, p_end)` 代替了 `range(0, num_pages)`。

## ③ LSE：合并需要的那个数

每个分片结束时手里有 m_s（这一段的最大分数）、l_s（以 m_s 为基准的指数和）和 acc_s。它输出：

```text
o_s   = acc_s / l_s                 # 这一段内部归一化过的输出
lse_s = m_s + log(l_s)              # = log( Σ_{j∈段 s} exp(score_j) )
```

`exp(lse_s) = exp(m_s) · l_s` 正好是这一段**没有减 max 的原始分母**。整体的 softmax 输出是：

```text
out = Σ_s exp(lse_s) · o_s  /  Σ_s exp(lse_s)
```

为了不溢出，和 softmax 一样先减去 max：`w_s = exp(lse_s − max lse)`，`out = Σ w_s · o_s / Σ w_s`。这就是 merge kernel 做的事。

**手算验证**（和 T5 讲解里的例子相同）：1 个 query，分数 s = [1, 2, 3, 0]，v = [10, 20, 30, 40]，分成两片。

| | 分片 0：s = [1, 2] | 分片 1：s = [3, 0] |
|---|---|---|
| m_s | 2 | 3 |
| l_s | e⁻¹ + 1 = 1.368 | 1 + e⁻³ = 1.050 |
| o_s = Σ p·v / l_s | (0.368·10 + 20) / 1.368 = **17.31** | (30 + 0.050·40) / 1.050 = **30.47** |
| lse_s = m_s + ln l_s | 2 + 0.313 = **2.313** | 3 + 0.049 = **3.049** |
| w_s = exp(lse_s − 3.049) | e⁻⁰·⁷³⁵ = 0.479 | 1 |

合并：(0.479 × 17.31 + 1 × 30.47) / (0.479 + 1) = 38.77 / 1.479 = **26.21**，和 T5 里一次算完的结果相同。

::: flow merge kernel 做了什么（一个 (b, h)）
cfg cw=220 nw=170
n l 0 0 "读 lse[b, h, 0..S−1]" "S 个 fp32" k=acc
n m 1 0 "m = max(lse)" "有限值" k=teal
n w 2 0 "w = exp(lse − m)" "空分片 → 0" k=teal
n o 2 1.3 "读 o_part[b, h, :, :]" "[S, D]" k=acc
n out 1 1.3 "Σ w·o / Σ w" "写 out[b, h]" k=pass
e l -> m
e m -> w
e w -> out
e o -> out
cap: 每个 (b, h) 一个 program，num_warps = 1 就够了：只有 S × D 个数要读（S = 64、D = 128 时 32 KB），计算量很小。
:::

## ④ 空分片：为什么要写 −inf

[[here:solutions/l3_split_kv_decode.py@L72-L74|72–74 行]]

```python
has = l_i > 0
o   = acc / tl.where(has, l_i, 1.0)[:, None]
lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))
```

空分片的循环一次都没执行：m_i = −inf，l_i = 0，acc = 0。

- 直接算 `acc / l_i` 是 0/0 = **NaN**，合并时 NaN 乘以权重 0 还是 NaN，会污染整个输出。所以先把分母替换成 1：o = 0 / 1 = 0。
- lse 写成 −inf，合并时 `w = exp(−inf − m) = 0`，这个分片完全不起作用。
- **为什么要嵌套 `tl.where`**：`tl.where` 两边的表达式都会被计算。`tl.log(l_i)` 在 l_i = 0 时是 −inf，这里虽然最后会被外层的 where 选掉，但把 log 的输入也换成 1 更保险，避免中间出现 −inf + (−inf) 之类的值。

merge kernel 的 [[here:solutions/l3_split_kv_decode.py@L91-L93|91–93 行]]：padding 的分片（offs_s ≥ NUM_SPLITS）也用 −inf 填充，权重同样为 0。注释说明了 m 一定是有限值：context_lens ≥ 1，所以至少分片 0 不为空。

## ⑤ 中间结果用 fp32

[[here:solutions/l3_split_kv_decode.py@L114-L116|114–116 行]]：o_part 是 `[B, Hq, NUM_SPLITS, D]` 的 fp32，lse 是 `[B, Hq, NUM_SPLITS]` 的 fp32。它们还要再加权合并一次，用 fp16 存会多损失一次精度。代价是多写多读一些字节：B = 1、Hq = 32、64 片、D = 128 时 o_part 是 1 MB，和 32768 个 token 的 KV（128 MB）相比可以忽略。

## ⑥ 分几片最合适

题目页实测（us）：

| B × ctx | L2 | 4 片 | 16 片 | 64 片 |
|---|---|---|---|---|
| 1 × 32768 | 1838 | 492 | 155 | **96** |
| 8 × 4096 | 261 | 103 | **95** | 107 |
| 64 × 1024 | **176** | 186 | 204 | 262 |

经验规则：让 `B × Hkv × NUM_SPLITS` 达到 SM 数的几倍，同时每片至少有若干页。

| B × ctx | 不分片的 program 数（B × Hkv） | 合适的分片 | 原因 |
|---|---|---|---|
| 1 × 32768 | 8 | 64 → 512 个 program | 并行度严重不足，每片还有 32 页 |
| 8 × 4096 | 64 | 16 → 1024 个 | 适中 |
| 64 × 1024 | 512 | 不分 | program 已经够多；分片只增加中间结果和第二次 launch |

::: callout analogy 搬家公司派车
一户人家（B = 1）要搬一屋子东西（32768 个 token），公司只派了 8 个人（8 个 program），另外 162 个人在休息。把东西按房间分成 64 份，派 512 个人同时搬，最后在新家汇总（merge）。但如果今天有 64 户人家要搬（B = 64），每户派 8 个人就已经把所有人都用上了，再拆分只会增加汇总的麻烦。
:::

真实系统在**每一步 decode 之前**根据 B、各序列的 ctx 和 SM 数决定分片：FlashInfer decode 的 `plan()` 就是做这件事的，而且能给不同长度的序列分不同数量的片（F1）。

## ⑦ 对照真实实现

- **FlashInfer 的 merge**：Python 接口 [[flashinfer:flashinfer/cascade.py@L42|merge_state]] 和 [[flashinfer:flashinfer/cascade.py@L170|merge_states]]（F2 的题目），CUDA 实现 [[flashinfer:include/flashinfer/attention/cascade.cuh@L45|MergeStateKernel]]。注意它用的是**以 2 为底**的 lse（[[flashinfer:include/flashinfer/attention/state.cuh@L45|get_lse = m + log2(d)]]，配合 [[flashinfer:include/flashinfer/attention/cascade.cuh@L58-L59|exp2]]），F2 会专门处理这个区别。在线合并的数据结构是 [[flashinfer:include/flashinfer/attention/state.cuh@L29|state_t]]，它的 [[flashinfer:include/flashinfer/attention/state.cuh@L53|merge()]] 就是这里的合并公式。
- **vLLM 的 Triton decode**：也是两阶段，第二阶段 [[vllm:vllm/v1/attention/ops/triton_decode_attention.py@L575|_fwd_kernel_stage2]] 按 split 顺序在线合并（维护 e_max，边读边重缩放），和这里一次读完再合并是等价的两种写法。
- **FlashInfer 的 decode kernel** 在分片时写出的 lse：[[flashinfer:include/flashinfer/attention/decode.cuh@L373-L374|按 kv_chunk_idx 写出 lse]]。
- 原始文章：[Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/)（PyTorch blog）。

## ⑧ 常见坑与自测

- **只存 o 不存 lse**：无法正确合并（题目页面试追问）。
- **空分片写了 NaN**：短序列的输出变成 NaN，长序列看起来正常，很难查。
- **合并时没减 max**：lse 是几十量级的数时 exp 溢出。
- **中间结果用 fp16**：误差比 L2 大一截，测试容差内可能还能过，但没必要。
- **分片范围算成 `split * ctx / NUM_SPLITS`（按 token 切）**：可以，但切点不在页边界上，一页会被两个分片各读一半；按页切更简单。

::: answer 自测 1：ctx = 100，PAGE = 16，NUM_SPLITS = 4。各分片负责哪些页？有空分片吗？
num_pages = 7，pages_per_split = 2。分片 0：页 0–1，分片 1：页 2–3，分片 2：页 4–5，分片 3：页 6（只有 4 个 token）。没有空分片。
:::

::: answer 自测 2：两个分片的 o 都是 [1, 1]，lse 分别是 0 和 10。合并结果是多少？只看 o 能判断吗？
o 都相同，所以合并结果也是 [1, 1]，不管权重如何。但如果 o 分别是 [1, 0] 和 [0, 1]，结果就是 (e⁻¹⁰·[1, 0] + [0, 1]) / (e⁻¹⁰ + 1) ≈ [0.00005, 0.99995]：几乎完全由第二个分片决定。只看 o 无法知道这个比例。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/l3_split_kv_decode.py|solutions/l3_split_kv_decode.py]] · [[here:mine/l3_split_kv_decode.py|mine/l3_split_kv_decode.py]] · [[here:tests/test_l3_split_kv_decode.py|tests/test_l3_split_kv_decode.py]] · [[here:content/l3.md|content/l3.md]]

**FlashInfer**：[[flashinfer:flashinfer/cascade.py@L42|merge_state]] · [[flashinfer:include/flashinfer/attention/cascade.cuh@L45|MergeStateKernel]] · [[flashinfer:include/flashinfer/attention/state.cuh@L29|state_t]]

**vLLM**：[[vllm:vllm/v1/attention/ops/triton_decode_attention.py@L575|_fwd_kernel_stage2]]

**文章**：[Flash-Decoding（PyTorch blog）](https://pytorch.org/blog/flash-decoding/) · [FlashInfer 论文](https://arxiv.org/abs/2501.01005)
