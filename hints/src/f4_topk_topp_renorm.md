title: F4 讲解：不排序的 top-k / top-p
lede: 采样前的 top-k / top-p 截断，教科书做法是排序加累积和；FlashInfer 从不排序，而是二分查找一个阈值。这一页先把“找阈值”变成一个单调函数的二分问题，再讲参考解的关键技巧：非负浮点数的二进制位模式和数值同序，所以可以对 int32 做精确二分。最后对比 FlashInfer 的三分 pivot 为什么快了 10 倍。

::: callout info 怎么读这一页
对照 [[here:solutions/f4_topk_topp_renorm.py|f4_topk_topp_renorm.py]] 读。kernel 的结构是“一个 program 处理一行，对整行扫很多遍”，每一遍都是 C2、T2 里那种行内归约。
:::

## ① 先看全貌：采样流水线里的 renorm

每个 decode step，每个请求都要从词表里采一个 token。V 是 15 万左右（Qwen 是 151936）。

::: flow 一次采样
cfg cw=210 nw=165
n l 0 0 "logits [B, V]" "lm_head 的输出" k=acc
n p 1 0 "softmax / temperature" "probs [B, V]" k=acc
n r 2 0 "top-k / top-p renorm" "截断并重新归一化" k=warn w=1.1
n s 3 0 "采样" "按概率抽一个 token" k=pass
e l -> p
e p -> r "本题"
e r -> s
cap: top-k：只保留概率最大的 k 个。top-p（nucleus）：从最大的开始往下取，直到累积概率达到 p 为止。截断之后除以保留部分的总和，让它重新加起来等于 1。
:::

**为什么不排序**：题目页的实测，B = 64、V = 151936 时光是 `torch.sort` 就要 1020 us，和一整层 decode 的时间相当。而我们其实不需要知道完整的顺序，只需要知道**一个阈值**：比它大的保留，比它小的丢掉。

**手算一个例子**：probs = [0.05, 0.40, 0.10, 0.30, 0.15]。

| | 阈值 t | 保留 | renorm 之后 |
|---|---|---|---|
| top-k，k = 2 | 0.30 | 0.40、0.30，和 = 0.70 | [0, 0.571, 0, 0.429, 0] |
| top-p，p = 0.8 | 0.15 | 0.40、0.30、0.15，和 = 0.85 | [0, 0.471, 0, 0.353, 0.176] |

top-p 的阈值是 0.15：从大到小累积 0.40 → 0.70 → **0.85**，到 0.15 时第一次达到 0.8。

## ② 找阈值 = 对单调函数做二分

定义两个函数：

```text
count(t) = 有多少个 x ≥ t          （top-k 用）
mass(t)  = Σ_{x ≥ t} x             （top-p 用）
```

t 越大，满足 x ≥ t 的元素越少，两个函数都**单调不增**。我们要找的是：

- top-k：最大的 t，使得 count(t) ≥ k
- top-p：最大的 t，使得 mass(t) ≥ p

单调函数上找“最后一个满足条件的点”，就是二分。维护 lo（满足条件）和 hi（不满足），每次取中点 mid，算一遍 count(mid) 或 mass(mid)，根据结果把 lo 或 hi 移到 mid。

上面例子 top-k = 2 时，在数值上二分的前几步：

| 步 | lo | hi | mid | count(mid) | ≥ 2？ |
|---|---|---|---|---|---|
| 0 | 0 | 0.40⁺ | 0.20 | 2（0.40、0.30） | 是 → lo = 0.20 |
| 1 | 0.20 | 0.40⁺ | 0.30 | 2 | 是 → lo = 0.30 |
| 2 | 0.30 | 0.40⁺ | 0.35 | 1 | 否 → hi = 0.35 |
| … | 0.30 | 区间不断缩小 | | | lo 停在 0.30 |

每一步都要扫一遍整行（V 个数），但不需要排序，也不需要额外的显存。

::: callout analogy 划录取分数线
一所学校要招 k 个人（top-k），或者要招到总分达到某个量为止（top-p）。教务处不需要把所有考生按分数排一遍名，只需要试一条分数线：线上有多少人？多了就把线往上提，少了就往下降。每试一次都要过一遍所有考生的卷子（扫一遍 V），但过卷子只需要数人数或加总分，比排名次简单得多。试几十次就能把线卡得精确到“刚好是某个考生的分数”。
:::

## ③ 关键技巧：对浮点数的二进制位做二分

在数值上二分有一个问题：要二分多少次才“够精确”？浮点数在 0 附近很密、在 1 附近很稀，固定次数的数值二分不一定能精确停在某个真实的概率值上。

参考解的办法：**对非负的 fp32，把它的 32 位当成 int32 来看，大小顺序和数值完全一致**。

| float | int32 位模式 |
|---|---|
| 0.0 | 0 |
| 最小的正 subnormal（≈1.4e-45） | 1 |
| 0.05 | 1028443341 |
| 0.15 | 1041865114 |
| 0.30 | 1050253722 |
| 0.40 | 1053609165 |
| 0.5 | 1056964608 |
| 1.0 | 1065353216 |

原因：IEEE 754 的 fp32 是“符号位 | 8 位指数 | 23 位尾数”。符号位为 0 时，指数在高位、尾数在低位，指数大的数一定更大，指数相同时尾数大的更大，这正好就是整数比较的规则。

于是二分的对象变成了整数区间 `[0, bits(max) + 1)`，长度小于 2³¹，**31 次二分一定把区间缩到 1**，lo 精确等于某个真实概率值的位模式。

::: flow 31 轮之后
cfg cw=230 nw=180
n i 0 0 "整数区间 [lo, hi)" "lo = 0，hi = bits(max) + 1" k=acc w=1.05
n r 1 0 "每轮：mid = (lo + hi) / 2" "扫一遍 V，判断 x_bits ≥ mid" k=teal w=1.15
n e 2.25 0 "hi − lo = 1" "lo 就是阈值的位模式" k=pass
e i -> r
e r -> e "31 轮"
cap: 比较 `x.to(tl.int32, bitcast=True) >= mid` 等价于 `x >= 位模式为 mid 的那个 float`，但不需要把 mid 转回 float，也不会遇到浮点二分的精度问题。bitcast 是重新解释同一组 32 位，不是数值转换。
:::

## ④ kernel 逐段讲

[[here:solutions/f4_topk_topp_renorm.py@L19-L67|_renorm_kernel（19–67 行）]]

一个 program 处理一行，整个 kernel 对这一行扫 1 + 31 + 2 = **34 遍**：

### 第 1 遍：求最大值，确定 hi

[[here:solutions/f4_topk_topp_renorm.py@L26-L31|26–31 行]]

```python
mx = tl.zeros([BLOCK], dtype=tl.float32)
for start in range(0, V, BLOCK):
    x = tl.load(base + start + offs, mask=start + offs < V, other=0.0)
    mx = tl.maximum(mx, x)
lo = 0
hi = tl.max(mx, axis=0).to(tl.int32, bitcast=True) + 1
```

V = 151936 放不进一个 block，所以用 `for start in range(0, V, BLOCK)` 分段扫，每段 4096 个。`mx` 是一个向量，每个位置记录“这个位置上见过的最大值”，扫完再 `tl.max` 归约成标量。后面的 count 和 mass 也是同样的“向量累加器 + 最后归约”模式。

初始的不变式：f(lo = 0) 一定满足条件（所有元素都 ≥ 0），f(hi) 一定不满足（没有元素 ≥ max⁺）。

### 31 轮二分

[[here:solutions/f4_topk_topp_renorm.py@L39-L55|39–55 行]]

```python
for _ in range(31):
    mid = lo + (hi - lo) // 2
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, V, BLOCK):
        x = tl.load(...)
        above = x.to(tl.int32, bitcast=True) >= mid
        acc += above.to(tl.float32) if IS_TOP_K else tl.where(above, x, 0.0)
    total = tl.sum(acc, axis=0)
    ok = total >= k   # 或 total >= p
    lo = tl.where(ok, mid, lo)
    hi = tl.where(ok, hi, mid)
```

- `lo + (hi − lo) // 2` 而不是 `(lo + hi) // 2`：后者在 int32 里可能溢出。
- **`IS_TOP_K` 是 constexpr**：同一个 kernel 编译成两个版本，`if IS_TOP_K` 在编译期就决定了，没有运行时分支。
- `lo`、`hi` 是标量，用 `tl.where` 更新，相当于标量的三元表达式。
- top-k 的 k 在 host 端转成了 float32（[[here:solutions/f4_topk_topp_renorm.py@L80|80 行]]），因为它要和 float 累加出来的计数比较。

### 最后 2 遍：求保留部分的和，再归一化写出

[[here:solutions/f4_topk_topp_renorm.py@L58-L67|58–67 行]]：保留 `x_bits ≥ lo` 的元素，先扫一遍求它们的和 denom，再扫一遍写出 `x / denom`（不保留的写 0）。

## ⑤ 为什么比 FlashInfer 慢 10 倍

题目页的实测（V = 151936）：

| | B = 1 | B = 64 |
|---|---|---|
| 参考解 top-k | 268 us | 312 us |
| FlashInfer top-k | 25 us | 88 us |
| 参考解 top-p | 268 us | 311 us |
| FlashInfer top-p | 76 us | 164 us |

两个原因：

**1. 扫的遍数。** 参考解固定 34 遍。FlashInfer 的 top-p renorm（[[flashinfer:include/flashinfer/sampling.cuh@L1709|TopPRenormProbKernel]]）每一遍同时试**两个** pivot（区间的 1/3 和 2/3 处，[[flashinfer:include/flashinfer/sampling.cuh@L1791-L1792|pivot_0、pivot_1]]），一遍就把区间缩到原来的 1/3。更重要的是它顺带记录 `min_gt_low`（大于 low 的最小真实概率）和 `max_le_high`（不超过 high 的最大真实概率），当两者相等时说明区间里已经只剩一个真实存在的值，立刻停止（[[flashinfer:include/flashinfer/sampling.cuh@L1781-L1787|循环不变式和停止条件]]）。真实的概率分布里，这通常几遍就收敛了。

**2. 一行只用一个 program。** B = 1 时整个 GPU 只有 1 个 program 在工作。每一遍扫 608 KB（151936 × 4 字节），数据基本在 L2 里，但 1 个 SM 的带宽有限。题目页提到，只把 BLOCK 从 1024 调到 4096、num_warps 从 4 调到 16，就快了 3.3 倍：一个 program 里的线程越多，扫一遍越快。

::: callout analogy 猜数字
二分查找像“猜一个 1 到 2³¹ 之间的数，每次只告诉你大了还是小了”：最多要猜 31 次。FlashInfer 的做法像是每次报两个数，还能顺便知道“区间里实际有哪些候选数”。如果对方告诉你“1000 到 2000 之间其实只有 1500 这一个数”，你立刻就知道答案，不用再一次次对半猜下去。
:::

## ⑥ 边界情况：并列

[[here:solutions/f4_topk_topp_renorm.py@L6|6 行]]注释说“对没有并列值的行”语义和 FlashInfer 相同。如果第 k 大和第 k+1 大的概率完全相等，阈值落在这个值上，`x ≥ t` 会把两个都保留下来，top-k 实际保留了 k + 1 个。排序的做法可以按下标打破并列，阈值法做不到。测试用的是随机概率，几乎不会出现完全相等的值。

## ⑦ 对照真实实现

- Python 接口：[[flashinfer:flashinfer/sampling.py@L2172|top_k_renorm_probs]] · [[flashinfer:flashinfer/sampling.py@L2083|top_p_renorm_probs]] · 一步完成截断和采样的 [[flashinfer:flashinfer/sampling.py@L1903|top_k_top_p_sampling_from_probs]]
- CUDA：[[flashinfer:include/flashinfer/sampling.cuh@L1709|TopPRenormProbKernel]]；采样 kernel [[flashinfer:include/flashinfer/sampling.cuh@L869|TopKSamplingFromProbKernel]]、[[flashinfer:include/flashinfer/sampling.cuh@L1230|TopKTopPSamplingFromProbKernel]]；每一遍归约的是 [[flashinfer:include/flashinfer/sampling.cuh@L117|ValueCount]]（质量和个数一起归约）
- 讲解文章：[Sorting-Free GPU Kernels for LLM Sampling（FlashInfer blog）](https://flashinfer.ai/2025/03/10/sampling.html)
- 浮点位模式：[IEEE 754 单精度格式](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)

## ⑧ 常见坑与自测

- **对负数用位模式比较**：负数的 int32 位模式顺序是反的。这里的输入是概率，都 ≥ 0，所以没问题；如果对 logits 做同样的事就要先处理符号位。
- **`(lo + hi) // 2` 溢出**：hi 接近 2³¹ 时相加溢出成负数。
- **mass 用 fp16 累加**：15 万个数加起来误差很大，top-p 的阈值会偏。
- **二分次数不够**：少于 31 次时区间没缩到 1，阈值不精确，保留的个数可能多一两个。

::: answer 自测 1：probs = [0.5, 0.2, 0.2, 0.1]，top-k = 2，结果是什么？
阈值落在 0.2 上（count(0.2) = 3 ≥ 2，count(0.2⁺) = 1 < 2），x ≥ 0.2 保留了 3 个：0.5、0.2、0.2，和 = 0.9，结果 [0.556, 0.222, 0.222, 0]。这就是第 ⑥ 节说的并列情况，保留了 k + 1 个。
:::

::: answer 自测 2：V = 151936，参考解处理一行一共要读多少字节？
34 遍 × 151936 × 4 字节 ≈ 20.7 MB。而这一行本身只有 608 KB：每一遍都在重复读同一份数据。FlashInfer 遍数少，而且能把一行缓存在片上，所以快得多。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/f4_topk_topp_renorm.py|solutions/f4_topk_topp_renorm.py]] · [[here:mine/f4_topk_topp_renorm.py|mine/f4_topk_topp_renorm.py]] · [[here:tests/test_f4_topk_topp_renorm.py|tests/test_f4_topk_topp_renorm.py]] · [[here:content/f4.md|content/f4.md]]

**FlashInfer**：[[flashinfer:flashinfer/sampling.py@L2172|sampling.py]] · [[flashinfer:include/flashinfer/sampling.cuh@L1709|sampling.cuh · TopPRenormProbKernel]] · [Sorting-Free Sampling 博客](https://flashinfer.ai/2025/03/10/sampling.html)
