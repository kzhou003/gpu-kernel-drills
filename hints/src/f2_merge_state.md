title: F2 讲解：merge_state，以 2 为底的 LSE 合并
lede: L3 的合并公式变成了 FlashInfer 的公开 API。这一页讲三件事：FlashInfer 为什么用以 2 为底的 LSE、它和自然对数怎么换算（用 L3 的同一个例子验证），merge_state 和 merge_states 两个 kernel 的下标怎么算，以及两个空状态合并时 NaN 是怎么来的、FlashInfer 用什么技巧挡掉它。

::: callout info 怎么读这一页
对照 [[here:solutions/f2_merge_state.py|f2_merge_state.py]] 读。合并公式的推导在 [L3 讲解](l3_split_kv_decode.html) 第 ③ 节，这里只讲 FlashInfer 版本的不同之处。
:::

## ① 先看全貌：attention 状态 (v, s)

对同一个 query、**一部分** key 做完 attention，得到一个“状态”：

- `v`：在这部分 key 上 softmax 归一化后的输出，形状 `[D]`
- `s`：这部分 key 的分数的 log-sum-exp

两个状态（同样的 query、不相交的 key 集合）可以**精确地**合并成一个，就像这部分 key 是一次算完的。合并操作满足交换律和结合律，所以可以按任意顺序、任意分组合并任意多个状态。

::: flow 哪些场景要合并 attention 状态
cfg cw=230 nw=175
n sk 0 0 "split-KV decode" "各分片合并（L3）" k=acc
n cas 0 1.2 "cascade attention" "共享前缀 + 各自后缀" k=acc
n cp 0 2.4 "chunked prefill / ring attn" "分块、分卡的部分结果" k=acc w=1.2
n m 1.3 1.2 "merge_state / merge_states" "(v, s) → (v, s)" k=warn w=1.15
n o 2.6 1.2 "最终输出" "和一次算完相同" k=pass
e sk -> m
e cas -> m
e cp -> m
e m -> o
cap: 只要 attention 被拆成了几部分来算，最后都要靠 (v, s) 合并。所以 FlashInfer 把它单独做成了 API，而且所有 attention kernel 都可以选择输出 s（return_lse=True）。
:::

::: callout analogy 合并两份调查问卷的结果
两个城市各做了一份满意度调查，各自报告“平均分”（v）和“样本量的对数”（s）。合并成全国结果时，按样本量加权：样本量大的城市权重大。只报平均分不报样本量就没法合并。用对数记样本量是因为它可能大得离谱（exp(1000)），对数之后就是一个普通大小的数。至于对数用 e 为底还是 2 为底，只是换了个单位，就像用“米”还是“英尺”记距离：只要两份报告用同一个单位，合并结果就一样。
:::

## ② 以 2 为底：为什么、怎么换算

FlashInfer 里所有的 s 都是**以 2 为底**的：

```text
s = log2( Σ_j exp(score_j) )        自然对数版本：lse = s · ln 2
```

**为什么用 2 为底**：GPU 有专门的 `ex2`（2 的幂）指令，`exp(x)` 其实是先算 `x · log2(e)` 再调 `ex2`。FlashInfer 的 attention kernel 从一开始就把 softmax scale 乘上 log2(e)，之后全程用 exp2 和 log2，省掉每个元素的一次乘法（T5 讲解里提到的 exp2 技巧）。输出的 s 自然也是以 2 为底。

换算只差一个常数：

| | 自然对数（L3） | 以 2 为底（FlashInfer） |
|---|---|---|
| 定义 | lse = ln Σ exp(score) | s = log₂ Σ exp(score) = lse / ln 2 |
| 权重 | exp(lse_a − m) | 2^(s_a − m) |
| 合并后的 s | m + ln(w_a + w_b) | m + log₂(w_a + w_b) |
| 换算 | lse = s × 0.6931 | s = lse × 1.4427 |

**用 L3 的例子验证**：分数 [1, 2, 3, 0]，分成 A = [1, 2]，B = [3, 0]。

| | A | B |
|---|---|---|
| 自然对数 lse | 2.313 | 3.049 |
| 以 2 为底 s = lse × 1.4427 | **3.337** | **4.398** |
| m = max(s) | 4.398 | |
| 权重 2^(s − m) | 2^(−1.061) = **0.479** | 1 |

权重和 L3 里用自然对数算出来的 0.479 完全一样，合并后的 v 当然也一样（26.21）。合并后的 s = 4.398 + log₂(1.479) = 4.398 + 0.565 = **4.963**；换回自然对数是 4.963 × 0.693 = 3.440，而直接算 ln(e¹ + e² + e³ + e⁰) = ln 31.19 = 3.440 ✓。

**最常见的 bug**：把 FlashInfer 返回的 s 当成自然对数的 lse 去和别的实现合并，或者在 merge 里用了 `exp` 而不是 `exp2`。权重会被系统性地拉偏，结果看起来“差不多对”，但测试过不了。

## ③ merge_state：两个状态

[[here:solutions/f2_merge_state.py@L19-L34|merge_state_kernel（19–34 行）]]

```python
pid = tl.program_id(0).to(tl.int64)       # 一个 (位置, head) 一个 program
sa, sb = tl.load(sa_ptr + pid), tl.load(sb_ptr + pid)
m  = tl.maximum(sa, sb)
wa, wb = tl.exp2(sa - m), tl.exp2(sb - m)
va = tl.load(va_ptr + pid * D + offs, ...).to(tl.float32)
vb = tl.load(vb_ptr + pid * D + offs, ...).to(tl.float32)
v  = (wa * va + wb * vb) / (wa + wb)
tl.store(so_ptr + pid, m + tl.log2(wa + wb))
```

- v 的形状是 `[S, H, D]`，s 是 `[S, H]`，都连续。把 (位置, head) 合并成一维的 pid：s 的地址就是 `pid`，v 的地址是 `pid · D`。grid 是 `(S · H,)`。
- 每个 program 读 2 个标量、2 × D 个向量元素，写 1 个标量、D 个元素。这是纯 memory-bound 的 kernel。
- 合并在 fp32 里做，v 写回时转回输入的 dtype，s 始终是 fp32。

## ④ merge_states：N 个状态

[[here:solutions/f2_merge_state.py@L37-L55|merge_states_kernel（37–55 行）]]

输入 v 是 `[S, N, H, D]`，s 是 `[S, N, H]`：N 个状态夹在位置和 head 之间。一个 program 负责一个 (pos, h)，一次读进它的全部 N 个状态：

```python
pos, h = pid // H, pid % H
s = tl.load(s_ptr + (pos * N + offs_n) * H + h, mask=n_mask, other=float("-inf"))     # [BLOCK_N]
m = tl.max(s, axis=0)
w = tl.exp2(s - m)                                    # padding 的状态 → 2^-inf = 0
v = tl.load(v_ptr + ((pos * N + offs_n[:, None]) * H + h) * D + offs_d[None, :], ...)   # [BLOCK_N, D]
out = tl.sum(w[:, None] * v, axis=0) / tl.sum(w, axis=0)
```

**地址怎么来的**：s 的形状是 `[S, N, H]`，元素 `s[pos, n, h]` 的偏移是 `(pos · N + n) · H + h`。

S = 2、N = 3、H = 2 时 s 在内存里的排列，pid = 3（pos = 1，h = 1）要读的是标出来的三个：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 70px repeat(12, minmax(42px, 1fr)); min-width: 640px">
<div class="lbl">偏移</div><div>0</div><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>7</div><div>8</div><div>9</div><div>10</div><div>11</div>
<div class="lbl">(pos,n,h)</div><div>0,0,0</div><div>0,0,1</div><div>0,1,0</div><div>0,1,1</div><div>0,2,0</div><div>0,2,1</div><div>1,0,0</div><div class="c3">1,0,1</div><div>1,1,0</div><div class="c3">1,1,1</div><div>1,2,0</div><div class="c3">1,2,1</div>
</div></div>
<p class="legend">(1·3 + n)·2 + 1 = 7, 9, 11。一个 program 读的 N 个 s 在内存里相隔 H 个元素，不连续。N 一般很小（几个到几十个分片），这点跨步无所谓。</p>
:::

**`BLOCK_N = max(2, next_power_of_2(N))`**：`tl.arange` 的长度要是 2 的幂；N = 1 时也取 2，多出来的一个用 −inf 填充，权重为 0。

::: flow merge_states 的一个 program
cfg cw=225 nw=175
n s 0 0 "读 N 个 s" "跨步 H，padding = −inf" k=acc
n w 1 0 "w = 2^(s − max s)" "[BLOCK_N]" k=teal
n v 1 1.3 "读 N 个 v" "[BLOCK_N, D]" k=acc
n o 2 0.65 "Σ w·v / Σ w" "写 v_out、s_out" k=pass
e s -> w
e w -> o
e v -> o
cap: 和 L3 的 merge kernel 是同一个算法，只是 s 换成了以 2 为底，而且张量布局是 FlashInfer 的 [S, N, H, D]。
:::

## ⑤ 边界情况：两个空状态

如果某个状态对应的 key 集合是空的（比如某个分片没有分到任何页），它的 s = −inf，v = 0。

- 一个空状态和一个正常状态合并：m 是正常状态的 s，空状态的权重 2^(−inf − m) = 0，结果就是正常状态。✓
- **两个空状态合并**：m = −inf，`s_a − m = (−inf) − (−inf) = NaN`，权重是 NaN，输出全是 NaN。

参考解没有处理第二种情况，因为测试里不会出现。FlashInfer 的 [[flashinfer:include/flashinfer/attention/state.cuh@L53-L65|state_t::merge]] 用了一个小技巧：`ptx_exp2(max(m_prev − m, −inf))`。`fmax(NaN, −inf)` 在 CUDA 里返回 −inf（fmax 的规则是有一个参数是 NaN 时返回另一个），于是权重变成 2^(−inf) = 0，NaN 被挡掉了。源码注释写的正是“max() drops the NaN from merging two empty states”。[[flashinfer:include/flashinfer/attention/cascade.cuh@L58-L59|MergeStateKernel]] 里也是同样的写法。

## ⑥ cascade：merge 的主要用户

题目页的面试追问讲了 cascade attention：64 个请求共享一个 8K token 的系统 prompt 时，

::: flow cascade attention
cfg cw=230 nw=175
n q 0 0.65 "64 个请求的 query" "" k=acc
n pre 1 0 "对共享前缀做一次 attention" "前缀 KV 只读一次" k=pass w=1.15
n suf 1 1.3 "各自对后缀做 attention" "每个请求读自己的" k=teal w=1.1
n m 2.3 0.65 "merge_state" "(v₁, s₁) + (v₂, s₂)" k=warn
e q -> pre
e q -> suf
e pre -> m
e suf -> m
cap: 不用 cascade 时，前缀的 KV 要被读 64 次。cascade 之后读 1 次，而且 64 个 query 一起对同一块 KV 做 attention，可以组成矩阵乘、用上 Tensor Core。merge 本身只读写 O(B·H·D) 的数据。
:::

FlashInfer 的接口是 [[flashinfer:flashinfer/cascade.py@L226|MultiLevelCascadeAttentionWrapper]]，内部就是先分层做 attention，再调 merge。

## ⑦ 对照真实实现

- Python 接口：[[flashinfer:flashinfer/cascade.py@L42|merge_state]] · [[flashinfer:flashinfer/cascade.py@L109|merge_state_in_place]] · [[flashinfer:flashinfer/cascade.py@L170|merge_states]]
- CUDA：[[flashinfer:include/flashinfer/attention/cascade.cuh@L45|MergeStateKernel]] · [[flashinfer:include/flashinfer/attention/cascade.cuh@L220|MergeStatesKernel]] · N 很大时的 [[flashinfer:include/flashinfer/attention/cascade.cuh@L283|MergeStatesLargeNumIndexSetsKernel]]
- 状态结构体：[[flashinfer:include/flashinfer/attention/state.cuh@L29|state_t]]，[[flashinfer:include/flashinfer/attention/state.cuh@L45|get_lse() = m + log2(d)]]
- 题目页提到的性能差距（156 us 对 135 us）来自向量化：FlashInfer 用 [[flashinfer:include/flashinfer/vec_dtypes.cuh@L724|vec_t]] 每次读 16 字节，每个线程处理多个元素。

## ⑧ 常见坑与自测

- **exp 和 exp2 混用**：最常见，结果偏差不大但系统性错误。
- **s 用了 fp16**：s 的值可能是几十到几百，fp16 的精度不够。
- **merge_states 的地址按 `[S, H, N]` 算**：布局是 `[S, N, H]`，N 在 H 前面。
- **v 没转 fp32 就加权**：fp16 的加权求和在 N 较大时误差明显。

::: answer 自测 1：s_a = 10，s_b = 10（以 2 为底），v_a = [1, 0]，v_b = [0, 1]。合并后的 v 和 s 是多少？
m = 10，w_a = w_b = 1。v = [0.5, 0.5]，s = 10 + log₂ 2 = **11**。两部分的“分母”一样大，合并后总分母翻倍，以 2 为底的对数正好加 1。
:::

::: answer 自测 2：L3 的实现输出的是自然对数的 lse = 5。要把它交给 FlashInfer 的 merge_state，s 应该填多少？
s = 5 × 1.4427 = 7.21。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/f2_merge_state.py|solutions/f2_merge_state.py]] · [[here:mine/f2_merge_state.py|mine/f2_merge_state.py]] · [[here:tests/test_f2_merge_state.py|tests/test_f2_merge_state.py]] · [[here:content/f2.md|content/f2.md]]

**FlashInfer**：[[flashinfer:flashinfer/cascade.py@L42|cascade.py]] · [[flashinfer:include/flashinfer/attention/cascade.cuh@L45|cascade.cuh]] · [[flashinfer:include/flashinfer/attention/state.cuh@L29|state.cuh]] · [文档 · cascade](https://docs.flashinfer.ai/api/cascade.html)

**背景**：[FlashInfer 论文](https://arxiv.org/abs/2501.01005)（讲了 attention 状态的合并）· [Cascade Inference 博客](https://flashinfer.ai/2024/02/02/cascade-inference.html)
