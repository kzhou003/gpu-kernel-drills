title: T5 讲解：FlashAttention forward
lede: 把 T2 的 online softmax 和 T4 的 tl.dot 拼在一起，就是 FlashAttention。这一页先算清楚 naive attention 到底多写了多少字节，再逐段读参考解：program 怎么分、q 为什么只读一次、K 为什么转置着读、(m, l, acc) 三个状态怎么更新（附一个能手算的例子），以及 causal 怎么跳过一半的块。

::: callout info 怎么读这一页
对照 [[here:solutions/t5_flash_attn.py|t5_flash_attn.py]] 读。前置知识：[T2 讲解](t2_softmax.html) 第 ⑤ 节的 online softmax，[T4 讲解](t4_matmul.html) 的二维指针块和 tl.dot。
:::

## ① 先看全貌：省的不是 FLOPs，是显存读写

attention：`O = softmax(Q Kᵀ · scale) V`，Q、K、V、O 的形状都是 `[B, H, S, D]`。

naive 实现分三步，每一步都把中间结果写回显存：

::: flow naive attention vs FlashAttention
cfg cw=210 nw=165
n q1 0 0 "S = Q·Kᵀ" "写 S×S 矩阵" k=todo
n p1 1 0 "P = softmax(S)" "读 S×S，写 S×S" k=todo
n o1 2 0 "O = P·V" "读 S×S" k=todo
n q2 0 1.4 "Q 的一块" "读一次，留在寄存器" k=pass
n p2 1 1.4 "K、V 逐块流过" "分数只在片上存在" k=pass
n o2 2 1.4 "O 的一块" "写一次" k=pass
g 0 0 2 0 "naive：S×S 的矩阵在显存里来回 3 趟" k=todo
g 0 1.4 2 1.4 "FlashAttention：S×S 从不落地" k=pass
e q1 -> p1
e p1 -> o1
e q2 -> p2
e p2 -> o2
cap: FLOPs 一样多（FlashAttention 还多了一点重缩放），但显存读写从 O(S²) 降到 O(S·D)。attention 在长序列下受显存带宽限制，所以 IO 少了就快了。
:::

算一下题目页的测试规模（B = 4、H = 16、S = 4096、D = 64，fp16）：

| | 字节数 |
|---|---|
| Q、K、V、O 各一份：4 × B·H·S·D × 2 | 4 × 4·16·4096·64 × 2 = **128 MB** |
| 一份 S×S 的分数矩阵：B·H·S² × 2 | 4·16·4096² × 2 = **2 GB** |

naive 实现要写一次 S、读一次 S、写一次 P、读一次 P：分数矩阵和 softmax 结果加起来写 4 GB、读 4 GB，是输入输出本身的 64 倍。S 再翻倍，这一项就变成 4 倍。

::: callout analogy 阅卷
一个老师（program）要给 64 个学生（一块 query）各写一份评语（O），每份评语要参考全校 4096 份作业（K、V）。naive 做法是先把“每个学生 × 每份作业”的相关度打成一张 64 × 4096 的大表，存进档案室（显存），再取出来归一化，再取出来加权汇总。FlashAttention 的做法是：作业一摞一摞地送来（BLOCK_N 份），老师边看边更新手里 64 份评语的草稿（acc），从不制作那张大表。代价是每来一摞，如果发现了比之前更相关的作业（max 变大），要把之前的草稿按比例缩一下（alpha）。
:::

## ② program 怎么分

[[here:solutions/t5_flash_attn.py@L27-L35|27–35 行]] · host 端 [[here:solutions/t5_flash_attn.py@L93|grid（93 行）]]

```python
grid = (triton.cdiv(S, block_m), B * H)        # 二维 grid

start_m = tl.program_id(0)                     # 第几块 query
bh      = tl.program_id(1)                     # 第几个 (batch, head)
off_b, off_h = bh // H, bh % H
base = off_b.to(tl.int64) * stride_b + off_h.to(tl.int64) * stride_h
```

- grid 的第 0 维是 query 块，第 1 维把 batch 和 head 合并成一维。S = 4096、BLOCK_M = 64、B·H = 64 时一共 64 × 64 = 4096 个 program。
- 不同的 (batch, head) 之间完全独立；同一个 (batch, head) 里不同的 query 块也独立，它们读同一份 K、V，但各算各的输出。
- `base` 是这个 (batch, head) 在张量里的起点，用 int64 算（B·H·S·D 很容易超过 2³¹）。之后 q、k、v、o 的地址都是 `base + 位置 × stride_s + d`，因为 host 端已经 `.contiguous()`，四个张量的 stride 相同，最后一维 stride 是 1。

## ③ 循环外：q 只读一次，三个状态初始化

[[here:solutions/t5_flash_attn.py@L38-L43|38–43 行]]

```python
q   = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :], mask=offs_m[:, None] < S, other=0.0)
m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # 每行到目前为止的最大分数
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # 每行的 Σ exp(score − m_i)
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # 每行的 Σ exp(score − m_i) · v
```

q 是 `[BLOCK_M, D]` = 64 × 64 个 fp16，整个循环里一直留在寄存器里，每一块 K 都和它做一次 tl.dot。这就是 query 块内的复用。

## ④ 主循环：一块 K、V 进来时发生什么

[[here:solutions/t5_flash_attn.py@L52-L76|52–76 行]]

::: flow 循环的一轮（一块 BLOCK_N 个 key）
cfg cw=215 nw=170
n k 0 0 "读 K 块（转置）" "[D, BLOCK_N]" k=acc
n s 1 0 "s = q·k × scale" "[BLOCK_M, BLOCK_N]" k=acc
n mk 2 0 "mask" "越界 / causal → −inf" k=warn
n m 2 1.3 "m_new = max(m_i, rowmax(s))" "" k=teal w=1.25
n p 1 1.3 "p = exp(s − m_new)" "alpha = exp(m_i − m_new)" k=teal
n l 0 1.3 "l_i = l_i·alpha + Σp" "" k=teal
n v 0 2.6 "读 V 块" "[BLOCK_N, D]" k=acc
n a 1 2.6 "acc = acc·alpha + p·v" "tl.dot，Tensor Core" k=pass w=1.1
e k -> s
e s -> mk
e mk -> m
e m -> p
e p -> l
e l -> v
e v -> a
cap: 每一轮有两次 tl.dot（q·k 和 p·v），中间是 online softmax 的状态更新。s 和 p 都只是 BLOCK_M × BLOCK_N 的小块，用完就丢。
:::

### 读 K：直接读成转置的

[[here:solutions/t5_flash_attn.py@L55-L57|55–57 行]]

```python
k = tl.load(K + base + cols[None, :] * stride_s + offs_d[:, None], mask=cols[None, :] < S, other=0.0)
s = tl.dot(q, k) * sm_scale
```

K 在内存里是 `[S, D]`。这里把位置放在列（`cols[None, :]`）、维度放在行（`offs_d[:, None]`），读出来就是 `[D, BLOCK_N]`，也就是 Kᵀ 的一块。于是 `tl.dot(q, k)` 是普通的 `[BLOCK_M, D] @ [D, BLOCK_N]`，不需要额外转置。这是 T4 讲过的“用 stride 表达转置”。

### mask

[[here:solutions/t5_flash_attn.py@L59-L62|59–62 行]]：key 位置超出 S 的设成 −inf；causal 时还要求 `query 位置 ≥ key 位置`。−inf 经过 exp 变成 0，相当于这个 key 不存在。

### online softmax 更新

[[here:solutions/t5_flash_attn.py@L67-L76|67–76 行]]

```python
m_new = tl.maximum(m_i, tl.max(s, axis=1))     # 每行的新最大值
p     = tl.exp(s - m_new[:, None])             # 以新最大值为基准的权重
alpha = tl.exp(m_i - m_new)                    # 旧状态的换算系数
l_i   = l_i * alpha + tl.sum(p, axis=1)
acc   = acc * alpha[:, None]
acc   = tl.dot(p.to(v.dtype), v, acc)          # acc += p @ v
m_i   = m_new
```

和 T2 的 online softmax 一样，只多了一个 acc：分母 l 和分子 acc 都是“以 m 为基准的加权和”，基准变了，两者都要乘 alpha。

**手算一个例子**：1 个 query，4 个 key，BLOCK_N = 2，D = 1。分数 s = [1, 2, 3, 0]，v = [10, 20, 30, 40]。

| 步骤 | m | p | l | acc |
|---|---|---|---|---|
| 初始 | −inf | | 0 | 0 |
| 第 1 块：s = [1, 2] | 2 | [e⁻¹, 1] = [0.368, 1] | 0·0 + 1.368 = **1.368** | 0 + 0.368·10 + 1·20 = **23.68** |
| 第 2 块：s = [3, 0]，alpha = e⁻¹ = 0.368 | 3 | [1, e⁻³] = [1, 0.050] | 1.368·0.368 + 1.050 = **1.553** | 23.68·0.368 + 30 + 0.050·40 = **40.70** |
| 结束：o = acc / l | | | | 40.70 / 1.553 = **26.21** |
| 直接算 | 权重 ∝ [e⁻², e⁻¹, 1, e⁻³] = [0.135, 0.368, 1, 0.050]，Σ = 1.553 | | | (1.35 + 7.36 + 30 + 1.99) / 1.553 = **26.21** ✓ |

第一轮 m_i = −inf，alpha = exp(−inf − m_new) = 0，正好把初始的 0 清掉，不需要特判。

**为什么不会出现 NaN**：如果某一行在第一块里所有 key 都是 −inf，m_new 也会是 −inf，`s − m_new` 就是 (−inf) − (−inf) = NaN。注释（64–66 行）说明了为什么这里不会发生：每个合法的 query 行至少能看到 key 0（causal 时 query 位置 ≥ 0），而 key 0 在第一块里。L2 的 paged decode 里如果有整行被 mask 掉的情况，就要显式处理。

### p 转成 fp16 再乘

`tl.dot(p.to(v.dtype), v, acc)`：p 是 fp32，Tensor Core 要求两个输入都是 fp16/bf16，所以先转换。p 的值都在 [0, 1] 之间，转成 fp16 的精度损失很小，累加仍在 fp32。这是所有 FlashAttention 实现的标准做法。

## ⑤ causal：跳过一半的块

[[here:solutions/t5_flash_attn.py@L47-L50|47–50 行]]

```python
hi = tl.minimum((start_m + 1) * BLOCK_M, S) if CAUSAL else S
for start_n in range(0, hi, BLOCK_N): ...
```

causal 时 query i 只能看 key 0..i。第 start_m 块 query 的最后一个位置是 (start_m + 1)·BLOCK_M − 1，所以 key 只需要循环到这里。S = 4 × 64，BLOCK_M = BLOCK_N = 64 时：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 120px repeat(4, minmax(70px, 1fr)); min-width: 460px">
<div class="lbl"></div><div class="lbl" style="text-align:center">key 块 0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div><div class="lbl" style="text-align:center">3</div>
<div class="lbl">query 块 0</div><div class="c1">对角<small>需要 mask</small></div><div class="cx">跳过</div><div class="cx">跳过</div><div class="cx">跳过</div>
<div class="lbl">query 块 1</div><div class="c2">全部可见</div><div class="c1">对角</div><div class="cx">跳过</div><div class="cx">跳过</div>
<div class="lbl">query 块 2</div><div class="c2">全部可见</div><div class="c2">全部可见</div><div class="c1">对角</div><div class="cx">跳过</div>
<div class="lbl">query 块 3</div><div class="c2">全部可见</div><div class="c2">全部可见</div><div class="c2">全部可见</div><div class="c1">对角</div>
</div></div>
<p class="legend">16 个块里只算 10 个；S 越大越接近一半。绿色块里所有位置都可见，其实不需要 tl.where；只有橙色的对角块需要逐元素 mask。题目页建议把循环拆成两段来省掉绿色块里的 mask。</p>
:::

`CAUSAL` 是 constexpr，所以 `if CAUSAL:` 在编译期就决定了，causal 和非 causal 是两个不同的 kernel，没有运行时分支。

## ⑥ 结束和 host 端

- [[here:solutions/t5_flash_attn.py@L78-L80|78–80 行]]：`o = acc / l_i`，转回 fp16/bf16 写出。如果要支持 backward 或 L3 的分片合并，还要存 `lse = m_i + log(l_i)`。
- [[here:solutions/t5_flash_attn.py@L83-L97|host 端]]：`sm_scale` 默认 1/√D；`HEAD_DIM` 必须是 2 的幂（`tl.arange` 的要求），参考解只支持 16–128；固定 `num_warps=4, num_stages=2`。

## ⑦ 对照真实实现

- **Triton 官方 tutorial 06**：[[triton:python/tutorials/06-fused-attention.py@L48|_attn_fwd_inner]] 是同一个循环，[[triton:python/tutorials/06-fused-attention.py@L75-L82|m_ij、exp2、alpha]] 一一对应。它用了 **exp2 技巧**：[[triton:python/tutorials/06-fused-attention.py@L219|qk_scale *= 1.44269504]]（1/ln2），之后用 `exp2` 代替 `exp`，硬件的 `ex2` 指令更快。
- **FlashAttention-2（CUDA / CuTe）**：主循环在 [[fa:csrc/flash_attn/src/flash_fwd_kernel.h@L55|compute_attn_1rowblock]]，causal 跳块在 [[fa:csrc/flash_attn/src/flash_fwd_kernel.h@L91-L93|n_block_max]]；online softmax 的重缩放在 [[fa:csrc/flash_attn/src/softmax.h@L137|softmax_rescale_o]]，[[fa:csrc/flash_attn/src/softmax.h@L157-L158|scores_scale 就是这里的 alpha]]。
- **FlashInfer 的 prefill**：[[flashinfer:include/flashinfer/attention/prefill.cuh@L2085|update_mdo_states]]（m、d、o 就是这里的 m、l、acc），kernel 入口 [[flashinfer:include/flashinfer/attention/prefill.cuh@L3088|SinglePrefillWithKVCacheKernel]]。
- 论文：[FlashAttention](https://arxiv.org/abs/2205.14135) · [FlashAttention-2](https://arxiv.org/abs/2307.08691) · [FlashAttention-3](https://arxiv.org/abs/2407.08608)

## ⑧ 常见坑与自测

- **acc 忘了乘 alpha**：只更新了 l，结果偏向前面的块。
- **用 m_i 而不是 m_new 算 p**：max 变大时 exp 可能溢出。
- **causal 的 hi 用了 BLOCK_N**：`(start_m + 1) * BLOCK_N` 在 BLOCK_M ≠ BLOCK_N 时会多算或少算；正确的上界是 query 块的结束位置。
- **K 按 `[BLOCK_N, D]` 读进来再 `tl.trans`**：也对，但多一步；直接用 stride 读成转置更省事。

::: answer 自测 1：S = 1000，BLOCK_M = BLOCK_N = 64，非 causal。每个 program 的循环跑几轮？最后一轮有多少个有效 key？
cdiv(1000, 64) = 16 轮。前 15 轮覆盖 960 个 key，最后一轮 40 个有效，24 个被 `cols < S` mask 成 −inf。
:::

::: answer 自测 2：causal，S = 1024，BLOCK_M = BLOCK_N = 64。一共算多少个 64×64 的分数块？和非 causal 相比省了多少？
query 块共 16 个，第 i 块算 i + 1 个 key 块，总数 1 + 2 + … + 16 = 136。非 causal 是 16 × 16 = 256，省了约 47%。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/t5_flash_attn.py|solutions/t5_flash_attn.py]] · [[here:mine/t5_flash_attn.py|mine/t5_flash_attn.py]] · [[here:tests/test_t5_flash_attn.py|tests/test_t5_flash_attn.py]] · [[here:content/t5.md|content/t5.md]]

**实现**：[[triton:python/tutorials/06-fused-attention.py@L48|Triton tutorial 06]] · [[fa:csrc/flash_attn/src/flash_fwd_kernel.h@L55|FA2 forward]] · [[fa:csrc/flash_attn/src/softmax.h@L137|FA2 online softmax]] · [[fa:hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp|FA3 主循环（Hopper）]] · [[fa:flash_attn/cute/flash_fwd_sm100.py|FA4（CuTe DSL）]] · [[flashinfer:include/flashinfer/attention/prefill.cuh@L2085|FlashInfer update_mdo_states]]

**论文**：[FlashAttention](https://arxiv.org/abs/2205.14135) · [FlashAttention-2](https://arxiv.org/abs/2307.08691) · [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
