title: L2 讲解：paged decode attention 与 GQA
lede: decode 时每个序列只有一个新 query，attention 变成了“读一遍 KV cache”。这一页先算清楚 decode 为什么是 memory-bound 的，再逐段读参考解：为什么一个 program 负责一个 KV head、GQA 的 query 怎么补到 16 行喂给 tl.dot、page 循环里的间接寻址，以及 B = 1 时为什么只剩 4% 的带宽。

::: callout info 怎么读这一页
对照 [[here:solutions/l2_paged_decode.py|l2_paged_decode.py]] 读。前置：[L1 讲解](l1_paged_kv_write.html) 的 cache 布局，[T5 讲解](t5_flash_attn.html) 的 online softmax。这一题的循环体和 T5 几乎一样，新东西在循环外面。
:::

## ① 先看全貌：decode 就是读 KV

| | prefill（T5） | decode（L2） |
|---|---|---|
| 每个序列的 query 数 | S 个（整段 prompt） | **1 个**（刚生成的 token） |
| `q·kᵀ` 是什么 | 矩阵 × 矩阵 | 向量 × 矩阵 |
| 每个 KV token 的计算 | 和 S 个 query 各算一次 | 只和 1 个 query 算一次：约 4·D FLOPs |
| 每个 KV token 的读取 | 读一次，被很多 query 复用 | 读 2·D 个数（K 和 V），只用一次 |
| 瓶颈 | 算力 | **显存带宽** |

decode 的算术强度约为 1 FLOP/字节，远低于 GPU 的拐点。所以 **decode attention 的时间 ≈ KV cache 的字节数 ÷ 带宽**，所有优化都围绕“少读、读快”。

题目页的测试：B = 64、ctx = 1024、Hkv = 8、D = 128、fp16。KV 字节数 = 2 × 64 × 1024 × 8 × 128 × 2 = 268 MB，实测 176 us，即 1520 GB/s（峰值的 85%）。

## ② GQA：几个 query head 共享一个 KV head

LLaMA-3、Qwen 等模型用 **GQA**：Hq 个 query head 分成 Hkv 组，每组共享一个 KV head。题目的配置是 Hq = 32、Hkv = 8，GROUP = 4。

::: flow 两种 program 划分：KV 被读几次
cfg cw=255 nw=175
n kv 0 0.65 "KV head 3" "ctx × D 的 K 和 V" k=warn
n a 1 0 "grid = (B, Hq)" "query head 12–15 各一个 program" k=todo w=1.25
n a2 2 0 "KV head 3 被读 4 次" "" k=todo
n b 1 1.3 "grid = (B, Hkv)" "一个 program 管 4 个 query head" k=pass w=1.25
n b2 2 1.3 "KV head 3 只读 1 次" "" k=pass
e kv -> a
e kv -> b
e a -> a2
e b -> b2
cap: query head h 用的是 KV head h // GROUP。按 query head 开 program 时，同一组的 4 个 program 各自把同一份 KV 读一遍；按 KV head 开 program，一份 KV 读进来，给 4 个 query 一起用。decode 是 memory-bound 的，读取量就是时间，这是 GQA 在 kernel 层面的主要收益。
:::

::: callout analogy 合租看报纸
4 个室友（query head）都想看同一份报纸（KV head）。如果每个人各自去报亭买一份（按 query head 开 program），报亭（显存）要卖 4 份、跑 4 趟。合租的做法是买一份放在客厅桌上（一个 program 读一次），4 个人围着一起看。报纸越厚（ctx 越长），省得越多。
:::

## ③ 循环外：program、q 的打包

[[here:solutions/l2_paged_decode.py@L30-L45|30–45 行]] · host 端 [[here:solutions/l2_paged_decode.py@L85-L96|85–96 行]]

```python
b   = tl.program_id(0).to(tl.int64)      # 第几个序列
kvh = tl.program_id(1).to(tl.int64)      # 第几个 KV head
ctx = tl.load(context_lens + b)          # 这个序列有多少个 KV token

offs_g  = tl.arange(0, GROUP_PAD)        # 组内的 query head（补齐到 16）
g_mask  = offs_g < GROUP
q_heads = kvh * GROUP + offs_g
q = tl.load(Q + b * stride_qb + q_heads[:, None] * stride_qh + offs_d[None, :],
            mask=g_mask[:, None], other=0.0)          # [GROUP_PAD, D]
```

- grid 是 `(B, Hkv)`，一个 program 负责一个序列的一个 KV head。
- **为什么补齐到 16 行**：`tl.dot` 用 Tensor Core，要求矩阵的行数至少 16。GROUP = 4 时只有 4 行，所以 `GROUP_PAD = max(16, next_power_of_2(GROUP)) = 16`，多出的 12 行填 0。

KV head 3 对应的 q 块：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 140px repeat(16, minmax(26px, 1fr)); min-width: 620px">
<div class="lbl">offs_g</div><div>0</div><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>7</div><div>8</div><div>9</div><div>10</div><div>11</div><div>12</div><div>13</div><div>14</div><div>15</div>
<div class="lbl">query head</div><div class="c0">12</div><div class="c0">13</div><div class="c0">14</div><div class="c0">15</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div><div class="cx">·</div>
</div></div>
<p class="legend">q_heads = 3 × 4 + offs_g。前 4 行是真实的 query head 12–15，后 12 行 mask 掉、值为 0，最后也不写回。Tensor Core 有 75% 的算力花在了 0 上，但 decode 本来就不是算力瓶颈，这点浪费无所谓；关键是 KV 只读了一次。</p>
:::

- `m_i`、`l_i`、`acc` 和 T5 一样，每一行一个状态，只是“行”从 query 位置变成了 query head。

## ④ page 循环：间接寻址

[[here:solutions/l2_paged_decode.py@L47-L59|47–59 行]]

```python
num_pages = tl.cdiv(ctx, PAGE)
for p in range(0, num_pages):
    phys  = tl.load(block_tables + b * stride_btb + p).to(tl.int64)    # 逻辑页 p → 物理块
    tok   = p * PAGE + offs_t
    t_mask = tok < ctx                                                 # 最后一页往往不满
    kv_off = phys * stride_cb + offs_t[:, None] * stride_ct + kvh * stride_ch + offs_d[None, :]
    k = tl.load(K_cache + kv_off, mask=t_mask[:, None], other=0.0)     # [PAGE, D]
    v = tl.load(V_cache + kv_off, mask=t_mask[:, None], other=0.0)
    s = tl.dot(q, tl.trans(k)) * scale                                 # [GROUP_PAD, PAGE]
    s = tl.where(t_mask[None, :], s, float("-inf"))
```

这里的 BLOCK_N 就是一页（PAGE = 16 个 token）。和 T5 唯一的区别是：**下一块 K、V 在哪里，要先查 block table 才知道**。

::: flow 第 p 页的地址怎么算
cfg cw=220 nw=170
n bt 0 0 "block_tables[b][p]" "物理块号 phys" k=warn w=1.05
n off 1 0 "phys · stride_cb" "跳到这个物理块" k=acc
n t 2 0 "+ offs_t · stride_ct" "块内第几个 token" k=acc
n h 2 1.3 "+ kvh · stride_ch" "第几个 KV head" k=acc
n d 1 1.3 "+ offs_d" "head_dim（stride 1）" k=acc
n kv 0 1.3 "[PAGE, D] 的 K 块" "一次 load" k=pass
e bt -> off
e off -> t
e t -> h
e h -> d
e d -> kv
cap: cache 的形状是 [num_blocks, PAGE, Hkv, D]，四个 stride 分别对应这四维。同一个 KV head 的一页数据在内存里不是完全连续的（中间隔着其他 head），但每个 token 的 D 个数是连续的。
:::

**手算一个例子**：ctx = 37，PAGE = 16，block_tables[b] = [9, 3, 7, …]。

| p | phys | tok | 有效 token 数 |
|---|---|---|---|
| 0 | 9 | 0–15 | 16 |
| 1 | 3 | 16–31 | 16 |
| 2 | 7 | 32–47 | 37 − 32 = **5**，后 11 个 mask 成 −inf |

num_pages = cdiv(37, 16) = 3。block_tables 第 3 项之后的内容不会被读到。

**`tl.trans(k)`**：k 读进来是 `[PAGE, D]`，转置成 `[D, PAGE]` 再和 `[GROUP_PAD, D]` 的 q 相乘。T5 是用 stride 直接读成转置的，这里先读再转置，两种写法都可以。

## ⑤ online softmax：和 T5 一样

[[here:solutions/l2_paged_decode.py@L63-L73|63–73 行]]：`m_new`、`p_`、`alpha`、`l_i`、`acc` 的更新和 T5 逐字相同。第 0 页一定包含 token 0（context_lens ≥ 1），所以第一轮之后 m_new 就是有限值，不会出现 −inf − (−inf)。

循环结束 `out = acc / l_i`，只写回 `g_mask` 为 True 的那 GROUP 行。

## ⑥ 为什么 B = 1 时只有 73 GB/s

题目页的实测：

| B × ctx | program 数 = B × Hkv | 有效带宽 |
|---|---|---|
| 64 × 1024 | 512 | 1520 GB/s |
| 8 × 4096 | 64 | 514 GB/s |
| 1 × 32768 | **8** | **73 GB/s** |

RTX 5090 有 170 个 SM。B = 1 时只有 8 个 program，162 个 SM 闲着；而这 8 个 program 每个都要串行走 32768 / 16 = 2048 页。总的读取量没变，但只用了不到 5% 的硬件。

::: flow 并行度不够
cfg cw=230 nw=175
n one 0 0 "B = 1，Hkv = 8" "8 个 program" k=todo
n sm 1 0 "170 个 SM" "8 个在忙" k=todo
n fix 0 1.3 "L3：沿 ctx 切成 64 片" "8 × 64 = 512 个 program" k=pass w=1.1
n sm2 1 1.3 "170 个 SM" "全部在忙" k=pass
e one -> sm
e fix -> sm2
cap: 解决办法是把一个序列的 KV 切成多段，每段一个 program，最后再合并。合并需要每段的 LSE，这就是 L3 的 split-KV（Flash-Decoding）。
:::

::: callout analogy 一个人读一本厚书
给 8 个人每人一本 2000 页的书，让他们各自从头读到尾（每个 program 串行走 2048 页），图书馆里另外 162 个座位空着。更快的做法是把每本书拆成 64 份，分给 64 个人同时读，最后把各自的读书笔记合并起来。合并笔记需要一点额外的规则（LSE），这就是 L3。
:::

## ⑦ 对照真实实现

- **vLLM 的 Triton decode attention**（两阶段，第一阶段就是这里的结构再加上分片）：[[vllm:vllm/v1/attention/ops/triton_decode_attention.py@L68|_fwd_kernel_stage1]] 里 [[vllm:vllm/v1/attention/ops/triton_decode_attention.py@L104|cur_kv_head = cur_head // kv_group_num]] 是 query head 到 KV head 的映射，[[vllm:vllm/v1/attention/ops/triton_decode_attention.py@L129-L138|kv_page_number]] 是 page 的间接寻址；GQA 打包版是 [[vllm:vllm/v1/attention/ops/triton_decode_attention.py@L278|_fwd_grouped_kernel_stage1]]（BLOCK_H 就是这里的 GROUP_PAD）。
- **vLLM 的 PagedAttention 设计文档**：[[vllm:docs/design/paged_attention.md|paged_attention.md]]，讲的是最早的 CUDA 版本（每个 warp 负责一部分 token），配了很多图。
- **FlashInfer**：[[flashinfer:include/flashinfer/attention/decode.cuh@L624|BatchDecodeWithPagedKVCacheKernel]]；Python 端是 [[flashinfer:flashinfer/decode.py@L1354|BatchDecodeWithPagedKVCacheWrapper]]，它的 `plan()` 会根据 batch 和长度决定要不要分片（F1 会用到）。

## ⑧ 常见坑与自测

- **q_heads 算错**：用了 `kvh + offs_g * Hkv`（交错分组），而模型实际是连续分组（head h 对应 KV head h // GROUP）。两种约定都存在，要和模型一致。
- **最后一页没 mask**：读到上一个序列留下的旧 KV，结果悄悄出错。
- **phys 没转 int64**：`phys * stride_cb` 在大 cache 上溢出。
- **补齐行没 mask 写回**：把 padding 行的 0/0 = NaN 写进了别的 head 的输出。

::: answer 自测 1：Hq = 64，Hkv = 8，GROUP_PAD 是多少？Tensor Core 的利用率呢？
GROUP = 8，GROUP_PAD = max(16, 8) = 16，16 行里 8 行有效，利用率 50%（题目页面试追问里提到的数字）。
:::

::: answer 自测 2：B = 8，ctx = 4096，Hkv = 8，D = 128，fp16。理论上最少要读多少字节？如果带宽是 1.5 TB/s，最快要多久？
2（K 和 V）× 8 × 4096 × 8 × 128 × 2 字节 = 134 MB，÷ 1.5 TB/s ≈ 89 us。题目页实测 261 us，因为只有 64 个 program，填不满 170 个 SM。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/l2_paged_decode.py|solutions/l2_paged_decode.py]] · [[here:mine/l2_paged_decode.py|mine/l2_paged_decode.py]] · [[here:tests/test_l2_paged_decode.py|tests/test_l2_paged_decode.py]] · [[here:content/l2.md|content/l2.md]]

**vLLM**：[[vllm:vllm/v1/attention/ops/triton_decode_attention.py@L68|triton_decode_attention.py]] · [[vllm:docs/design/paged_attention.md|PagedAttention 设计文档]]

**FlashInfer**：[[flashinfer:include/flashinfer/attention/decode.cuh@L624|BatchDecodeWithPagedKVCacheKernel]] · [[flashinfer:flashinfer/decode.py@L1354|BatchDecodeWithPagedKVCacheWrapper]]

**论文**：[GQA](https://arxiv.org/abs/2305.13245) · [PagedAttention](https://arxiv.org/abs/2309.06180)
