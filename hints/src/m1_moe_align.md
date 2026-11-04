title: M1 讲解：MoE 路由与 moe_align_block_size
lede: MoE 层的第一步是决定每个 token 去哪几个专家，第二步是把 (token, slot) 按专家排好、每组补齐到 BLOCK_M 的倍数，好让一个 grouped GEMM 一次算完所有专家。这一页用一个 4 token × top-2 × 4 专家的例子把 route 和 align 的每一个中间张量都写出来，逐行讲清楚 bincount、稳定排序、组内排名和 padding 的作用。

::: callout info 怎么读这一页
对照 [[here:solutions/m1_moe_align.py|m1_moe_align.py]] 读。这题是纯 PyTorch，难点不在 GPU 编程，而在**数据布局**：M2 的 grouped GEMM 完全依赖这里产出的三个张量。
:::

## ① 先看全貌：为什么要排序和补齐

MoE 把 MLP 换成 E 个专家，每个 token 只送到其中 top-k 个。直接写成循环：

```python
for e in range(E):
    idx = tokens_routed_to(e)
    y[idx] += w[idx, e] * (x[idx] @ W_e)       # E 次小 GEMM，E 次 kernel launch
```

E = 64–256 时要启动几百个小 kernel，每个都喂不饱 GPU。高效的做法是**一次 launch** 算完所有专家：把所有 (token, slot) 对按专家排成一列，每个专家的那一段补齐到 BLOCK_M 的整数倍。这样每 BLOCK_M 行组成的块只属于**一个**专家，GEMM kernel 的每个 program 只要知道“我这块是哪个专家”，就能去读对应的权重。

::: flow 从路由到 grouped GEMM
cfg cw=210 nw=165
n lg 0 0 "router logits [T, E]" "" k=acc
n rt 1 0 "route" "topk_ids、topk_weights" k=vio
n al 2 0 "moe_align_block_size" "按专家排序 + 补齐" k=warn w=1.1
n gm 3.1 0 "grouped GEMM（M2）" "一次 launch" k=pass
n out 3.1 1.3 "三个输出" "sorted_ids、expert_ids、总长" k=warn w=1.2
e lg -> rt
e rt -> al
e al -> gm
e al -> out dash
cap: M1 产出 sorted_token_ids（每一行算哪个 (token, slot)）和 expert_ids（每一块用哪个专家），M2 的 kernel 只读这两个张量就知道该干什么。
:::

::: callout analogy 机场托运行李
每位旅客（token）有两件行李（top-2），分别要去两个不同的城市（专家）。机场不会为每件行李单独派一架飞机，而是先把所有行李按目的地分拣到一起（按专家排序），再装进标准集装箱，每个集装箱装 BLOCK_M 件，装不满的用空箱位补齐（padding）。这样每个集装箱只去一个城市，装货机（GEMM 的 program）拿起一个集装箱，看一眼标签（expert_ids）就知道往哪架飞机上送。
:::

## ② route：softmax → top-k → 重新归一化

[[here:solutions/m1_moe_align.py@L18-L22|route（18–22 行）]]

```python
probs = torch.softmax(router_logits.float(), dim=-1)    # 每个 token 在 E 个专家上的概率
w, ids = torch.topk(probs, top_k, dim=-1)               # 取最大的 k 个
w = w / w.sum(-1, keepdim=True)                         # 让这 k 个权重加起来等于 1
```

一个 token、4 个专家，logits = [2.0, 1.0, 0.5, −1.0]，top-2：

| 步骤 | 专家 0 | 专家 1 | 专家 2 | 专家 3 |
|---|---|---|---|---|
| softmax | 0.610 | 0.224 | 0.136 | 0.030 |
| top-2 | **0.610** | **0.224** | | |
| 重新归一化（÷ 0.834） | **0.731** | **0.269** | | |

topk_ids = [0, 1]，topk_weights = [0.731, 0.269]。最后的输出是 `0.731 · expert_0(x) + 0.269 · expert_1(x)`。

docstring 提到不同模型的路由方式不同：Mixtral、Qwen-MoE 是这里的“softmax + top-k + 重新归一化”；DeepSeek-V3 用 sigmoid 分数加分组限制（先选专家组，再在组内选专家）。M2 的 GEMM 部分对这些路由方式都一样，只要拿到 topk_ids 和 topk_weights。

## ③ moe_align_block_size：一个完整的例子

[[here:solutions/m1_moe_align.py@L25-L44|moe_align_block_size（25–44 行）]]

T = 4 个 token，K = 2，E = 4 个专家，block_m = 2：

```text
topk_ids = [[0, 2],      # token 0 去专家 0 和 2
            [2, 1],      # token 1
            [0, 2],      # token 2
            [3, 2]]      # token 3
```

### 第 1 步：展平，给每个 (token, slot) 一个编号

```python
flat = topk_ids.reshape(-1)          # [0, 2, 2, 1, 0, 2, 3, 2]
```

编号 f = t × K + k：f = 5 是 token 2 的第 1 个 slot（去专家 2）。后面所有的“token id”其实都是这个 f。

### 第 2 步：每个专家几个，补齐后多长，从哪开始

```python
counts    = torch.bincount(flat, minlength=E)            # [2, 1, 4, 1]
padded    = (counts + block_m - 1) // block_m * block_m  # [2, 2, 4, 2]
pad_start = torch.cumsum(padded, 0) - padded             # [0, 2, 4, 8]
total     = padded.sum()                                 # 10
```

`bincount` 数每个值出现几次。`cumsum − 自己` 是**排他前缀和**：每个专家的段从哪里开始。和 F1 里从页数算 indptr 是同一个技巧。

### 第 3 步：稳定排序，算组内排名

```python
order = torch.argsort(flat, stable=True)                 # [0, 4, 3, 1, 2, 5, 7, 6]
sorted_experts = flat[order]                             # [0, 0, 1, 2, 2, 2, 2, 3]
group_start = torch.cumsum(counts, 0) - counts           # [0, 2, 3, 7]（不补齐时的起点）
rank = arange(n) - group_start[sorted_experts]           # [0, 1, 0, 0, 1, 2, 3, 0]
dest = pad_start[sorted_experts] + rank                  # [0, 1, 2, 4, 5, 6, 7, 8]
```

- **为什么要稳定排序**：同一个专家内部保持原来的 token 顺序（f = 1, 2, 5, 7，而不是任意顺序）。结果确定、可复现，和 vLLM 的输出也能逐元素对上。
- **rank**：排好之后，每个元素在自己专家的组里排第几。
- **dest**：在**补齐后**的布局里的位置 = 这个专家段的起点 + 组内排名。

### 第 4 步：写出 sorted_token_ids 和 expert_ids

```python
sorted_token_ids = torch.full((total,), n)               # 先全部填 padding 值 n = T*K = 8
sorted_token_ids[dest] = order                           # 再把真实的 f 放到 dest 位置
expert_ids = repeat_interleave(arange(E), padded // block_m)
```

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 150px repeat(10, minmax(44px, 1fr)); min-width: 660px">
<div class="lbl">位置</div><div>0</div><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>7</div><div>8</div><div>9</div>
<div class="lbl">sorted_token_ids</div><div class="c0">0</div><div class="c0">4</div><div class="c1">3</div><div class="cx">8</div><div class="c2">1</div><div class="c2">2</div><div class="c2">5</div><div class="c2">7</div><div class="c3">6</div><div class="cx">8</div>
<div class="lbl">对应 token</div><div class="c0">t0</div><div class="c0">t2</div><div class="c1">t1</div><div class="cx">pad</div><div class="c2">t0</div><div class="c2">t1</div><div class="c2">t2</div><div class="c2">t3</div><div class="c3">t3</div><div class="cx">pad</div>
</div></div>
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 150px repeat(5, minmax(88px, 1fr)); min-width: 660px">
<div class="lbl">块（每块 2 行）</div><div>块 0</div><div>块 1</div><div>块 2</div><div>块 3</div><div>块 4</div>
<div class="lbl">expert_ids</div><div class="c0">专家 0</div><div class="c1">专家 1</div><div class="c2">专家 2</div><div class="c2">专家 2</div><div class="c3">专家 3</div>
</div></div>
<p class="legend">token id 是 f = t·K + k，所以 token = f // K。padding 的值是 T·K = 8，一个越界的编号，GEMM 读到它就知道这一行是空的（mask 掉）。专家 1 和专家 3 各只有 1 个 token，各浪费了 1 行。</p>
:::

### padding 的代价

| | 这个例子 | 真实场景（T = 4096，K = 8，E = 64，block_m = 64） |
|---|---|---|
| 真实行数 T·K | 8 | 32768 |
| 每个专家平均 | 2 | 512 |
| 每个专家最多浪费 | block_m − 1 = 1 | 63 |
| 浪费比例上限 | E·(block_m−1) / (T·K) = 50% | 64 × 63 / 32768 ≈ 12% |

token 越多、block_m 越小，浪费越少；但 block_m 太小 GEMM 的效率又低。decode 时 T 很小，每个专家可能只分到一两个 token，padding 浪费会非常大。这正是 M2 里 decode 场景特别慢的原因。

## ④ 对照 vLLM

vLLM 的 Python 接口：[[vllm:vllm/model_executor/layers/fused_moe/moe_align_block_size.py@L55|moe_align_block_size]]（注意参数顺序是 `topk_ids, block_size, num_experts`，和这题不同）。它的 docstring 里还讲了 expert parallel 的处理：不在本卡上的专家，expert_ids 标成 −1，GEMM 直接跳过这些块。

CUDA 实现在 [[vllm:csrc/moe/moe_align_sum_kernels.cu@L371|moe_align_block_size_kernel]] 和 [[vllm:csrc/moe/moe_align_sum_kernels.cu@L387|count_and_sort_expert_tokens_kernel]]：

| 参考解（PyTorch） | vLLM CUDA |
|---|---|
| `bincount` | 每个线程对自己负责的 token 做 [[vllm:csrc/moe/moe_align_sum_kernels.cu@L184|atomicAdd 到 shared memory 里的计数]] |
| `cumsum`（排他前缀和） | [[vllm:csrc/moe/moe_align_sum_kernels.cu@L192-L205|cub::BlockScan ExclusiveSum]]，顺便得出 num_tokens_post_pad |
| `argsort(stable=True)` + rank | 第二个 kernel 用 [[vllm:csrc/moe/moe_align_sum_kernels.cu@L361-L364|atomicAdd 领取组内位置]]：**不是**稳定排序，组内顺序不确定 |
| `sorted_token_ids` 的长度 | vLLM 预先按上界分配，真实长度放在 num_tokens_post_pad 里（参考解 docstring 第 12–13 行说明了这个区别） |

vLLM 的 CUDA 版组内顺序不确定，但 GEMM 的结果不受影响：每一行算的是同一个 (token, slot) 和同一个专家的乘积，只是写回的顺序不同。测试是按“每个专家的 token 集合”而不是按位置比较的话就不受影响。

## ⑤ 常见坑与自测

- **把 sorted_token_ids 当成 token 编号**：它是 f = t·K + k，要 `// K` 才是 token。M2 里读 x 的时候用 `f // K`，读路由权重的时候用 f 本身。
- **padding 值用 −1 或 0**：0 是一个合法的 f，会被当成真实数据算进去；参考解和 vLLM 都用 T·K。
- **用 Python 循环按专家拼**：E = 256 时每一步都多几百次 kernel launch，比 GEMM 本身还慢。
- **counts 漏了没有 token 的专家**：`bincount` 要带 `minlength=E`，否则最后几个专家的计数缺失，后面的 cumsum 全部错位。

::: answer 自测 1：上面的例子改成 block_m = 4，total、sorted_token_ids 和 expert_ids 分别是什么？
padded = [4, 4, 4, 4]，total = 16，pad_start = [0, 4, 8, 12]。sorted_token_ids = [0, 4, 8, 8 | 3, 8, 8, 8 | 1, 2, 5, 7 | 6, 8, 8, 8]，expert_ids = [0, 1, 2, 3]。16 行里只有 8 行是真实数据，浪费了一半。
:::

::: answer 自测 2：sorted_token_ids 某个位置是 7（K = 2），对应哪个 token 的哪个 slot？它的路由权重在 topk_weights 的哪里？
f = 7 → token 7 // 2 = 3，slot 7 % 2 = 1。权重是 topk_weights[3, 1]，也就是 topk_weights 展平后的第 7 个。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/m1_moe_align.py|solutions/m1_moe_align.py]] · [[here:mine/m1_moe_align.py|mine/m1_moe_align.py]] · [[here:tests/test_m1_moe_align.py|tests/test_m1_moe_align.py]] · [[here:content/m1.md|content/m1.md]]

**vLLM**：[[vllm:vllm/model_executor/layers/fused_moe/moe_align_block_size.py@L55|moe_align_block_size.py]] · [[vllm:csrc/moe/moe_align_sum_kernels.cu@L371|moe_align_sum_kernels.cu]] · [[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py|fused_moe.py]]（M2）

**论文**：[Mixtral of Experts](https://arxiv.org/abs/2401.04088) · [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)（sigmoid 路由与分组限制）
