title: M2 讲解：fused MoE grouped GEMM
lede: 一次 kernel launch 算完所有专家的矩阵乘：这正是 vLLM fused_moe_kernel 的结构。这一页接着 M1 的例子，讲清楚每个 program 怎么从 expert_ids 知道该读哪个专家的权重、A 为什么要 gather（以及 GEMM1 和 GEMM2 的 gather 为什么不一样）、路由权重为什么在 GEMM2 的 epilogue 里乘，最后把整层 SwiGLU MoE 串起来，并和 vLLM 的 kernel 逐行对照。

::: callout info 怎么读这一页
对照 [[here:solutions/m2_fused_moe.py|m2_fused_moe.py]] 读。前置：[M1 讲解](m1_moe_align.html) 的三个输出，[T4 讲解](t4_matmul.html) 的 Triton GEMM（这里的主循环和 T4 几乎一样），[F3 讲解](f3_fused_norm_act.html) 的 silu_and_mul。
:::

## ① 先看全貌：一层 MoE 的五步

[[here:solutions/m2_fused_moe.py@L79-L91|moe_layer（79–91 行）]]

::: flow SwiGLU MoE 层（T 个 token，top-K）
cfg cw=195 nw=160
n r 0 0 "route + align" "M1" k=vio
n g1 1 0 "GEMM1：x @ W13ᵀ" "[T·K, 2I]，gather x" k=acc w=1.05
n s 2.05 0 "silu_and_mul" "[T·K, I]" k=pass
n g2 3.05 0 "GEMM2：h @ W2ᵀ" "[T·K, H]，× 路由权重" k=acc w=1.05
n c 3.05 1.3 "按 K 求和" "view(T, K, H).sum(1)" k=warn
e r -> g1
e g1 -> s
e s -> g2
e g2 -> c
cap: GEMM1 的输入是 T 行的 x，输出是 T·K 行（每个 (token, slot) 一行）；GEMM2 的输入、输出都是 T·K 行。两个 GEMM 用的是同一套 sorted_ids 和 expert_ids，只是读 A 的方式不同。
:::

W13 是每个专家的 gate 和 up 两个投影拼在一起（`[E, 2I, H]`，前 I 行是 gate，后 I 行是 up），和 F3 里 gate_up_proj 的布局一样。

## ② grid：每个 M 块只属于一个专家

[[here:solutions/m2_fused_moe.py@L28-L34|28–34 行]] · host 端 [[here:solutions/m2_fused_moe.py@L71|grid（71 行）]]

```python
grid = (sorted_ids.numel() // block_m, cdiv(N, block_n))

pid_m, pid_n = tl.program_id(0), tl.program_id(1)
offs_m   = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
flat_ids = tl.load(sorted_ids_ptr + offs_m)            # 这一块的 BLOCK_M 个 (token, slot) 编号
row_mask = flat_ids < num_valid                        # padding 行的值是 T·K，被 mask 掉
expert   = tl.load(expert_ids_ptr + pid_m)             # 这一块属于哪个专家
```

M1 的例子（T = 4、K = 2、block_m = 2）里，sorted_ids 长 10，有 5 个 M 块：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 120px repeat(5, minmax(100px, 1fr)); min-width: 660px">
<div class="lbl">pid_m</div><div>0</div><div>1</div><div>2</div><div>3</div><div>4</div>
<div class="lbl">flat_ids</div><div class="c0">0, 4</div><div class="c1">3, <s>8</s></div><div class="c2">1, 2</div><div class="c2">5, 7</div><div class="c3">6, <s>8</s></div>
<div class="lbl">expert</div><div class="c0">0</div><div class="c1">1</div><div class="c2">2</div><div class="c2">2</div><div class="c3">3</div>
<div class="lbl">GEMM1 读 x 的行</div><div class="c0">t0, t2</div><div class="c1">t1, —</div><div class="c2">t0, t1</div><div class="c2">t2, t3</div><div class="c3">t3, —</div>
</div></div>
<p class="legend">每个 program 只读一个专家的权重 W[expert]。划掉的 8 是 padding，row_mask 为 False。GEMM1 读 x 的行号是 flat_id // K。</p>
:::

和 T4 的普通 GEMM 相比，只有两处不同：**A 的行不连续**（要通过 flat_ids 去 gather），**B 的起点由 expert 决定**（`w_ptr + expert * stride_we`）。

::: callout analogy 流水线上的工位卡
GEMM 的每个 program 是流水线上的一个工位。工位开工前先看两张卡：一张写着“今天加工哪几件”（flat_ids，最多 BLOCK_M 件，空位写着“无”），一张写着“用哪套模具”（expert）。拿到模具后，从仓库的不同货架上把这几件原料取过来（gather），加工方法和普通工位完全一样（T4 的主循环）。所有工位同时开工（一次 launch），各用各的模具。
:::

## ③ A 的 gather：GEMM1 和 GEMM2 不一样

[[here:solutions/m2_fused_moe.py@L36-L46|36–46 行]]

```python
if A_ROWS_ARE_TOKENS:          # GEMM1：A 是 x [T, H]
    a_rows = flat_ids // top_k
else:                          # GEMM2：A 是 h [T·K, I]
    a_rows = flat_ids
a_ptrs = a_ptr + a_rows[:, None] * stride_am + offs_k[None, :]
w_ptrs = w_ptr + expert * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None]
```

| | A | A 有几行 | flat_id f 对应 A 的哪一行 |
|---|---|---|---|
| GEMM1 | x | T（每个 token 一行） | f // K：同一个 token 的 K 个 slot 读**同一行** x |
| GEMM2 | h | T·K（每个 (token, slot) 一行） | f：每个 slot 有自己的中间结果 |

`A_ROWS_ARE_TOKENS` 是 constexpr，所以两种 gather 是同一个 kernel 编译出来的两个版本。

**W 的转置**：W[e] 存成 `[N, K]` 的行优先（PyTorch Linear 的权重格式），GEMM 需要的是 Wᵀ。和 T4、T5 一样，用 stride 直接读成 `[BLOCK_K, BLOCK_N]` 的转置块：`offs_n[None, :] * stride_wn + offs_k[:, None]`。

**为什么 gather 比连续读慢**：同一个 M 块里的行在 x 里是分散的（t0、t2），每一行是连续的 H 个数，所以每行内部仍然是合并访存，但不同行之间无法连成更大的连续块，硬件预取和 TMA 之类的块拷贝也用不上。题目页提到这是和稠密 GEMM 差距的原因之一。

## ④ 主循环和 epilogue

[[here:solutions/m2_fused_moe.py@L48-L61|48–61 行]]

主循环和 T4 一样：沿 K 方向每次取 BLOCK_K，`tl.dot` 累加到 fp32 的 acc。mask 多了一项 `row_mask[:, None]`，padding 行读 0。

```python
if MUL_WEIGHT:                                  # 只在 GEMM2 打开
    tw = tl.load(topk_w_ptr + flat_ids, mask=row_mask, other=0.0)
    acc = acc * tw[:, None]
out_ptrs = out_ptr + flat_ids[:, None] * stride_om + offs_n[None, :]
tl.store(out_ptrs, acc.to(...), mask=row_mask[:, None] & (offs_n[None, :] < N))
```

- **输出写回到 flat_id 对应的行**（scatter）：out 是 `[T·K, N]`，按 (token, slot) 的原始顺序排列，不是排序后的顺序。所以最后可以直接 `view(T, K, H).sum(1)` 把同一个 token 的 K 个结果加起来。
- **路由权重为什么在 GEMM2 里乘**：最终输出是 `Σ_k w_k · expert_k(x)`，而 expert(x) = W2 · silu_and_mul(W13 · x)。权重是标量，可以挪到最后一步乘。放在 GEMM2 的 epilogue 里，几乎不花额外时间，也省掉一次单独的逐元素 kernel。如果放在 GEMM1 里，silu 是非线性的，结果就不对了。
- **out 初始化为 0**（[[here:solutions/m2_fused_moe.py@L69|69 行]]）：所有合法行都会被写到，不清零也可以；清零是为了让未被覆盖的情况更容易被发现。

## ⑤ 性能：为什么只有稠密 GEMM 的一半

题目页的实测：T = 4096、E = 64、top-8、H = 2048、I = 768（接近 Qwen3-30B-A3B），bf16。

| | FLOPs | 时间 | TFLOP/s |
|---|---|---|---|
| GEMM1：2 · T·K · H · 2I | 2 × 32768 × 2048 × 1536 = 206 G | | |
| GEMM2：2 · T·K · I · H | 2 × 32768 × 768 × 2048 = 103 G | | |
| 参考解整层 | 309 G | 3245 us | 95 |
| 同 FLOPs 的两个稠密 cuBLAS GEMM | 309 G | 1461 us | 212 |

差距的来源（题目页列出的几条）：

- **固定的 tile 配置**：64×64×32 对这个形状不是最优。vLLM 为每种 (E, N, dtype, GPU) 准备了调好的 JSON 配置，仓库里有三百多个 [[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py@L1130|get_config_file_name]] 指向的配置文件，运行时由 [[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py@L1503|try_get_optimal_moe_config]] 选择。
- **padding**：每个专家最多浪费 BLOCK_M − 1 行（M1 第 ③ 节）。
- **gather**：A 的行不连续。
- **silu_and_mul 和最后的求和是单独的 PyTorch 操作**：多了中间结果的写回和读取。

::: flow decode 时为什么更糟
cfg cw=230 nw=175
n p 0 0 "prefill：T = 4096" "每个专家约 512 行" k=pass
n p2 1 0 "compute-bound" "调 tile、量化" k=pass
n d 0 1.3 "decode：T = 16" "每个专家 0–2 行" k=todo
n d2 1 1.3 "memory-bound" "读一遍被激活专家的权重" k=todo w=1.1
n d3 2.1 1.3 "BLOCK_M = 64" "每块 62 行以上是 padding" k=todo
e p -> p2
e d -> d2
e d2 -> d3
cap: decode 时 16 × 8 = 128 个 (token, slot) 分散到 64 个专家上，每个专家只有一两行，Tensor Core 的大部分算力花在 padding 上；而这一步又要把几乎所有专家的权重都读一遍。vLLM 对小 batch 会选更小的 BLOCK_SIZE_M。
:::

::: callout analogy 包车
prefill 像一个旅行团 4096 人去 64 个景点，每个景点平均 500 多人，每辆大巴（BLOCK_M = 64 座）都能坐满。decode 像只有 16 个散客，每人要去 8 个景点，平摊到每个景点只有一两个人，却还是给每个景点派一辆 64 座的大巴：车费（读权重）一分不少，座位几乎全空着。
:::

## ⑥ 对照 vLLM 的 fused_moe_kernel

参考解的结构就是从 vLLM 抄来的：[[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py@L298|fused_moe_kernel]]

| 参考解 | vLLM |
|---|---|
| `flat_ids = tl.load(sorted_ids_ptr + offs_m)` | [[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py@L409-L410|offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)]] |
| `row_mask = flat_ids < num_valid` | [[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py@L421|token_mask = offs_token < num_valid_tokens]] |
| `expert = tl.load(expert_ids_ptr + pid_m)` | [[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py@L423-L424|off_experts = …；if off_experts == -1]]（专家并行时不在本卡的专家直接写 0 跳过） |
| `a_rows = flat_ids // top_k` | [[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py@L473-L474|offs_token // top_k * stride_am]] |
| `acc *= tw[:, None]` | [[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py@L605|moe_weight = tl.load(topk_weights_ptr + …)]]（`MUL_ROUTED_WEIGHT`） |

vLLM 的版本还处理了 FP8 / INT8 的 per-tensor 和 block-wise scale（M3 会讲 block scaling）、专家并行，以及按 GROUP_SIZE_M 分组的 program 顺序（T4 讲过的 L2 优化）。

## ⑦ 常见坑与自测

- **GEMM1 用了 `a_rows = flat_ids`**：读 x 越界（x 只有 T 行）。
- **GEMM2 用了 `flat_ids // top_k`**：读错了中间结果，每个 slot 都用了 slot 0 的 h。
- **路由权重乘了两次**（GEMM1 和 GEMM2 各一次），或者乘在 silu 之前。
- **out 按排序后的顺序写**：最后的 `view(T, K, H).sum(1)` 就对不上 token 了。
- **expert 没转 int64**：`expert * stride_we` 在大模型上溢出（E = 256、每个专家几千万个参数）。

::: answer 自测 1：M1 的例子里 pid_m = 3 这个 program（GEMM1），读 x 的哪两行？写 out 的哪两行？用哪个专家的权重？
flat_ids = [5, 7]，读 x 的 5 // 2 = 2 和 7 // 2 = 3 两行（t2、t3），写 out 的第 5、7 行，用专家 2 的 W13。
:::

::: answer 自测 2：top-K 从 8 改成 2，其余不变，GEMM1 和 GEMM2 的 FLOPs 怎么变？
都和 T·K 成正比，变成原来的 1/4：GEMM1 51.5 G，GEMM2 25.8 G。但读权重的量不一定变少：只要 token 足够多，每个专家都会被激活，每个专家的权重都要读一遍。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/m2_fused_moe.py|solutions/m2_fused_moe.py]] · [[here:mine/m2_fused_moe.py|mine/m2_fused_moe.py]] · [[here:tests/test_m2_fused_moe.py|tests/test_m2_fused_moe.py]] · [[here:content/m2.md|content/m2.md]] · [[here:drill.py|drill.py（load() 选择你的 M1 还是参考解）]]

**vLLM**：[[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py@L298|fused_moe_kernel]] · [[vllm:vllm/model_executor/layers/fused_moe/fused_moe.py@L1503|try_get_optimal_moe_config]] · [[vllm:vllm/model_executor/layers/fused_moe/moe_align_block_size.py@L55|moe_align_block_size]]

**Triton**：[[triton:python/tutorials/08-grouped-gemm.py|tutorial 08 grouped GEMM]]（另一种做法：每个 group 一个独立的 GEMM 描述）

**论文**：[Mixtral](https://arxiv.org/abs/2401.04088) · [MegaBlocks](https://arxiv.org/abs/2211.15841)（block-sparse 的 MoE 计算）
