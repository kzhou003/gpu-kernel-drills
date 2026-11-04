title: FlashInfer 原理：为 LLM serving 设计的 attention 引擎
lede: FlashInfer 基于什么观察、想解决什么问题、整体思路是什么，以及每一个设计具体怎么做：统一的 block-sparse KV 存储、可组合格式、稀疏加载到稠密计算、多种 tile 尺寸与 GQA 融合、JIT 自定义 attention 变体、负载均衡调度与 plan / run。每一节都配图、类比和钉死版本的源码链接，内容以 FlashInfer 论文（MLSys 2025）和仓库源码为准。

::: callout info 怎么读这一页
如果你做过这套题，会发现 FlashInfer 的很多零件你已经亲手写过：T5 的 FlashAttention、L1–L3 的分页 KV 和 split-KV、F1 的 CSR 和 plan / run、F2 的 LSE 合并。这一页把它们串成一个完整的系统设计。第 ⑪ 节有一张“题目 ↔ FlashInfer 设计”的对照表。论文原文：[FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)。
:::

## ① 一句话和全景图

**FlashInfer 是一个“attention 代码生成器 + 运行时调度器”**：编译期根据 attention 变体、任务类型和 KV 存储格式，用 JIT 生成专门的 kernel；运行期根据这一步每个请求的长度，在 CPU 上算出负载均衡的调度计划，再让 GPU 按计划执行。

::: flow FlashInfer 的整体结构（对应论文图 1）
cfg cw=225 nw=175
n spec 0 0 "attention 变体" "softcap、sink、RoPE 融合…" k=vio
n task 0 1.2 "任务信息" "prefill / decode / append" k=vio
n kv 0 2.4 "KV 存储格式" "page 大小、稀疏模式" k=vio
n jit 1.2 1.2 "JIT 编译" "填进 CUDA / CUTLASS 模板" k=acc w=1.05
n ker 2.4 1.2 "专用 kernel" "缓存到磁盘" k=acc
n len 1.2 2.9 "这一步的序列长度" "每个 decode step 都变" k=warn w=1.05
n plan 2.4 2.9 "plan（CPU）" "负载均衡调度" k=warn
n run 3.6 2.05 "run（GPU）" "所有层复用同一个 plan" k=pass
g 0 0 2.4 1.2 "编译期：只做一次" k=vio
g 1.2 2.9 2.4 2.9 "运行期：每一步" k=warn
e spec -> jit
e task -> jit
e kv -> jit
e jit -> ker
e ker -> run
e len -> plan
e plan -> run
cap: 把“会变的东西”分成两类：变体、任务、格式这类“换个模型才变”的，交给编译期 JIT；序列长度这类“每一步都变”的，交给运行期的 plan。两者分开，kernel 才能既专用又不用每一步重新编译。
:::

## ② 背景：serving 里的 attention 和训练时不一样

FlashAttention（T5 写过）解决了“不把 S×S 矩阵写进显存”的问题，它是为**训练**设计的：一个 batch 里的序列长度通常一样，Q、K、V 都是连续的稠密张量，attention 只有一种标准形式。

LLM **serving** 的情况完全不同：

| | 训练 / 离线 prefill | 在线 serving |
|---|---|---|
| 一个 batch 里的长度 | 通常相同或 padding 到相同 | 每个请求都不同，从 1 到 10 万 |
| 每个请求的 query 数 | 整个序列 | prefill 是一段，decode 只有 **1** 个，speculative 是几个 |
| KV 存在哪 | 连续张量 | 分页的 KV cache（L1）、radix tree、共享前缀 |
| attention 形式 | 标准 softmax | GQA、MLA、滑动窗口、logits soft-cap、attention sink…… |
| 一步之间什么在变 | 几乎不变 | 每一步每个请求都变长一点，请求随时进出 |
| 启动方式 | 普通 launch | 往往要被 **CUDA graph** 捕获（要求每次参数不变） |

论文的出发点就是：**每个 serving 系统都在为这些情况的某个子集手写 attention kernel，维护成本高，而且常常不是最优的**。FlashInfer 想用一个统一的引擎覆盖它们。

## ③ 四个观察，对应四组设计

::: flow 观察 → 设计
cfg cw=290 nw=210
n o1 0 0 "观察 1：KV 存储格式五花八门" "page table、radix tree、树形、稀疏" k=todo w=1.1
n o2 0 1.2 "观察 2：attention 变体越来越多" "每个变体都想要专用 kernel" k=todo w=1.1
n o3 0 2.4 "观察 3：输入长度动态且不均匀" "还要兼容 CUDA graph" k=todo w=1.1
n o4 0 3.6 "观察 4：算术强度差别很大" "decode 只有 1 个 query" k=todo w=1.1
n d1 1 0 "统一的 block-sparse 格式" "+ 可组合格式（第 ⑤ 节）" k=pass w=1.1
n d2 1 1.2 "可定制模板 + JIT" "（第 ⑧ 节）" k=pass w=1.1
n d3 1 2.4 "负载均衡调度 + plan / run" "（第 ⑨ 节）" k=pass w=1.1
n d4 1 3.6 "多种 tile 尺寸 + GQA 融合" "（第 ⑦ 节）" k=pass w=1.1
e o1 -> d1
e o2 -> d2
e o3 -> d3
e o4 -> d4
cap: 还有一个贯穿所有设计的数学基础：attention 的部分结果可以精确合并（第 ④ 节）。没有它，分块、拆分 KV、共享前缀都做不了。
:::

**观察 4 的量化**：FlashAttention 的算术强度大约是 O(1 / (1/l_qo + 1/l_kv))。serving 里 l_qo ≤ l_kv，所以强度 ≈ O(l_qo)：prefill 时 l_qo 是几千，是 compute-bound 的；decode 时 l_qo = 1，是 memory-bound 的（L2 讲过）。GQA 把强度提高到 O(g · l_qo)，g 是每组的 query head 数。**同一套 kernel 配置不可能同时适合这两端。**

::: callout analogy 一家要同时接很多种订单的工厂
传统的 attention 库像一条只生产一种标准件的流水线（训练用的 FlashAttention）。serving 的订单却是：货物存放在不同形状的仓库里（KV 格式），每个客户都有自己的定制要求（attention 变体），订单大小每分钟都在变（动态长度），有的订单只要一件、有的要一万件（算术强度）。FlashInfer 的做法是：把所有仓库统一编号（block-sparse），把定制要求写成配方让机器现场生产（JIT），每天开工前排一次班让每台机器的活一样多（plan），并准备几种不同尺寸的模具按订单大小选用（tile 尺寸）。
:::

## ④ 基石：attention state 和合并运算 ⊕

对一个 query、一部分 key（下标集合 I），定义它的 **attention state** = (O(I), LSE(I))：O 是只看这部分 key 时的输出，LSE 是这部分分数的 log-sum-exp。两个不相交部分的 state 可以精确合并：

```text
O(I ∪ J)   = ( e^{LSE(I)}·O(I) + e^{LSE(J)}·O(J) ) / ( e^{LSE(I)} + e^{LSE(J)} )
LSE(I ∪ J) = log( e^{LSE(I)} + e^{LSE(J)} )
```

这个 ⊕ 满足**交换律和结合律**，所以任意多块可以按任意顺序、任意分组合并。论文的原话是：在 FlashInfer 里，attention state 是 attention 运算的标准输出，⊕ 是标准的归约运算，**地位相当于 GEMM 里的加法**。

::: flow ⊕ 让哪些事变成可能
cfg cw=230 nw=175
n op 0 1 "attention state" "(O, LSE) 和 ⊕" k=warn
n a 1 0 "split-KV" "长序列拆给多个 CTA（L3）" k=pass
n b 1 1 "共享前缀 / cascade" "前缀和后缀分开算（F2）" k=pass
n c 1 2 "负载均衡调度" "按工作量切块再合并（⑨）" k=pass
e op -> a
e op -> b
e op -> c
cap: 你在 L3 写过 merge kernel，在 F2 写过 merge_state：它们就是 ⊕。FlashInfer 内部用以 2 为底的 LSE（F2 讲过），数学上等价。
:::

::: callout analogy 分店报账
连锁店的每家分店报两个数：“平均客单价”（O）和“客流量的对数”（LSE）。总部合并时按客流量加权平均客单价，客流量直接相加。分店之间怎么分组、先合并哪两家，结果都一样。有了这条规则，总部就可以把一家超大的分店拆成几个“虚拟分店”分别统计（split-KV），也可以把几家共用的仓库只盘点一次（共享前缀）。
:::

源码：[[flashinfer:include/flashinfer/attention/state.cuh@L29|state_t]]，合并在 [[flashinfer:include/flashinfer/attention/state.cuh@L53|state_t::merge]]；独立的合并 kernel 在 [[flashinfer:include/flashinfer/attention/cascade.cuh@L45|MergeStateKernel]]。

## ⑤ 设计 1：统一的 block-sparse KV 存储

### 观察：所有的 KV 存储格式其实都是一个稀疏矩阵

把 attention 看成一个矩阵：**行是 query，列是 KV cache 里的所有位置**。第 i 行第 j 列“有值”，表示 query i 要看 KV 位置 j。

- 分页 KV（vLLM 的 PagedAttention）：每个请求只看自己的那几页。
- radix tree（SGLang）：前缀共享的请求看同一批节点。
- 树形 attention（speculative decoding）、按重要性挑选的 KV（Quest 之类）：每个 query 看的位置更不规则。

这些都可以用同一种格式描述：**BSR（Block Sparse Row，块稀疏行）**。矩阵被切成 B_r × B_c 的块，只存非零块，用 `indptr`（每行块从哪开始）和 `indices`（每个非零块是第几列块）描述。

三个 decode 请求（每个 1 个 query），page 大小为 B_c，物理页共 8 个：A 用页 5、1、6，B 用页 2，C 用页 7、3。

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 90px repeat(8, minmax(46px, 1fr)); min-width: 520px">
<div class="lbl"></div><div class="lbl" style="text-align:center">页 0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div><div class="lbl" style="text-align:center">3</div><div class="lbl" style="text-align:center">4</div><div class="lbl" style="text-align:center">5</div><div class="lbl" style="text-align:center">6</div><div class="lbl" style="text-align:center">7</div>
<div class="lbl">请求 A</div><div></div><div class="c0">②</div><div></div><div></div><div></div><div class="c0">①</div><div class="c0">③</div><div></div>
<div class="lbl">请求 B</div><div></div><div></div><div class="c1">①</div><div></div><div></div><div></div><div></div><div></div>
<div class="lbl">请求 C</div><div></div><div></div><div></div><div class="c2">②</div><div></div><div></div><div></div><div class="c2">①</div>
</div></div>
<p class="legend">每个彩色格子是一个非零块（B_r = 1 个 query × B_c = 一页的 token）。圈里的数字是这一页在请求自己序列里的顺序。用 BSR 描述：indptr = [0, 3, 4, 6]，indices = [5, 1, 6, 2, 7, 3]，这正是 F1 里的 CSR 页表。</p>
:::

论文的定义：query 和输出用不带 padding 的 ragged tensor 存储；KV cache 用 BSR，**B_r 对应 query 的 tile 大小，B_c 由 KV cache 管理算法决定**（page 大小，可以小到 1，即 vector sparse）。kernel 支持任意的 (B_r, B_c)。

源码：分页格式的结构体 [[flashinfer:include/flashinfer/page.cuh@L38|paged_kv_t]]；通用的块稀疏接口 [[flashinfer:flashinfer/sparse.py@L363|BlockSparseAttentionWrapper]]（它的 [[flashinfer:flashinfer/sparse.py@L503|plan]] 直接接收 BSR 的 indptr 和 indices）、行块大小可变的 [[flashinfer:flashinfer/sparse.py@L1657|VariableBlockSparseAttentionWrapper]]。

::: callout analogy 图书馆的借阅登记表
不管书是按楼层放、按书架放、还是散放在各个阅览室（不同的 KV 存储方式），图书馆都可以用同一张登记表描述“谁要看哪几本书”：每个读者一行，只登记他要看的那几格（非零块）。管理员（kernel）只认这张表，不关心书原来是怎么放的。
:::

### 可组合格式：共享前缀用大块

单一格式有个两难：B_r 越大，同一个块里的 query 越多，它们读进 shared memory 的 KV 可以共享；但 B_r 太大又会浪费（块里有很多空行）。

**可组合格式**（composable formats）的思路：同一个稀疏矩阵，拆成几个子矩阵，**每个用最合适的块大小**。最典型的是共享前缀：4 个请求共用同一段系统 prompt（页 0、1），后面各自有不同的后缀。

::: raw
<div style="display:flex; gap:28px; flex-wrap:wrap; justify-content:center">
<div><p class="legend"><b>单一格式（B_r = 1）</b>：每个请求各读一遍前缀</p>
<div class="cells" style="grid-template-columns: 56px repeat(6, 40px)">
<div class="lbl"></div><div class="lbl" style="text-align:center">0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div><div class="lbl" style="text-align:center">3</div><div class="lbl" style="text-align:center">4</div><div class="lbl" style="text-align:center">5</div>
<div class="lbl">req 0</div><div class="c1">读</div><div class="c1">读</div><div class="c0">·</div><div></div><div></div><div></div>
<div class="lbl">req 1</div><div class="c1">读</div><div class="c1">读</div><div></div><div class="c0">·</div><div></div><div></div>
<div class="lbl">req 2</div><div class="c1">读</div><div class="c1">读</div><div></div><div></div><div class="c0">·</div><div></div>
<div class="lbl">req 3</div><div class="c1">读</div><div class="c1">读</div><div></div><div></div><div></div><div class="c0">·</div>
</div></div>
<div><p class="legend"><b>可组合格式</b>：前缀一个大块（B_r = 4），后缀小块（B_r = 1）</p>
<div class="cells" style="grid-template-columns: 56px repeat(6, 40px)">
<div class="lbl"></div><div class="lbl" style="text-align:center">0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div><div class="lbl" style="text-align:center">3</div><div class="lbl" style="text-align:center">4</div><div class="lbl" style="text-align:center">5</div>
<div class="lbl">req 0</div><div class="c2" style="grid-row: span 4; display:flex; align-items:center; justify-content:center">读 1 次</div><div class="c2" style="grid-row: span 4; display:flex; align-items:center; justify-content:center">读 1 次</div><div class="c0">·</div><div></div><div></div><div></div>
<div class="lbl">req 1</div><div></div><div class="c0">·</div><div></div><div></div>
<div class="lbl">req 2</div><div></div><div></div><div class="c0">·</div><div></div>
<div class="lbl">req 3</div><div></div><div></div><div></div><div class="c0">·</div>
</div></div>
</div>
<p class="legend">列是物理页。页 0、1 是共享前缀。左边：前缀被读了 4 遍，而且每个请求在自己的 CTA 里读，只能走全局内存或 L2。右边：前缀子矩阵是一个 4 × 2 的稠密块，4 个 query 在同一个 CTA 里，前缀只读进 shared memory 一次，4 个 query 共用。两部分各算出一个 attention state，再用 ⊕ 合并。</p>
:::

关键是：**这不需要移动 KV cache 里的任何数据**，只需要给两个子矩阵各算一套 indptr / indices。论文的实验（MLC-Engine，并行生成 n = 4）里，可组合格式让 ITL 降了 13.7%（8B）和 17.4%（70B）。

源码：[[flashinfer:flashinfer/cascade.py@L226|MultiLevelCascadeAttentionWrapper]]（多层共享前缀），F2 讲过它最后用 merge_state 合并；讲解文章 [Cascade Inference](https://flashinfer.ai/2024/02/02/cascade-inference.html)。

::: callout analogy 拼车
4 个人从同一个机场出发，前半程走同一条高速（共享前缀），下了高速各回各家（后缀）。单一格式是每人各叫一辆车，高速路费付 4 次。可组合格式是前半程拼一辆大巴（大块，共用 shared memory），下高速后再各自换小车（小块）。路线图（KV cache）没有任何变化，只是重新安排了谁和谁坐一辆车。
:::

## ⑥ 设计 2：稀疏地读，稠密地算

块稀疏带来一个问题：Tensor Core 要求输入是规整的稠密小矩阵（至少 16 行），而 B_c 可能是 1（每个 KV 位置单独分散存放）。

FlashInfer 的做法分两步：**先把分散的 KV 行“搬”成 shared memory 里连续的一块，再用稠密的 Tensor Core 计算**。

::: flow 从分散的 KV 到稠密的计算（论文图 4）
cfg cw=230 nw=175
n g 0 0.65 "全局内存里分散的 KV" "按 BSR 的 indices 算地址" k=todo w=1.05
n cp 1.1 0.65 "异步拷贝 128 字节一次" "LDGSTS（cp.async）" k=warn w=1.05
n sm 2.2 0.65 "shared memory 里连续的 tile" "和稠密情况完全一样" k=teal w=1.1
n tc 3.35 0.65 "稠密的 FlashAttention" "Tensor Core" k=pass
e g -> cp
e cp -> sm
e sm -> tc
cap: 每个 KV 位置的 head_dim（128 或 256）是连续的，所以每一行的读取仍然是合并访存。搬进 shared memory 之后，稀疏版和稠密版的 kernel 就完全一样了，差别只在加载模块。
:::

- **地址怎么算**：稀疏 KV 的地址来自 BSR 的 indices 数组；稠密 KV 的地址是行号的仿射变换。这和你在 L2 里写的 `phys = block_tables[b][p]`、`kv_off = phys * stride + …` 是同一件事。
- **TMA 用不上**：Hopper 的 TMA 只支持固定 stride 的访问，而稀疏 gather 需要任意的行地址。所以 FlashInfer 只在 KV 连续时用 TMA，其他情况退回 Ampere 风格的异步拷贝。
- **代价**：论文附录 B 测了 page size = 1（最极端的稀疏）对比连续 KV：decode 几乎没有差别（1% 以内），prefill 约慢 10%（FA3 模板上稀疏加载占更多寄存器，只能用更小的 KV tile）。

::: callout analogy 仓库拣货
订单上的货散在仓库的各个货架上（分散的 KV 页）。拣货员按清单（indices）把它们一件件取下来，整齐码在一个托盘上（shared memory 里的连续 tile）。之后上流水线加工（Tensor Core），流水线只认托盘，不关心货原来在哪个货架。拣货多花一点时间，但流水线本身完全不用改。
:::

源码：Hopper 上的稀疏加载主循环 [[flashinfer:include/flashinfer/attention/hopper/sparse_mainloop.cuh|sparse_mainloop.cuh]]（对比稠密的 [[flashinfer:include/flashinfer/attention/hopper/mainloop.cuh|mainloop.cuh]]）；FA2 模板在 [[flashinfer:include/flashinfer/attention/prefill.cuh|prefill.cuh]]。

## ⑦ 设计 3：多种 tile 尺寸，以及 GQA 的 head-group 融合

### 按算术强度选 tile

标准的 FlashAttention-2 只有少数几种 tile（比如 128 × 64），对 A100 上的 prefill 最优，对 decode 却很浪费：decode 只有 1 个 query，128 行的 tile 里 127 行是空的。

FlashInfer 准备了 query 方向 {1, 16, 32, 64, 128} × KV 方向 {32, 64, 128} 的多种 tile，按两条规则选：

1. 算出这一批的**平均 query 长度**，选不小于它的最小 query tile。
2. 在寄存器和 shared memory 的限制下，选能让 SM 占用率最高的 KV tile。

query tile 为 1 时用 CUDA core（Tensor Core 至少要 16 行），其余用 Tensor Core；Hopper 的 FA3 模板用 64 的倍数（wgmma 的要求）。

源码里的选择函数是 [[flashinfer:include/flashinfer/utils.cuh@L411|FA2DetermineCtaTileQ]]：平均长度超过 64 选 128，16 到 64 之间选 64，再短选 16。调用处是 [[flashinfer:include/flashinfer/attention/scheduler.cuh@L612-L619|PrefillSplitQOKVIndptr 里按 avg_packed_qo_len 选]]。

### head-group 融合：让 decode 也能用 Tensor Core

“平均 query 长度”用的是**打包后**的长度：[[flashinfer:include/flashinfer/attention/scheduler.cuh@L557-L562|packed_qo_len = qo_len × gqa_group_size]]。这就是论文附录 A 的 head-group 融合，也就是你在 L2 里做的事：同一个 KV head 下的几个 query head，和 query 长度这一维合并成一个大的“行”维度。

GQA 组大小 g = 4，每个请求 2 个 query（比如 speculative decoding）：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 110px repeat(8, minmax(52px, 1fr)); min-width: 560px">
<div class="lbl">打包后的行</div><div>0</div><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>7</div>
<div class="lbl">(token, head)</div><div class="c0">t0 h0</div><div class="c0">t0 h1</div><div class="c0">t0 h2</div><div class="c0">t0 h3</div><div class="c1">t1 h0</div><div class="c1">t1 h1</div><div class="c1">t1 h2</div><div class="c1">t1 h3</div>
<div class="lbl">读的 KV head</div><div class="c2">kv 0</div><div class="c2">kv 0</div><div class="c2">kv 0</div><div class="c2">kv 0</div><div class="c2">kv 0</div><div class="c2">kv 0</div><div class="c2">kv 0</div><div class="c2">kv 0</div>
</div></div>
<p class="legend">2 个 token × 4 个 head = 8 行，全部读同一个 KV head：KV 只读进 shared memory 一次，8 行一起用。packed_qo_len = 2 × 4 = 8，所以选 16 行的 query tile，可以用 Tensor Core。query 足够长时（prefill），query 维度本身就够填满 tile，融合就不那么关键了。</p>
:::

::: callout analogy 不同尺寸的锅
食堂准备了几种尺寸的锅：一两个人吃饭用小锅（decode，tile 16），一桌人用中锅（64），全班聚餐用大锅（prefill，128）。用大锅给一个人炒菜，火力（Tensor Core）大部分都浪费了。head-group 融合则是把同一桌点了同一道菜的几个人（共享 KV head 的 query head）合成一锅炒，凑满中锅的容量。
:::

## ⑧ 设计 4：可定制的 attention 模板 + JIT

### 观察：变体越来越多，但骨架都一样

GQA、MLA、滑动窗口、logits soft-cap（Gemma、Grok）、attention sink、自定义 mask、把 RoPE 融合进 attention、甚至不用 softmax 的 FlashSigmoid……为每一种手写一个专用 kernel 是不可持续的。但它们的骨架都是 FlashAttention：只是在几个固定的位置做了一点小修改。论文把这个设计空间写成：

```text
f_epilogue( scan( f_logits( f_q(Q) · f_k(K) ) ) · f_v(V) )
```

### 做法：变体 = 几个固定位置的函数（functor）

受 FlexAttention 启发，FlashInfer 的模板在这些位置留了“插槽”：

| 插槽 | 在哪里调用 | 能实现什么 |
|---|---|---|
| `QueryTransform` / `KeyTransform` / `ValueTransform` | 读进 Q、K、V 之后 | 融合 RoPE、归一化、投影（论文里 StreamingLLM 的 RoPE 融合只加了约 20 行） |
| `LogitsTransform` | q·k 之后、softmax 之前 | logits soft-cap、ALiBi |
| `LogitsMask` | 同上 | 因果 mask、滑动窗口、自定义 mask |
| online softmax 的 m / d 更新 | 每个 KV 块 | attention sink（往分母里加一项） |
| `OutputTransform` | 写回之前 | 最后的归一化 |
| `use_softmax` | 整体开关 | 不用 softmax 的变体（FlashSigmoid） |

用户写一段 CUDA 代码定义一个变体类，JIT 编译器把它填进模板，生成专用 kernel，编译好的 .so 缓存起来（F5 讲过 JIT 的缓存机制）。

::: flow 一个变体是怎么变成 kernel 的（论文图 5）
cfg cw=230 nw=175
n decl 0 0.65 "变体类的 CUDA 代码" "REGISTER_LOGITS_MASK …" k=vio w=1.05
n args 0 1.9 "额外参数" "张量名、标量名、dtype" k=vio
n tpl 1.1 1.3 "FlashAttention 模板" "FA2（sm80+）/ FA3（sm90）" k=acc w=1.05
n gen 2.2 1.3 "生成的 .cu" "变体内联进主循环" k=warn
n so 3.3 1.3 "编译、缓存" "~/.cache/flashinfer" k=pass
e decl -> tpl
e args -> tpl
e tpl -> gen
e gen -> so
cap: 变体的函数在编译期内联进 kernel，没有函数调用和分支的开销，所以生成的 kernel 和手写的专用 kernel 一样快。代价是第一次遇到某个组合时要编译几十秒。
:::

**一个真实的例子：attention sink**（gpt-oss 等模型用到），完整定义在 [[flashinfer:flashinfer/jit/attention/variants.py@L17-L54|variants.py]]，核心是三段：

```cpp
struct AttentionSink : AttentionVariantBase {
  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return (kv_idx + qo_len + window_left >= kv_len + qo_idx);   // 滑动窗口 + 因果
  })
  REGISTER_M_D_UPDATE(params, kv_tile_idx, qo_head_idx, m, d, scale, {
    // 第一个 KV 块时，把每个 head 的 sink 值当作一个额外的 logit 加进 online softmax 的 (m, d)
  })
  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    return output * scale * d_rcp;                                // 用加了 sink 的分母归一化
  })
};
```

测试里调用它的方式（[[flashinfer:tests/attention/test_attention_sink.py@L183-L210|test_attention_sink.py]]）：把变体名、变体代码、额外的张量 `sink` 和标量 `sm_scale` 作为 `jit_args` 传给 `BatchPrefillWithRaggedKVCacheWrapper`，就得到一个专用的 kernel。

默认的变体也是用同样的机制写的：[[flashinfer:include/flashinfer/attention/variants.cuh@L31-L32|DefaultAttention]]，[[flashinfer:include/flashinfer/attention/variants.cuh@L72-L73|soft-cap 在 LogitsTransform 里做 tanh]]，[[flashinfer:include/flashinfer/attention/variants.cuh@L88|滑动窗口在 LogitsMask 里]]。注册宏在 [[flashinfer:include/flashinfer/attention/variant_helper.cuh@L29-L69|variant_helper.cuh]]。

::: callout analogy 标准菜谱上的“自选调料”格子
FlashAttention 是一道标准菜谱：备料、切配、下锅、调味、装盘，步骤永远不变。FlashInfer 在菜谱上留了几个空格：“备料时额外处理”（QueryTransform，比如先腌一下，也就是 RoPE）、“调味”（LogitsTransform，soft-cap）、“哪些食材不要”（LogitsMask，滑动窗口）、“装盘前”（OutputTransform）。客人在空格里填上自己的要求，厨房（JIT）按填好的菜谱现做一份。菜谱的主体是同一份，所以每道定制菜都和招牌菜一样快。
:::

**为什么用 CUDA / CUTLASS 而不是 Triton**（论文附录 C）：需要 warp specialization、TMA 这类 Triton 当时还不支持或处于实验阶段的硬件特性，以及寄存器级别的精细控制和直接写 PTX。论文附录 G 和 FlexAttention 的对比里，FlashInfer 在 H100 上的各种变体都更快，长序列时差距更大。

## ⑨ 设计 5：负载均衡调度、plan / run 与 CUDA graph

### 观察：一个请求一个 CTA 会严重不均衡

serving 时同一个 batch 里的请求长度差别很大。最朴素的做法是每个请求（或每个 query tile）一个 CTA，最长的那个决定整体时间，其他 SM 早早就闲着了。

### 做法：先切块，再用最小堆分配（论文算法 1）

1. 每个工作单元的代价：cost = α · l_q + β · l_kv。源码里是 [[flashinfer:include/flashinfer/attention/scheduler.cuh@L981|cost_function = 2 · qo_len + kv_len]]。
2. 每块 KV 的最大长度 L_kv = 总工作量 ÷ CTA 数。比这个长的请求被切成几块。
3. 把所有块按长度从大到小排序。
4. 用一个**最小堆**维护每个 CTA 目前的总代价，每次把下一块分给最闲的 CTA。源码：[[flashinfer:include/flashinfer/attention/scheduler.cuh@L1097-L1120|MinHeap cta_cost_heap，pop 最小，加上 cost，再 push 回去]]。
5. 被切开的请求，各块的结果是 attention state，最后用 ⊕ 合并。

**算一个例子**：4 个 decode 请求，KV 长度 [1000, 100, 100, 50]，4 个 CTA，代价只看 l_kv。总量 1250，L_kv = ⌈1250 / 4⌉ = 313。请求 A 被切成 313、313、313、61 四块；排序后是 313、313、313、100、100、61、50。

::: raw
<style>
.sched { margin: 10px 0; }
.sched .row { display: grid; grid-template-columns: 70px 1fr 60px; align-items: center; gap: 8px; margin: 3px 0; font-size: 12.5px; }
.sched .bar { display: flex; height: 22px; background: var(--surface-2); border-radius: 4px; overflow: hidden; border: 1px solid var(--border); }
.sched .seg { height: 100%; font: 10.5px/22px var(--mono); text-align: center; color: var(--text); border-right: 1px solid var(--surface); overflow: hidden; white-space: nowrap; }
.sched .sa { background: var(--f-acc-bg); } .sched .sb { background: var(--f-warn-bg); } .sched .sc { background: var(--f-pass-bg); } .sched .sd { background: var(--f-vio-bg); }
.sched .t { font-family: var(--mono); text-align: right; }
.sched .lab { font-weight: 600; margin: 8px 0 2px; font-size: 13px; }
</style>
<div class="sched">
<div class="lab">朴素：每个请求一个 CTA（最慢的 CTA 决定总时间：1000）</div>
<div class="row"><span>CTA 0</span><div class="bar"><div class="seg sa" style="width:100%">A 1000</div></div><span class="t">1000</span></div>
<div class="row"><span>CTA 1</span><div class="bar"><div class="seg sb" style="width:10%">B</div></div><span class="t">100</span></div>
<div class="row"><span>CTA 2</span><div class="bar"><div class="seg sc" style="width:10%">C</div></div><span class="t">100</span></div>
<div class="row"><span>CTA 3</span><div class="bar"><div class="seg sd" style="width:5%">D</div></div><span class="t">50</span></div>
<div class="lab" style="margin-top:12px">FlashInfer：切块 + 最小堆分配（总时间：313，快约 3.2 倍）</div>
<div class="row"><span>CTA 0</span><div class="bar"><div class="seg sa" style="width:31.3%">A₁ 313</div></div><span class="t">313</span></div>
<div class="row"><span>CTA 1</span><div class="bar"><div class="seg sa" style="width:31.3%">A₂ 313</div></div><span class="t">313</span></div>
<div class="row"><span>CTA 2</span><div class="bar"><div class="seg sa" style="width:31.3%">A₃ 313</div></div><span class="t">313</span></div>
<div class="row"><span>CTA 3</span><div class="bar"><div class="seg sb" style="width:10%">B</div><div class="seg sc" style="width:10%">C</div><div class="seg sa" style="width:6.1%">A₄</div><div class="seg sd" style="width:5%">D</div></div><span class="t">311</span></div>
</div>
<p class="legend">分配过程：前三块 313 分给 CTA 0、1、2；之后 CTA 3 一直是最闲的，依次拿到 B（100）、C（100）、A₄（61）、D（50），总共 311。A 的四块各产生一个 attention state，最后用 ⊕ 合并成 A 的输出。</p>
:::

**和 Stream-K 的区别**：思路来自 GEMM 里的 Stream-K，但 Stream-K 用原子加合并部分结果，顺序不确定。serving 要求输出**确定**，所以 FlashInfer 不用原子操作，同样的长度输入总是产生同样的合并顺序（C2 讲过浮点原子加为什么不确定）。另外，不需要切的短请求直接写最终输出，不经过 workspace（论文附录 D.2）。

::: callout analogy 搬家公司派活
4 辆车（CTA），一户要搬 1000 箱，三户各搬 100、100、50 箱。一户派一辆车的话，搬 1000 箱的那辆要跑很久，其他三辆早早收工。调度员的做法是：先把那 1000 箱拆成几车，然后每次把下一批货交给“目前活最少”的那辆车（最小堆）。最后所有车差不多同时收工，大户的几车货到了新家再拼起来（⊕）。
:::

### plan / run：每一步排一次班，所有层共用

调度要读每个请求的长度、排序、用堆分配，这些都在 **CPU** 上做（`plan`）；结果异步拷进 GPU 的 workspace；GPU 上的 kernel（`run`）只按计划执行。

::: flow 一个 decode step 的时间线
cfg cw=215 nw=170
n up 0 0 "长度更新" "每个请求 +1 个 token" k=acc
n pl 1 0 "plan（CPU）" "切块、排序、最小堆" k=warn
n cp 2 0 "异步拷进 workspace" "固定的偏移位置" k=warn
n gr 3 0 "replay CUDA graph" "" k=pass
n l0 2 1.3 "第 0 层 run" "按计划执行" k=pass
n l1 3 1.3 "第 1 … L−1 层 run" "同一个计划" k=pass w=1.05
e up -> pl
e pl -> cp
e cp -> gr
e gr -> l0
e l0 -> l1
cap: 一个 decode step 里，所有层的 attention 形状完全相同（序列长度一样），所以 plan 只做一次，开销被几十层平摊。论文把这种分离称为 Inspector-Executor 模型，是并行计算里处理不规则负载的经典做法。
:::

### 怎么兼容 CUDA graph

CUDA graph 一旦捕获，kernel 的 grid 大小和参数（包括指针）就固定了。但每一步的调度结果都不一样。FlashInfer 的解决办法：

- **persistent kernel**：grid 大小固定（每个 SM 1–2 个 CTA），每个 CTA 循环处理 plan 分给它的工作列表，工作量变了，grid 不用变。源码：[[flashinfer:include/flashinfer/attention/persistent.cuh@L179|BlockBatchPagedAttentionPersistent]]（attention）和 [[flashinfer:include/flashinfer/attention/persistent.cuh@L546|BlockBatchReductionPersistent]]（⊕ 合并），每个 CTA 用 [[flashinfer:include/flashinfer/attention/persistent.cuh@L33-L37|get_block_coord(params, work_idx)]] 从 plan 里取出自己的下一块工作。
- **workspace 的每一段都在固定的偏移**：按上限一次分配好，所以传给 kernel 的指针每一步都一样，只是指向的内容变了。
- plan 本身在 CPU 上，不进 graph；graph 里只有 run。

::: callout analogy 公交公司的排班表
公交公司的车辆和线路是固定的（CUDA graph 里固定的 grid 和指针），但每天的客流不同。调度员每天早上根据客流排一张班表（plan），贴在车库固定的公告栏上（workspace 的固定位置）。司机每天来了都去同一个公告栏看今天跑哪几趟（persistent kernel 读工作列表）。车和公告栏都没变，变的只是公告栏上的内容。
:::

源码：Python 端 [[flashinfer:flashinfer/decode.py@L2124|BatchDecodeWithPagedKVCacheWrapper.plan]]（F1 用过）、[[flashinfer:flashinfer/prefill.py@L5245|BatchPrefillWithPagedKVCacheWrapper]]；C++ 端的计划 [[flashinfer:include/flashinfer/attention/scheduler.cuh@L692|PrefillPlanInfo]]、[[flashinfer:include/flashinfer/attention/scheduler.cuh@L1204|HolisticPlanInfo]] 和 [[flashinfer:include/flashinfer/attention/scheduler.cuh@L1285|TwoStageHolisticPlan]]。

## ⑩ 编程接口：serving 框架怎么用它

论文清单 1 的流程（简化）：

```python
workspace = torch.empty(...)                          # 预先分配，地址固定
attn = AttentionWrapper(attn_spec, task_info, workspace)   # 初始化时 JIT 编译 kernel
# 捕获阶段：每种配置先做一次 plan，再把所有层的 run 捕获进 CUDA graph
attn.plan(seqlen_info)
with torch.cuda.graph(g):
    for layer in layers:
        ...; attn.run(...); ...
# 生成循环：每一步只做 plan + replay
while not finished:
    seqlen_info.update()
    attn.plan(seqlen_info)      # CPU 上排班，写进 workspace
    g.replay()                  # GPU 上所有层按新计划执行
```

不同平均 query 长度、不同可组合格式的配置会编译成不同的 kernel、捕获成不同的 graph，运行时由 serving 框架按当前情况挑一个。论文里列出的集成方是 vLLM、SGLang 和 MLC-Engine。vLLM 的调用见 [[vllm:vllm/v1/attention/backends/flashinfer.py@L477-L481|flashinfer.py 里的 plan]]，SGLang 的见 [[sglang:python/sglang/srt/layers/attention/flashinfer_backend.py|flashinfer_backend.py]]。

## ⑪ 效果，以及和这套题的对应

论文（FlashInfer v0.2，A100 / H100）报告的结果：

| 场景 | 结果 | 主要来自哪个设计 |
|---|---|---|
| SGLang 端到端 serving（Llama 3.1 8B / 70B） | 对比 Triton 后端，inter-token latency 降低 29–69% | 全部 |
| 变长 batch 的 kernel | 均匀和偏斜分布下明显快于 FlashAttention | 负载均衡调度（⑨） |
| decode kernel | 快于 FlashAttention | 多种 tile 尺寸（⑦） |
| StreamingLLM 长上下文 | 延迟降低 28–30%；融合 RoPE 的 kernel 带宽利用率高 1.6–3.7 倍 | JIT 变体（⑧） |
| 并行生成（n = 4） | ITL 降低 13–17% | 可组合格式（⑤） |
| 稀疏 KV（page size 1）的代价 | decode < 1%，prefill 约 10% | 稀疏加载（⑥） |

你在这套题里已经亲手写过的部分：

| FlashInfer 的设计 | 对应的题 |
|---|---|
| FlashAttention 主循环、online softmax | [T5](t5_flash_attn.html) |
| 分页 KV、页表的间接寻址 | [L1](l1_paged_kv_write.html)、[L2](l2_paged_decode.html) |
| GQA head-group 融合（补到 16 行用 Tensor Core） | [L2](l2_paged_decode.html) |
| split-KV 和 ⊕ 合并 | [L3](l3_split_kv_decode.html)、[F2](f2_merge_state.html) |
| CSR 页表（BSR 的特例）、plan / run | [F1](f1_flashinfer_paged.html) |
| 稀疏地读、稠密地算（gather 进 shared memory） | [C3](c3_transpose.html)、[C4](c4_gemm.html) 的 tile 加载 |
| JIT、源码分层、贡献流程 | [F5](f5_flashinfer_contrib.html) |
| 不排序的采样（FlashInfer 的另一个子系统） | [F4](f4_topk_topp_renorm.html) |

## ⑫ 局限，以及之后的发展

论文自己指出的局限：

- **只做 forward**。训练需要可定制的 backward 模板，论文把它列为未来工作。
- **稀疏 KV 在 Hopper 上用不了 TMA**，只能走异步拷贝，prefill 约慢 10%。B_c 很大（比如 128）时可以用 TMA，但会牺牲格式的灵活性。

论文发表之后，FlashInfer 的范围扩展了很多（以本仓库钉住的版本为准）：Blackwell 上的 attention 后端（TensorRT-LLM、cuDNN、CuTe DSL）、MLA、采样（F4）、norm 和激活（F3）、FP8 / FP4 的 GEMM 和 MoE。核心的设计原则没有变：**统一的数据格式 + 编译期定制 + 运行期调度**。项目自己的说明见 [[flashinfer:CLAUDE.md|CLAUDE.md]]。

## 源码与资料索引

**论文与文章**：[FlashInfer 论文（arXiv 2501.01005）](https://arxiv.org/abs/2501.01005) · [Cascade Inference 博客](https://flashinfer.ai/2024/02/02/cascade-inference.html) · [Sorting-Free Sampling 博客](https://flashinfer.ai/2025/03/10/sampling.html) · [FlashInfer 文档](https://docs.flashinfer.ai) · [FlexAttention](https://pytorch.org/blog/flexattention/)

**数据结构**：[[flashinfer:include/flashinfer/attention/state.cuh@L29|state_t（attention state）]] · [[flashinfer:include/flashinfer/page.cuh@L38|paged_kv_t]] · [[flashinfer:flashinfer/sparse.py@L363|BlockSparseAttentionWrapper]] · [[flashinfer:flashinfer/cascade.py@L226|MultiLevelCascadeAttentionWrapper]]

**定制**：[[flashinfer:include/flashinfer/attention/variants.cuh@L31|DefaultAttention]] · [[flashinfer:include/flashinfer/attention/variant_helper.cuh@L29|REGISTER_* 宏]] · [[flashinfer:flashinfer/jit/attention/variants.py@L17|AttentionSink 变体]]

**调度**：[[flashinfer:include/flashinfer/attention/scheduler.cuh@L981|cost_function]] · [[flashinfer:include/flashinfer/attention/scheduler.cuh@L1097|最小堆分配]] · [[flashinfer:include/flashinfer/utils.cuh@L411|FA2DetermineCtaTileQ]] · [[flashinfer:include/flashinfer/attention/persistent.cuh@L179|persistent kernel]]

**相关讲解**：[T5](t5_flash_attn.html) · [L2](l2_paged_decode.html) · [L3](l3_split_kv_decode.html) · [F1](f1_flashinfer_paged.html) · [F2](f2_merge_state.html) · [F5](f5_flashinfer_contrib.html)
