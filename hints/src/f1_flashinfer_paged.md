title: F1 讲解：FlashInfer 的 CSR 分页格式与 plan / run
lede: FlashInfer 的分页 KV 不用补齐的 block table，而用 CSR 三元组（indptr、indices、last_page_len）。这一页用一个三序列的例子把两种格式互相换算一遍，逐行讲清楚参考解里的向量化转换（cumsum、布尔索引），再看 plan / run 为什么要分开、plan 在 CPU 上到底决定了什么。

::: callout info 怎么读这一页
对照 [[here:solutions/f1_flashinfer_paged.py|f1_flashinfer_paged.py]] 读。前置：[L1 讲解](l1_paged_kv_write.html) 的 block table，[L3 讲解](l3_split_kv_decode.html) 的分片。这一题的代码是纯 PyTorch，难点是格式本身。
:::

## ① 先看全貌：同一份数据，两种描述

三个序列，page_size = 4，长度分别是 9、2、5：

::: raw
<div style="display:flex; gap:32px; flex-wrap:wrap">
<div><p class="legend"><b>vLLM 的 block table</b>（补齐到最长）</p>
<div class="cells" style="grid-template-columns: 70px repeat(3, 48px) 70px">
<div class="lbl"></div><div class="lbl" style="text-align:center">0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div><div class="lbl" style="text-align:center">ctx</div>
<div class="lbl">序列 0</div><div class="c0">9</div><div class="c0">3</div><div class="c0">7</div><div>9</div>
<div class="lbl">序列 1</div><div class="c1">2</div><div class="cx">0</div><div class="cx">0</div><div>2</div>
<div class="lbl">序列 2</div><div class="c2">5</div><div class="c2">1</div><div class="cx">0</div><div>5</div>
</div></div>
<div><p class="legend"><b>FlashInfer 的 CSR</b>（不补齐）</p>
<div class="cells" style="grid-template-columns: 120px repeat(6, 44px)">
<div class="lbl">indices</div><div class="c0">9</div><div class="c0">3</div><div class="c0">7</div><div class="c1">2</div><div class="c2">5</div><div class="c2">1</div>
<div class="lbl">indptr</div><div>0</div><div>3</div><div>4</div><div>6</div><div class="lbl"></div><div class="lbl"></div>
<div class="lbl">last_page_len</div><div class="c0">1</div><div class="c1">2</div><div class="c2">1</div><div class="lbl"></div><div class="lbl"></div><div class="lbl"></div>
</div></div>
</div>
<p class="legend">序列 b 的页是 indices[indptr[b] : indptr[b+1]]：序列 0 是 [0:3] = 9, 3, 7；序列 1 是 [3:4] = 2；序列 2 是 [4:6] = 5, 1。灰色格子是 block table 为了补齐而存的无用值。</p>
:::

| 数组 | 形状 | 含义 |
|---|---|---|
| `indptr` | `[B + 1]` | 第 b 个序列的页在 indices 里从 `indptr[b]` 到 `indptr[b+1]`；`indptr[0] = 0`，`indptr[B]` = 总页数 |
| `indices` | `[总页数]` | 所有序列的物理页号首尾相接 |
| `last_page_len` | `[B]` | 每个序列最后一页用了几个 token，取值 1..page_size |

长度可以反推：`ctx[b] = (页数 − 1) × page_size + last_page_len[b]`。序列 0：(3 − 1) × 4 + 1 = 9 ✓。

**为什么用 CSR**：真实 batch 里序列长度相差很大，一个 32K 的序列和 63 个 100 token 的序列放在一起，block table 要补齐到 2048 列，99% 都是浪费。CSR 只存真正用到的页。这和稀疏矩阵的 CSR 格式一模一样：indptr 是行指针，indices 是列下标。

::: callout analogy 一本书的目录
block table 像一张固定大小的表格，每个人一行，每行都预留同样多的格子，用不完就空着。CSR 像一本书：正文（indices）把所有人的页码一个接一个地写下来，前面的目录（indptr）只记录“第 b 个人从正文的第几项开始”。目录只需要 B + 1 个数，正文没有一个空位。
:::

## ② block table → CSR：三步向量化

[[here:solutions/f1_flashinfer_paged.py@L14-L26|block_table_to_csr（14–26 行）]]

### 第 1 步：每个序列几页，cumsum 得到 indptr

```python
num_pages = (ctx + page_size - 1) // page_size          # [9, 2, 5] → [3, 1, 2]
indptr = torch.zeros(B + 1, dtype=torch.int32)
indptr[1:] = torch.cumsum(num_pages, 0)                 # [0, 3, 4, 6]
```

`cumsum` 是前缀和：[3, 1, 2] → [3, 4, 6]，前面补一个 0 就是 indptr。

### 第 2 步：布尔索引，按行优先挑出有效的页

```python
keep = torch.arange(max_blocks)[None, :] < num_pages[:, None]    # [B, max_blocks] 的 bool
indices = block_tables[keep]
```

`arange(3)[None, :]` 是 `[[0, 1, 2]]`，`num_pages[:, None]` 是 `[[3], [1], [2]]`，比较之后广播成：

::: raw
<div class="cells" style="grid-template-columns: 70px repeat(3, 60px); max-width: 300px">
<div class="lbl"></div><div class="lbl" style="text-align:center">0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div>
<div class="lbl">&lt; 3</div><div class="c2">True</div><div class="c2">True</div><div class="c2">True</div>
<div class="lbl">&lt; 1</div><div class="c2">True</div><div class="cx">False</div><div class="cx">False</div>
<div class="lbl">&lt; 2</div><div class="c2">True</div><div class="c2">True</div><div class="cx">False</div>
</div>
<p class="legend">block_tables[keep] 按行优先的顺序取出所有 True 位置的值：9, 3, 7, 2, 5, 1。每个序列的页恰好连在一起、顺序不变，这正是 CSR 需要的。</p>
:::

注释（21–22 行）强调的就是这一点：布尔索引按行优先遍历，所以不需要任何 Python 循环。

### 第 3 步：最后一页的长度

```python
last_page_len = ctx - (num_pages - 1) * page_size      # [9−8, 2−0, 5−4] = [1, 2, 1]
```

等价于 `(ctx − 1) % page_size + 1`。注意不能写成 `ctx % page_size`：ctx 正好是 page_size 的整数倍时它会得到 0，而正确答案是 page_size（最后一页是满的）。这也是为什么要求 ctx ≥ 1。

## ③ CSR → block table：反过来

[[here:solutions/f1_flashinfer_paged.py@L29-L40|csr_to_block_table（29–40 行）]]

```python
num_pages = indptr[1:] - indptr[:-1]                   # 相邻差：[3, 1, 2]
keep = torch.arange(max_blocks)[None, :] < num_pages[:, None]
block_tables = torch.zeros(B, max_blocks, dtype=torch.int32)
block_tables[keep] = indices                           # 布尔索引赋值，同样是行优先
context_lens = (num_pages - 1) * page_size + last_page_len
```

和正向转换对称：`indptr[1:] − indptr[:-1]` 是 cumsum 的逆运算；布尔索引赋值按同样的行优先顺序把 indices 填回去，补齐的位置是 0。

## ④ plan / run：为什么要分开

[[here:solutions/f1_flashinfer_paged.py@L43-L60|flashinfer_decode（43–60 行）]]

```python
workspace = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
wrapper.plan(indptr, indices, last, Hq, Hkv, D, page_size, q_data_type=q.dtype, kv_data_type=k_cache.dtype)
return wrapper.run(q, (k_cache, v_cache), return_lse=return_lse)
```

::: flow 一个 decode step 里 plan 和 run 各做什么
cfg cw=255 nw=175
n b 0 0 "新的 batch" "indptr、last_page_len" k=acc
n p 1 0 "plan()（CPU）" "要不要分片、分几段" k=warn
n w 2 0 "workspace" "调度结果拷到 GPU" k=vio
n r0 1 1.3 "run()：第 0 层" "只启动 kernel" k=pass
n r1 2 1.3 "run()：第 1 … L−1 层" "复用同一个 plan" k=pass w=1.1
e b -> p
e p -> w
e w -> r0
e r0 -> r1 "复用"
cap: 同一个 decode step 里，每一层 attention 的 batch 形状（每个序列多长、用哪些页）完全相同，只有 q 和 KV cache 的内容不同。所以 plan 每一步只做一次，32 层或 80 层的 run() 都复用它。
:::

**plan 决定了什么**：L3 里你手动选 `num_splits`；FlashInfer 的 plan 根据 batch 大小、每个序列的页数和 GPU 的 SM 数自动选。核心逻辑在 [[flashinfer:include/flashinfer/attention/scheduler.cuh@L150|BatchDecodeWithPagedKVCacheWorkEstimationDispatched]]：

- [[flashinfer:include/flashinfer/attention/scheduler.cuh@L182|max_grid_size = 每 SM 能驻留的 block 数 × SM 数]]
- [[flashinfer:include/flashinfer/attention/scheduler.cuh@L183-L190|如果 batch_size × Hkv 已经够填满 GPU，就不分片]]（L3 里 B = 64 时的结论）
- [[flashinfer:include/flashinfer/attention/scheduler.cuh@L197-L199|否则二分搜索每片的页数]]，让总的分片数刚好不超过 max_grid_size。具体实现是 [[flashinfer:include/flashinfer/attention/scheduler.cuh@L74-L98|PartitionPagedKVCacheBinarySearchMinNumPagePerBatch]]：长序列自动分更多片，短序列分得少或者不分。

题目页的实测：FlashInfer 在三种形状下都自动接近了你在 L3 里手动找到的最优值（98 us、98 us、180 us）。

**为什么 plan 要在 CPU 上做**：它需要读每个序列的长度做一些分支判断和搜索，这些逻辑在 CPU 上写起来简单；结果写进 workspace 之后，run() 只读 GPU 上的数据，不和 CPU 同步。这样 run() 可以被 **CUDA graph** 捕获：graph 只能录制 GPU 上的操作，不能包含 CPU 逻辑。

**workspace** 是一块预先分配好的 GPU 内存（这里 128 MB），用来放 plan 的调度结果和 split-KV 的中间结果（L3 里的 o_part 和 lse）。真实系统里它只分配一次，所有层、所有 step 共用。

::: callout analogy 航班的排班表
每天早上调度员（plan）根据当天的乘客名单排一次班：哪些航线要加开几架飞机（分几片）、每架飞机装哪些乘客。排班表贴在机场的公告板上（workspace）。之后这一天里，同一条航线的每一班都按同一张表起飞（每一层的 run），不需要每次起飞前都重新排班。第二天名单变了（下一个 decode step），再排一次。
:::

## ⑤ 真实系统里怎么调用

- **vLLM**：[[vllm:vllm/v1/attention/backends/flashinfer.py@L477-L481|self._context.plan(paged_kv_indptr=…, paged_kv_last_page_len=…)]]，传的就是这里的 CSR 三元组。vLLM 内部用 block table，在构建 attention metadata 时转换成 CSR，正是这题的 `block_table_to_csr`。
- **SGLang**：[[sglang:python/sglang/srt/layers/attention/flashinfer_backend.py|flashinfer_backend.py]]
- **FlashInfer 本身**：[[flashinfer:flashinfer/decode.py@L1354|BatchDecodeWithPagedKVCacheWrapper]]，[[flashinfer:flashinfer/decode.py@L2124|plan()]]；C++ 端 [[flashinfer:csrc/batch_decode.cu@L39|BatchDecodeWithPagedKVCachePlan]] 和 [[flashinfer:csrc/batch_decode.cu@L129|BatchDecodeWithPagedKVCacheRun]]。kernel 里读页的结构体是 [[flashinfer:include/flashinfer/page.cuh@L38|paged_kv_t]]，它的 [[flashinfer:include/flashinfer/page.cuh@L57-L58|indptr 字段]]就是这里的 indptr。
- **文档**：[attention API](https://docs.flashinfer.ai/api/attention.html)。

## ⑥ 常见坑与自测

- **`last_page_len = ctx % page_size`**：整页的序列得到 0。
- **indptr 用了 int64**：FlashInfer 的接口期望 int32，类型不对会报错或者读错。
- **用 Python 循环逐个序列拼 indices**：结果对，但 batch 大时很慢；而且这段代码在每个 decode step 都要跑。
- **把 plan 放进每一层**：每层都做一次 CPU 调度和同步，延迟明显变大。

::: answer 自测 1：page_size = 16，四个序列的长度是 [16, 17, 1, 40]。写出 indptr 和 last_page_len。
页数 = [1, 2, 1, 3]，indptr = [0, 1, 3, 4, 7]。last_page_len = [16, 1, 1, 8]。第一个序列正好一整页，所以是 16 而不是 0。
:::

::: answer 自测 2：indptr = [0, 2, 2, 5]，这是合法的吗？
indptr[1] = indptr[2] = 2，说明第二个序列有 0 页，即长度为 0。CSR 本身能表示，但这一题的转换要求 ctx ≥ 1（last_page_len 必须在 1..page_size 之间），decode 时每个序列至少有一个 token，所以不会出现这种情况。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/f1_flashinfer_paged.py|solutions/f1_flashinfer_paged.py]] · [[here:mine/f1_flashinfer_paged.py|mine/f1_flashinfer_paged.py]] · [[here:tests/test_f1_flashinfer_paged.py|tests/test_f1_flashinfer_paged.py]] · [[here:content/f1.md|content/f1.md]]

**FlashInfer**：[[flashinfer:flashinfer/decode.py@L1354|BatchDecodeWithPagedKVCacheWrapper]] · [[flashinfer:include/flashinfer/attention/scheduler.cuh@L150|decode 调度启发式]] · [[flashinfer:csrc/batch_decode.cu@L39|plan（C++）]] · [[flashinfer:include/flashinfer/page.cuh@L38|paged_kv_t]] · [[flashinfer:tests/attention/test_batch_decode_kernels.py|test_batch_decode_kernels.py]] · [文档](https://docs.flashinfer.ai/api/attention.html)

**调用方**：[[vllm:vllm/v1/attention/backends/flashinfer.py@L477|vLLM]] · [[sglang:python/sglang/srt/layers/attention/flashinfer_backend.py|SGLang]]

**论文**：[FlashInfer](https://arxiv.org/abs/2501.01005)
