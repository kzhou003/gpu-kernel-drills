title: L1 讲解：分页 KV cache 的写入
lede: vLLM 每一步 forward 都要先把新 token 的 K、V 写进分页的 cache。这一页讲清楚为什么要分页、slot 是怎么算出来的（带一张能手算的 block table）、cache 的四维布局怎么变成一维偏移、padding token 为什么是 −1，以及 vLLM 的 CUDA 实现和这里的 Triton 版逐行对应。

::: callout info 怎么读这一页
对照 [[here:solutions/l1_paged_kv_write.py|l1_paged_kv_write.py]] 读。这题代码不多，难点是**数据布局**。L2、L3、F1 都建立在这一题的布局上，值得把第 ③ 节的例子完整算一遍。
:::

## ① 先看全貌：为什么要分页

推理服务里同时有很多序列，每个序列的长度在不断增长，而且事先不知道最终会多长。如果每个序列预留一段连续的 KV cache：

| 方式 | 问题 |
|---|---|
| 按最大长度预留（比如 4096） | 大部分序列只用几百个 token，剩下的空间全部浪费 |
| 按需扩容 | 要找更大的连续空间并整体拷贝，还会留下碎片 |

**PagedAttention** 借用了操作系统虚拟内存的做法：把 cache 切成固定大小的 **block**（page，通常 16 个 token），所有序列从同一个池子里按需取 block。每个序列有一张 **block table**，记录它的第 i 个逻辑 block 存在哪个物理 block。

::: flow 两个序列共用一个 block 池
cfg cw=200 nw=160
n s0 0 0 "序列 0" "block_table = [9, 3, 7]" k=acc w=1.1
n s1 0 1.3 "序列 1" "block_table = [2, 5]" k=vio w=1.1
n p 1.4 0.65 "物理 block 池" "所有序列共享" k=warn
n b9 2.6 0 "物理块 9、3、7" "序列 0 的第 0、1、2 页" k=acc w=1.1
n b2 2.6 1.3 "物理块 2、5" "序列 1 的第 0、1 页" k=vio w=1.1
e s0 -> p "查表"
e s1 -> p "查表"
e p -> b9
e p -> b2
cap: 物理 block 的编号是乱序的，因为它们是从池子里随用随取的。序列增长时只要再要一个 block、在 block table 末尾加一项，不需要移动已有的数据。浪费的只有每个序列最后一个没填满的 block。
:::

::: callout analogy 活页夹
每个序列像一个活页夹（block table），里面的每一页（block）都是从同一摞空白活页纸（block 池）里随手抽的，每页正好写 16 行。活页夹里第 3 页可能是那摞纸里的第 7 张，没关系，活页夹记住顺序就行。写满一页就再抽一张夹进去，永远不需要把整本重新誊抄到一本更厚的本子上。两个序列有相同的开头（共享前缀）时，甚至可以把同一张纸夹进两个活页夹里。
:::

## ② cache 的布局：四维变一维

`k_cache` 和 `v_cache` 的形状都是 `[num_blocks, block_size, H_kv, D]`，并且是连续的。

::: flow 从 (物理块, 块内位置, head, d) 到扁平偏移
cfg cw=230 nw=180
n nd 0 0 "k_cache[b][o][h][d]" "四维下标" k=acc
n sl 1 0 "slot = b·BS + o" "前两维合并" k=warn
n off 2 0 "偏移 = slot·(H·D) + h·D + d" "" k=pass w=1.2
e nd -> sl
e sl -> off
cap: 因为 cache 连续，前两维 [num_blocks, block_size] 可以合并成一个“slot”维度：一共 num_blocks × block_size 个 slot，每个 slot 存一个 token 的 H·D 个数。slot s 的数据就在扁平偏移 [s·H·D, (s+1)·H·D)。
:::

所以写一个 token，kernel 只需要知道它的 slot，然后把 H·D 个数连续地写进去。

## ③ slot_mapping：纯 PyTorch 的 gather

[[here:solutions/l1_paged_kv_write.py@L15-L25|slot_mapping（15–25 行）]]

```python
logical_block  = positions // block_size                       # 序列里的第几页
physical_block = block_table[seq_idx, logical_block].long()    # 查表：一次 gather 出 T 个
return physical_block * block_size + positions % block_size
```

**手算一个例子**：block_size = 4，

```text
block_table = [[9, 3, 7],     # 序列 0
               [2, 5, 0]]     # 序列 1（第 3 项是 padding，没用到）
```

这一步有 3 个新 token：

| token | seq_idx | position | 逻辑块 = pos // 4 | 物理块 = block_table[seq][逻辑块] | 块内偏移 = pos % 4 | slot |
|---|---|---|---|---|---|---|
| 0 | 0 | 5 | 1 | 3 | 1 | 3·4 + 1 = **13** |
| 1 | 0 | 9 | 2 | 7 | 1 | 7·4 + 1 = **29** |
| 2 | 1 | 2 | 0 | 2 | 2 | 2·4 + 2 = **10** |

在 cache 里的位置（每格是一个 slot，按物理块分组）：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 86px repeat(4, minmax(44px, 1fr)) 16px 86px repeat(4, minmax(44px, 1fr)); min-width: 720px">
<div class="lbl">物理块 2</div><div>8</div><div>9</div><div class="c3">10<small>tok 2</small></div><div>11</div><div class="lbl"></div>
<div class="lbl">物理块 3</div><div>12</div><div class="c0">13<small>tok 0</small></div><div>14</div><div>15</div>
<div class="lbl">物理块 7</div><div>28</div><div class="c0">29<small>tok 1</small></div><div>30</div><div>31</div><div class="lbl"></div>
<div class="lbl">物理块 9</div><div>36</div><div>37</div><div>38</div><div>39</div>
</div></div>
<p class="legend">蓝色是序列 0 的新 token，紫色是序列 1 的。slot 13 在 k_cache 里的扁平偏移是 13 × H × D。</p>
:::

`block_table[seq_idx, logical_block]` 是 PyTorch 的**高级索引**：两个长度为 T 的整数张量一起作下标，得到 T 个元素，相当于 `[block_table[seq_idx[t], logical_block[t]] for t in range(T)]`，但是一次向量化完成，没有 Python 循环。在 GPU 上它本身就是一个 gather kernel。

在真实的 vLLM 里，slot_mapping 由调度器在 CPU 上准备好，和输入一起送到 GPU。

## ④ kernel：一个 token 一个 program

[[here:solutions/l1_paged_kv_write.py@L28-L46|reshape_and_cache_kernel（28–46 行）]]

```python
t    = tl.program_id(0).to(tl.int64)           # 第几个新 token
slot = tl.load(slot_ptr + t)                   # 它的 slot；−1 表示 padding
offs = tl.arange(0, BLOCK)                     # H·D 个值，当作一维
n_valid = tl.where(slot >= 0, HD, 0)
mask = offs < n_valid
k = tl.load(key_ptr + t * key_stride + offs, mask=mask)
tl.store(k_cache_ptr + slot * HD + offs, k, mask=mask)     # v 同理
```

- **一个 program 搬一个 token 的 H·D 个数**。H = 8、D = 128、fp16 时是 2 KB 连续数据，读和写都是连续的。不同 token 的 slot 是分散的，但每个 token 内部是连续的，所以写入仍然有不错的带宽（题目页实测 1254 GB/s）。
- **slot = −1 的 padding token**：CUDA graph 要求每次 batch 的形状固定，实际 token 不够时会补齐，补出来的 token 的 slot 是 −1。`n_valid = 0` 让 mask 全为 False，这个 program 什么都不读也不写。
- **为什么不写成 `(offs < HD) & (slot >= 0)`**：意思一样，但 `slot >= 0` 是标量，和向量做 `&` 在 CPU 解释器上会报错。注释（34–36 行）说明了这个兼容性写法。
- **`key_stride`**：key 可能是 fused QKV 投影结果的一个 view。比如 `qkv` 的形状是 `[T, 3, H, D]`，`key = qkv[:, 1]`，每行之间隔着 3·H·D 个元素，但每行内部的 H·D 个数是连续的。传行 stride 进来就能直接读，不用拷贝。
- **全部用 int64**：`slot * HD` 很容易超过 2³¹，例如 10 万个 block × 16 × 1024 ≈ 16 亿。

::: flow 一个 token 的旅程
cfg cw=220 nw=170
n kin 0 0 "key[t]（H·D 个连续值）" "行 stride 可以不是 H·D" k=acc w=1.1
n s 1 0 "slot = slots[t]" "−1 → 全部 mask 掉" k=warn
n kc 2 0 "k_cache[slot·H·D …]" "连续写 H·D 个值" k=pass w=1.1
e kin -> s
e s -> kc
cap: kernel 里没有 block table。查表在 slot_mapping 那一步就完成了，搬数据的 kernel 只看到一个一维的 slot 号。这样把“地址计算”和“搬数据”分开，两个都很简单。
:::

## ⑤ host 端的检查

[[here:solutions/l1_paged_kv_write.py@L49-L62|reshape_and_cache（49–62 行）]]

```python
assert k_cache.is_contiguous() and v_cache.is_contiguous()
assert t.stride(2) == 1 and t.stride(1) == D       # 每个 token 的 [H, D] 必须连续
reshape_and_cache_kernel[(T,)](..., HD, BLOCK=triton.next_power_of_2(HD))
```

kernel 把一个 token 的 H·D 个数当成一维连续数组读，所以只要求 `[H, D]` 这两维连续，第 0 维（token）可以有任意 stride。cache 必须完全连续，否则 `slot · H · D` 的公式就不成立。

## ⑥ 对照真实实现

**vLLM 的 CUDA 版**和这里几乎逐行对应：[[vllm:csrc/cache_kernels.cu@L336|reshape_and_cache_flash_kernel]]

| Triton 参考解 | vLLM CUDA |
|---|---|
| `t = tl.program_id(0)` | [[vllm:csrc/cache_kernels.cu@L347|token_idx = blockIdx.x]] |
| `slot = tl.load(slot_ptr + t)`，−1 时全部 mask | [[vllm:csrc/cache_kernels.cu@L348-L352|slot_idx < 0 时直接 return]] |
| `slot * HD` | [[vllm:csrc/cache_kernels.cu@L353-L365|block_idx · block_stride + block_offset · page_stride]]（支持非连续的 cache 和 HND 布局） |
| `key_ptr + t * key_stride` | [[vllm:csrc/cache_kernels.cu@L358-L359|key + token_idx · key_stride]] |
| 直接拷贝 | 还支持写入时量化成 FP8（`CopyWithScaleOp`） |

旧版的 [[vllm:csrc/cache_kernels.cu@L278-L284|reshape_and_cache_kernel]] 用的是另一种布局（把 D 再切成 16 字节的小块），是为老的 paged attention kernel 的向量化读取设计的。

**FlashInfer** 用的是另一种 page 描述方式：不是 block table，而是 CSR 格式的 `indptr` + `indices`（F1 会讲）。对应的写入接口是 [[flashinfer:flashinfer/page.py@append_paged_kv_cache|append_paged_kv_cache]]，CUDA 实现在 [[flashinfer:include/flashinfer/page.cuh@L419|AppendPagedKVCacheKernel]]，page 结构体是 [[flashinfer:include/flashinfer/page.cuh@L38|paged_kv_t]]。

## ⑦ 常见坑与自测

- **slot 用了 int32**：大 cache 上 `slot * HD` 溢出，写到错误的位置，表现为其他序列的 KV 被覆盖。
- **把 position 当成全局 token 下标**：position 是 token 在**它自己的序列**里的位置，不是在这一批新 token 里的下标。
- **忘了处理 −1**：padding token 写到 slot −1，也就是 cache 起点之前，越界写。
- **cache 不连续**（比如 `k_cache[:, :, :4]` 切出来的 view）：公式不成立。参考解用 assert 挡住了。

::: answer 自测 1：block_size = 16，序列的 block_table = [4, 11, 2]，position = 40 的 token 的 slot 是多少？
逻辑块 = 40 // 16 = 2，物理块 = block_table[2] = 2，块内偏移 = 40 % 16 = 8。slot = 2 × 16 + 8 = 40。（巧合：这次 slot 和 position 相同。）
:::

::: answer 自测 2：一个序列现在长 33 个 token，block_size = 16，它占了几个 block？浪费了多少个 slot？
⌈33 / 16⌉ = 3 个 block，共 48 个 slot，用了 33 个，浪费 15 个。连续分配如果按最大长度 4096 预留，会浪费 4063 个。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/l1_paged_kv_write.py|solutions/l1_paged_kv_write.py]] · [[here:mine/l1_paged_kv_write.py|mine/l1_paged_kv_write.py]] · [[here:tests/test_l1_paged_kv_write.py|tests/test_l1_paged_kv_write.py]] · [[here:tests/_paged.py|tests/_paged.py（测试用的 paged 数据构造）]] · [[here:content/l1.md|content/l1.md]]

**vLLM**：[[vllm:csrc/cache_kernels.cu@L336|reshape_and_cache_flash_kernel]] · [[vllm:csrc/cache_kernels.cu@L278|reshape_and_cache_kernel（旧布局）]]

**FlashInfer**：[[flashinfer:flashinfer/page.py@append_paged_kv_cache|append_paged_kv_cache]] · [[flashinfer:include/flashinfer/page.cuh@L38|paged_kv_t]] · [[flashinfer:tests/attention/test_page.py|test_page.py]]

**论文**：[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
