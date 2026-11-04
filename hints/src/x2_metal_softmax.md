title: X2 讲解：Metal 里的行归约（simdgroup + threadgroup 内存）
lede: 把 C2 的 reduction 和 T2 的 softmax 搬到 Apple GPU 上。这一页逐行读参考解：一个 threadgroup 一行、线程怎么跨步扫一行、simd_sum / simd_max 对应 CUDA 的什么、tg_reduce 为什么让每个 simdgroup 都做第二轮归约（于是所有线程都拿到结果）、为什么要两次 barrier，以及三遍读的 softmax 为什么在短行上比 MLX 慢一倍。

::: callout info 怎么读这一页
对照 [[here:solutions/x2_metal_softmax.py|x2_metal_softmax.py]] 读。前置：[C2 讲解](c2_reduce_sum.html)（两层归约）、[T2 讲解](t2_softmax.html)（减 max 的 softmax）、[X1 讲解](x1_metal_elementwise.html)（metal_kernel 和 grid 的语义）。这一页主要讲 Metal 版和 CUDA 版的不同。
:::

## ① 先看全貌：一个 threadgroup 负责一行

::: flow 一行 softmax 的三遍（每遍都读整行）
cfg cw=225 nw=170
n p1 0 0 "第 1 遍：跨步求 max" "每个线程扫 n / nthr 个" k=acc
n r1 1 0 "tg_reduce<max>" "simd_max + threadgroup 内存" k=teal w=1.1
n p2 2 0 "第 2 遍：Σ exp(x − m)" "" k=acc
n r2 2 1.3 "tg_reduce<sum>" "" k=teal
n p3 1 1.3 "第 3 遍：exp(x − m) / s" "写回 out" k=pass w=1.05
e p1 -> r1
e r1 -> p2
e p2 -> r2
e r2 -> p3
cap: 和 T2 不同，这里的整行不在寄存器里，而是每一遍都从内存重新读。好处是行多长都能处理；代价是读三次。T2 的 softmax 一次把整行读进寄存器，只读一次。
:::

| CUDA（C2） | Metal（X2） |
|---|---|
| block | threadgroup：一行一个 |
| warp | simdgroup（32 个线程） |
| `__shfl_down_sync` 循环 5 轮 | `simd_sum(v)` / `simd_max(v)`：一个函数调用，硬件做完 |
| `__shared__ float warp_sums[8]` | `threadgroup float scratch[32]` |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `lane = threadIdx.x & 31` | `thread_index_in_simdgroup`（内置属性） |
| `warp = threadIdx.x >> 5` | `simdgroup_index_in_threadgroup`（内置属性） |

## ② kernel 开头：内置属性和 auto

[[here:solutions/x2_metal_softmax.py@L28-L38|_SOFTMAX 开头（28–38 行）]]

```cpp
uint row  = threadgroup_position_in_grid.x;       // 第几个 threadgroup = 第几行
uint tid  = thread_position_in_threadgroup.x;     // 组内第几个线程
uint nthr = threads_per_threadgroup.x;
uint lane = thread_index_in_simdgroup;            // 0..31
uint sg   = simdgroup_index_in_threadgroup;       // 第几个 simdgroup
uint nsg  = (nthr + 31) / 32;
threadgroup float scratch[32];
auto xr = x + row * n;
```

- 这些名字都是 Metal 的内置属性，函数体里用到了，MLX 就自动加进签名（X1 讲过）。Metal 直接给出 lane 和 simdgroup 编号，不用自己算 `& 31`、`>> 5`。
- `threadgroup float scratch[32]`：每个 simdgroup 一个槽，最多 1024 / 32 = 32 个 simdgroup。
- **为什么用 `auto`**（注释 37 行）：MLX 可能把很小的输入放在 `constant` 地址空间而不是 `device`，写死 `const device T*` 就会类型不匹配。用 `auto` 让编译器推断。

host 端（[[here:solutions/x2_metal_softmax.py@L82-L87|82–87 行]]）：`grid=(rows * t, 1, 1), threadgroup=(t, 1, 1)`。别忘了 X1 讲的：grid 是**线程总数**，所以 rows 行就是 rows × t 个线程。

### 每行几个线程

[[here:solutions/x2_metal_softmax.py@L77-L79|_threads（77–79 行）]]：ncols < 1024 时取不小于 ncols 的 2 的幂（至少 32），否则 1024。

| ncols | 线程数 t | 每个线程扫几个 |
|---|---|---|
| 1 | 32 | 1（31 个线程闲着） |
| 100 | 128 | 1（28 个闲着） |
| 1000 | 1024 | 1 |
| 4096 | 1024 | 4 |
| 32000 | 1024 | 31–32 |

## ③ tg_reduce：每个线程都拿到结果

[[here:solutions/x2_metal_softmax.py@L14-L25|tg_reduce（14–25 行）]]

```cpp
template <bool IS_MAX>
inline float tg_reduce(float v, threadgroup float* scratch, uint lane, uint sg, uint nsg) {
    v = IS_MAX ? simd_max(v) : simd_sum(v);          // ① simdgroup 内归约
    if (lane == 0) scratch[sg] = v;                  // ② 每个 simdgroup 交一份
    threadgroup_barrier(mem_flags::mem_threadgroup); // ③ 等大家都交完
    float init = IS_MAX ? -INFINITY : 0.0f;
    v = lane < nsg ? scratch[lane] : init;           // ④ 读回部分结果
    v = IS_MAX ? simd_max(v) : simd_sum(v);          //    每个 simdgroup 都做一遍
    threadgroup_barrier(mem_flags::mem_threadgroup); // ⑤ 让调用方能马上复用 scratch
    return v;
}
```

和 C2 的 block 归约结构一样，有两个不同：

**不同 1：simd_sum 一次调用就完成 simdgroup 内的归约。** 而且它的结果**每个 lane 都有**，不像 C2 的 `__shfl_down_sync` 只有 lane 0 是对的。

**不同 2：第二轮归约每个 simdgroup 都做，而不是只有第 0 个。** C2 只需要一个数去做 atomicAdd，所以只让 warp 0 归约。softmax 不同：每个线程后面都要用 max 和 sum 去算自己负责的那些元素。如果只有 simdgroup 0 有结果，还得再写一次 scratch、再 barrier 一次广播给大家。让每个 simdgroup 都读一遍 scratch 并自己归约，算了重复的活，但省掉了一次广播，所有线程都直接拿到结果。

::: flow tg_reduce 里的数据流（4 个 simdgroup）
cfg cw=250 nw=170
n s0 0 0 "simdgroup 0" "simd_sum → 所有 lane 都有 p0" k=teal w=1.12
n s3 0 1.3 "simdgroup 3" "simd_sum → p3" k=teal
n sc 1.05 0.65 "scratch[0..3]" "threadgroup 内存" k=warn
n b 2.05 0.65 "barrier" "" k=todo
n r 3.05 0.65 "每个 simdgroup 读 scratch" "再 simd_sum，人人都有总和" k=pass w=1.1
e s0 -> sc "lane 0 写"
e s3 -> sc
e sc -> b
e b -> r
cap: 第 ④ 步里 lane ≥ nsg 的线程填 init（max 填 −inf，sum 填 0），和 C2 里 lane ≥ WARPS 填 0 是同一个道理。
:::

**为什么最后还要一次 barrier（⑤）**：tg_reduce 会被调用两次（先 max 后 sum），两次用的是同一个 scratch。没有第 ⑤ 步的话，一个跑得快的 simdgroup 可能已经进入第二次 tg_reduce，在第 ② 步把自己的部分和写进 scratch，而另一个慢的 simdgroup 还在第 ④ 步读第一次的 max，就会读到被覆盖的值。这叫“读后写”冲突。

::: callout analogy 开会传达结果
C2 的做法是：各组组长把数字交给秘书（scratch），只有第一组的组长汇总，汇总结果交给总部（atomicAdd），其他人不需要知道。X2 不一样，每个人后面都要用这个总数干活。如果还是只让第一组汇总，就得再开一次会把结果告诉大家。参考解的做法是：所有组长交完数字后，**每个组都自己抄一份白板上的数字、自己加一遍**。大家重复做了一点加法，但不用再开第二次会。散会前再等一下（第二次 barrier），确保所有人都抄完了，才允许有人擦白板写下一个议题。
:::

## ④ 三遍扫描和 RMSNorm

[[here:solutions/x2_metal_softmax.py@L40-L49|40–49 行]]

```cpp
for (uint c = tid; c < n; c += nthr) m = max(m, float(xr[c]));        // 第 1 遍
m = tg_reduce<true>(m, ...);
for (uint c = tid; c < n; c += nthr) s += metal::exp(float(xr[c]) - m);   // 第 2 遍
s = tg_reduce<false>(s, ...);
for (uint c = tid; c < n; c += nthr) out[row * n + c] = T(metal::exp(float(xr[c]) - m) * inv);  // 第 3 遍
```

`for (c = tid; c < n; c += nthr)` 是一行之内的 grid-stride loop（C1、C2 讲过）：同一轮里相邻线程读相邻元素，合并访存。

RMSNorm（[[here:solutions/x2_metal_softmax.py@L52-L68|52–68 行]]）是同样的结构，只需要一次归约（Σx²），所以只扫两遍。

### 为什么短行比 MLX 慢

题目页的实测（M2，fp16，2048 行）：

| N | mx.softmax | 参考解 |
|---|---|---|
| 1024 | 310 us | 640 us |
| 4096 | 716 us | 1152 us |
| 32000 | 4432 us | 4032 us |

MLX 的 softmax 按行长选两种实现（[[mlx:mlx/backend/metal/softmax.cpp@L53|axis_size > looped_limit ? "looped_" : "block_"]]，[[mlx:mlx/backend/metal/softmax.cpp@L13|SOFTMAX_LOOPED_LIMIT = 4096]]）：

- 行不超过 4096 时用 [[mlx:mlx/backend/metal/kernels/softmax.h@L11|softmax_single_row]]：每个线程一次读 N_READS 个元素放进寄存器（[[mlx:mlx/backend/metal/kernels/softmax.h@L26-L36|AccT ld 数组]]），整行只读一次，就像 T2。
- 行更长时用 [[mlx:mlx/backend/metal/kernels/softmax.h@L101|softmax_looped]]：多遍读。

参考解不管多长都读三遍，所以 N ≤ 4096 时慢了将近一倍；N = 32000 时两边都是多遍读，参考解反而略快。

::: callout analogy 抄书
三遍法像抄一本书的索引：第一遍从头翻到尾找最大的页码，第二遍再翻一遍算总数，第三遍再翻一遍写出来。书很薄时（短行），完全可以把整本书放在桌上一眼看完（读进寄存器），只翻一遍。书太厚放不下（长行），才只好一遍遍地翻。MLX 根据书的厚度选办法，参考解永远翻三遍。
:::

## ⑤ 对照 MLX 源码

- softmax：[[mlx:mlx/backend/metal/kernels/softmax.h@L11|softmax_single_row]]，其中 [[mlx:mlx/backend/metal/kernels/softmax.h@L23|threadgroup AccT local_max]] 就是这里的 scratch，[[mlx:mlx/backend/metal/kernels/softmax.h@L50-L52|simd_max 之后写 local_max]] 就是 tg_reduce 的第 ①② 步。
- RMSNorm：[[mlx:mlx/backend/metal/kernels/rms_norm.metal@L13|rms_single_row]]，[[mlx:mlx/backend/metal/kernels/rms_norm.metal@L44-L64|simd_sum → barrier → 第二轮 simd_sum → rsqrt]]。它把 rsqrt 的结果写进 threadgroup 内存再广播，是“只让一个 simdgroup 归约 + 广播”的另一种写法。
- dispatch 逻辑：[[mlx:mlx/backend/metal/softmax.cpp@L46-L65|softmax.cpp]]，怎么根据行长选 kernel 和 threadgroup 大小。

## ⑥ 常见坑与自测

- **把 grid 写成行数**：`grid=(rows, 1, 1)` 只会启动 rows 个线程（X1 讲的语义），每行只有一个线程或者根本没有。
- **少了第二次 barrier**：两次 tg_reduce 之间偶尔读到被覆盖的 scratch，结果随机出错。
- **第二轮没给 lane ≥ nsg 填 init**：读到 scratch 里上一次留下的垃圾值。
- **用 `const device T*` 而不是 `auto`**：小输入走 `constant` 地址空间时编译失败。
- **在 half 里求 exp 的和**：和 T2 一样，统一转 float。

::: answer 自测 1：ncols = 100 时，t = 128，nsg 是多少？第二轮归约里每个 simdgroup 读 scratch 的哪几个槽？
nsg = (128 + 31) / 32 = 4。每个 simdgroup 的 lane 0–3 读 scratch[0..3]，lane 4–31 填 init。4 个 simdgroup 各自做同样的归约，得到同样的结果。
:::

::: answer 自测 2：把 softmax 改成 online 的“两遍法”，第一遍同时维护什么？tg_reduce 要怎么改？
第一遍每个线程维护自己的 (m, s)：遇到更大的值时 `s = s·exp(m − m_new) + exp(x − m_new)`（T2 第 ⑤ 节的公式）。归约时不能分别对 m 和 s 归约，要把两个 (m, s) 按 `m = max(m₁, m₂)`、`s = s₁·exp(m₁ − m) + s₂·exp(m₂ − m)` 合并，所以 tg_reduce 要改成归约一个二元组，simd_max / simd_sum 不能直接用，得用 `simd_shuffle_xor` 自己写。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/x2_metal_softmax.py|solutions/x2_metal_softmax.py]] · [[here:mine/x2_metal_softmax.py|mine/x2_metal_softmax.py]] · [[here:tests/test_x2_metal_softmax.py|tests/test_x2_metal_softmax.py]] · [[here:content/x2.md|content/x2.md]]

**MLX**：[[mlx:mlx/backend/metal/kernels/softmax.h@L11|softmax.h]] · [[mlx:mlx/backend/metal/kernels/rms_norm.metal@L13|rms_norm.metal]] · [[mlx:mlx/backend/metal/softmax.cpp@L46|softmax.cpp]] · [[mlx:mlx/backend/metal/kernels/layer_norm.metal|layer_norm.metal]]

**文档**：[Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)（SIMD-group functions、threadgroup_barrier 两节） · [MLX · Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
