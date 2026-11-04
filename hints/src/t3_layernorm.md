title: T3 讲解：LayerNorm / RMSNorm 的 forward
lede: 和 T2 一样一行一个 program，但多了两个陷阱：方差用哪个公式算，以及 padding 位置减完均值之后不再是 0。逐行读参考解，附带 fp32 下“负方差”的具体例子、为什么要把 mean 和 rstd 存下来，以及 RMSNorm 少做了什么。

::: callout info 怎么读这一页
对照 [[here:solutions/t3_layernorm.py|t3_layernorm.py]] 读。结构和 T2 几乎一样（一行一个 program、整行在寄存器里、两次行内归约），如果 T2 还没看，先看 [T2 讲解](t2_softmax.html)。
:::

## ① 先看全貌：两个公式

| | LayerNorm（GPT-2、BERT） | RMSNorm（LLaMA、Qwen、DeepSeek） |
|---|---|---|
| 公式 | y = (x − μ) · rstd · w + b | y = x · rstd · w |
| rstd | 1 / √(σ² + ε)，σ² 是**有偏**方差（÷N） | 1 / √(mean(x²) + ε) |
| 行内归约 | 2 次：Σx 求 μ，Σ(x−μ)² 求 σ² | 1 次：Σx² |
| 参数 | w、b 各 N 个 | 只有 w |
| 额外输出 | 每行的 mean、rstd（backward 要用） | 无 |

::: flow layer_norm_kernel 里一行的数据流
cfg cw=225 nw=165
n x 0 0 "读 x 一行" "padding 填 0" k=acc
n mu 1 0 "mean = Σx / N" "归约 ①" k=teal
n xc 2 0 "xc = where(mask, x−mean, 0)" "padding 重新清零" k=warn w=1.3
n var 2 1.3 "var = Σxc² / N" "归约 ②" k=teal
n rs 1 1.3 "rstd = 1/√(var + ε)" "标量" k=teal
n y 0 1.3 "y = xc · rstd · w + b" "写回一行" k=pass w=1.1
n st 0 2.5 "存 mean[row]、rstd[row]" "给 backward 用" k=vio w=1.1
e x -> mu
e mu -> xc
e xc -> var
e var -> rs
e rs -> y
e y -> st dash
cap: 整行只从显存读一次 x、写一次 y，两次归约都在片上完成。w 和 b 每个 program 都读一遍，但它们只有 N 个数，所有 program 共享，基本都命中 L2。
:::

::: callout analogy 统一考试的标准分
LayerNorm 就是把一个班的成绩换算成标准分：先算平均分（mean），再算每个人偏离平均分多少、偏离的幅度有多大（方差），最后用“偏离 ÷ 标准差”让每个班的成绩都在同一个尺度上，再乘上学校规定的权重（w）、加上基础分（b）。RMSNorm 偷了个懒：不减平均分，直接用“成绩的均方根”做尺度。实践证明对大模型来说效果差不多，还更省事。
:::

## ② kernel 逐行讲

[[here:solutions/t3_layernorm.py@L16-L38|layer_norm_kernel（16–38 行）]]

### 读和均值

```python
row  = tl.program_id(0).to(tl.int64)
cols = tl.arange(0, BLOCK_N)
mask = cols < N
x    = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
mean = tl.sum(x, axis=0) / N
```

前四行和 T2 一样。这里 padding 填的是 **0**（T2 填 -inf），因为下一步是求和：0 不影响 Σx。注意除以的是真实的 N，不是 BLOCK_N。

### 陷阱 1：padding 减完均值就不是 0 了

[[here:solutions/t3_layernorm.py@L27-L29|27–29 行]]

```python
xc  = tl.where(mask, x - mean, 0.0)
var = tl.sum(xc * xc, axis=0) / N
```

padding 位置的 x 是 0，但 `x − mean = −mean`，平方之后是 mean²，会被加进方差。N = 5、BLOCK_N = 8 时：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 170px repeat(8, minmax(48px, 1fr)); min-width: 600px">
<div class="lbl">x</div><div class="c0">2</div><div class="c0">4</div><div class="c0">4</div><div class="c0">5</div><div class="c0">5</div><div class="cx">0</div><div class="cx">0</div><div class="cx">0</div>
<div class="lbl">x − mean（mean = 4）</div><div class="c1">−2</div><div class="c1">0</div><div class="c1">0</div><div class="c1">1</div><div class="c1">1</div><div class="c1" style="border-style:dashed!important">−4</div><div class="c1" style="border-style:dashed!important">−4</div><div class="c1" style="border-style:dashed!important">−4</div>
<div class="lbl">tl.where(mask, …, 0)</div><div class="c2">−2</div><div class="c2">0</div><div class="c2">0</div><div class="c2">1</div><div class="c2">1</div><div class="cx">0</div><div class="cx">0</div><div class="cx">0</div>
</div></div>
<p class="legend">正确的方差 = (4 + 0 + 0 + 1 + 1) / 5 = 1.2。不用 tl.where 的话，三个 padding 各贡献 16，方差变成 (6 + 48) / 5 = 10.8，错了 9 倍。而且 N 恰好是 2 的幂时没有 padding，测试发现不了这个 bug。</p>
:::

`tl.where(cond, a, b)` 是逐元素的三元选择：cond 为 True 取 a，否则取 b。Triton 里没有逐元素的 `if`，都用它。

### 陷阱 2：方差用哪个公式

参考解用的是**两遍法**：先求 mean，再求 Σ(x − mean)²。教科书上还有一个“一遍法”：var = E[x²] − E[x]²，只需要同时累加 Σx 和 Σx²。它在数学上等价，在浮点上可能灾难性地出错：

| | x = [10001, 10002, 10003]，fp32 |
|---|---|
| 真实方差 | 2/3 ≈ 0.667 |
| E[x²] | (100020001 + 100040004 + 100060009) / 3 = 100040004.67 |
| E[x]² | 10002² = 100040004 |
| 问题 | fp32 在 1 亿附近的间隔是 **8**，两个数都只能被存成 8 的倍数（例如 100040008 和 100040000） |
| 一遍法的结果 | 只能是 0、±8 这样的值：可能是 8（错 12 倍），可能是 0（rstd 爆炸），也可能是**负数**（√ 之后是 NaN） |
| 两遍法 | x − mean = [−1, 0, 1]，Σ = 2，÷3 = 0.667 ✓ |

数据的均值远大于它的波动时（LLM 的激活值里经常出现个别很大的通道），两个很大的数相减，有效数字全被抵消掉了，这叫**灾难性抵消**。两遍法先把均值减掉，再平方，就没有这个问题。

整行在寄存器里，所以“两遍”只是对寄存器里的数据算两次，不需要多读显存。如果行太长、必须分段从显存读，又想只读一遍，就用 **Welford 算法**：边读边更新均值和 M2，数值上也是稳定的。PyTorch 的 CUDA LayerNorm 就用它（见第 ⑤ 节）。

::: callout analogy 量珠穆朗玛峰上的一张纸
你想知道一张纸有多厚，于是量“山顶到海平面的高度（含纸）”和“山顶到海平面的高度（不含纸）”，然后相减。两个数都是 8848 米左右，量具的精度是 1 厘米，纸的厚度 0.1 毫米完全淹没在误差里，甚至可能算出负数。正确做法是先站到山顶上（减去均值），再直接量纸（两遍法）。
:::

### rstd、仿射变换和写回

[[here:solutions/t3_layernorm.py@L30-L38|30–38 行]]

```python
rstd = 1.0 / tl.sqrt(var + eps)
w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
y = xc * rstd * w + b
tl.store(y_ptr + row * y_row_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)
tl.store(mean_ptr + row, mean)     # 标量写：每个 program 一个数
tl.store(rstd_ptr + row, rstd)
```

- **eps** 防止一行全相同时方差为 0、除以 0。LayerNorm 默认 1e-5，RMSNorm 常用 1e-6。
- 存 **rstd**（1/σ），而不是 σ 或 var：backward 里用到的正是 rstd，乘法比除法便宜。
- 最后两行是**标量 store**：mean 和 rstd 是标量，`mean_ptr + row` 是一个地址。
- **为什么要存 mean 和 rstd**：backward 计算 dx 需要 x̂ = (x − mean) · rstd。存下来每行只多 8 字节，省得 backward 再做两次归约。

## ③ RMSNorm：少一次归约

[[here:solutions/t3_layernorm.py@L41-L49|rms_norm_kernel（41–49 行）]]

```python
x    = tl.load(..., other=0.0).to(tl.float32)
rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
w    = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
tl.store(y_ptr + ..., (x * rstd * w).to(...), mask=mask)
```

不减均值，所以 padding 填 0 之后平方还是 0，不需要 `tl.where`。没有 b，也不存统计量（这题只要求 forward）。

题目页的实测里 RMSNorm 和 LayerNorm **一样快**（N = 8192 时都是 176 us）：两者都要读 M×N 的 x、写 M×N 的 y，少一次寄存器里的归约几乎不省时间。memory-bound 的 kernel 只有减少主要数据的字节数才有用。

## ④ host 端：reshape 和 num_warps

[[here:solutions/t3_layernorm.py@L52-L73|52–73 行]]

```python
x2 = x.reshape(-1, shape[-1])            # [B, S, N] → [B·S, N]
if x2.stride(-1) != 1: x2 = x2.contiguous()
BLOCK_N = triton.next_power_of_2(N)      # 超过 16384 就报错
num_warps = min(max(BLOCK_N // 256, 1), 8)
```

- **reshape 成二维**：LayerNorm 只对最后一维归一化，前面所有维度都可以合并成“行”。`reshape` 在可能时返回视图，不拷贝。
- **num_warps 的规则**：大约每 256 个元素一个 warp，最少 1、最多 8 个。

| N | BLOCK_N | num_warps | 每线程的 fp32 个数 |
|---|---|---|---|
| 768 | 1024 | 4 | 8 |
| 4096 | 4096 | 8 | 16 |
| 8192 | 8192 | 8 | 32 |

- **返回值**：y 恢复成原来的形状；mean 和 rstd 始终是 fp32 的 `[M]`。

## ⑤ 对照真实实现

- **Triton 官方 tutorial 05**：[[triton:python/tutorials/05-layer-norm.py@L49|_layer_norm_fwd_fused]] 支持任意长的行：用 [[triton:python/tutorials/05-layer-norm.py@L68|for off in range(0, N, BLOCK_SIZE)]] 分段读，求方差时 [[triton:python/tutorials/05-layer-norm.py@L74-L80|再读一遍、减均值后平方]]，也是两遍法，只不过两遍都从显存读。
- **FlashInfer 的 CUDA RMSNorm**：[[flashinfer:include/flashinfer/norm.cuh@RMSNormKernel|RMSNormKernel]]，Σx² 先在线程内累加，再用 [[flashinfer:include/flashinfer/norm.cuh@L97|warp shuffle]] 和 [[flashinfer:include/flashinfer/norm.cuh@L100-L109|shared memory]] 两层归约，正是 C2 的结构。
- **vLLM 的 RMSNorm**：[[vllm:csrc/layernorm_kernels.cu@L15|rms_norm_kernel]]，用 [[vllm:csrc/layernorm_kernels.cu@L68|cub::BlockReduce]] 做 block 归约。F3 会实现它的融合版 `fused_add_rms_norm`。
- **PyTorch 的 CUDA LayerNorm**：[[pytorch:aten/src/ATen/native/cuda/layer_norm_kernel.cu@L58|RowwiseMomentsCUDAKernel]] 用 [[pytorch:aten/src/ATen/native/cuda/layer_norm_kernel.cu@L79-L84|Welford 算法]] 一遍算出均值和方差。

## ⑥ 常见坑与自测

- **忘了 `tl.where`**：N 不是 2 的幂时方差偏大。
- **除以 BLOCK_N 而不是 N**：padding 位置被当成了数据。
- **用无偏方差（÷ N−1）**：`F.layer_norm` 用的是有偏方差（÷ N）。
- **w、b 不转 fp32**：fp16 的 w 和 fp32 的 y 相乘时类型提升规则可能和预期不同，统一转 fp32 最省心。
- **mean、rstd 用了输入的 dtype**：它们应该是 fp32，backward 需要精度。

::: answer 自测 1：x = [1, 1, 1, 1]，LayerNorm 的 y 是多少？为什么不会除以 0？
mean = 1，xc = [0, 0, 0, 0]，var = 0，rstd = 1/√(0 + 1e-5) ≈ 316。y = 0 · 316 · w + b = b。eps 保证了分母不为 0；xc 全是 0，所以 y 就是 bias。
:::

::: answer 自测 2：同一个 x = [3, 4]，LayerNorm（w=1, b=0）和 RMSNorm（w=1）的结果分别是多少？（忽略 eps）
LayerNorm：mean = 3.5，xc = [−0.5, 0.5]，var = 0.25，rstd = 2，y = [−1, 1]。
RMSNorm：mean(x²) = (9 + 16)/2 = 12.5，rstd = 1/√12.5 ≈ 0.283，y ≈ [0.849, 1.131]。
RMSNorm 不减均值，所以输出不以 0 为中心，只是把整体尺度归一化。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/t3_layernorm.py|solutions/t3_layernorm.py]] · [[here:mine/t3_layernorm.py|mine/t3_layernorm.py]] · [[here:tests/test_t3_layernorm.py|tests/test_t3_layernorm.py]] · [[here:content/t3.md|content/t3.md]]

**参考实现**：[[triton:python/tutorials/05-layer-norm.py@L49|Triton tutorial 05]] · [[flashinfer:flashinfer/triton/norm.py|FlashInfer Triton norm]] · [[flashinfer:include/flashinfer/norm.cuh@RMSNormKernel|FlashInfer RMSNormKernel]] · [[vllm:csrc/layernorm_kernels.cu@L15|vLLM rms_norm_kernel]] · [[pytorch:aten/src/ATen/native/cuda/layer_norm_kernel.cu@L58|PyTorch RowwiseMomentsCUDAKernel]]

**API 与背景**：[tl.where](https://triton-lang.org/main/python-api/generated/triton.language.where.html) · [tl.sqrt](https://triton-lang.org/main/python-api/generated/triton.language.sqrt.html) · [Welford 在线方差算法](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm) · [LayerNorm 论文](https://arxiv.org/abs/1607.06450) · [RMSNorm 论文](https://arxiv.org/abs/1910.07467)
