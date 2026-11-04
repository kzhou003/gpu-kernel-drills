title: M3 讲解：FP8 细粒度量化与 block-scaled GEMM
lede: DeepSeek-V3 / DeepGEMM 的 FP8 配方：激活每 1×128 一个 scale，权重每 128×128 一个 scale，GEMM 每走一个 128 宽的 K 块就乘一次 scale。这一页先用一个能手算的例子说明 e4m3 能表示什么、为什么粒度要细，再逐行读参考解的量化 kernel 和 GEMM kernel，讲清楚 scale 为什么可以从求和里提出来，以及 BLOCK_K 为什么必须是 128。

::: callout info 怎么读这一页
对照 [[here:solutions/m3_fp8_block_gemm.py|m3_fp8_block_gemm.py]] 读。GEMM 的骨架是 [T4](t4_matmul.html)，量化 kernel 的结构是 [T3](t3_layernorm.html) 那种“一个 program 一组、组内归约”。需要 sm_89 以上的 GPU 才能跑 FP8 的 tl.dot（RTX 50 系列可以）。
:::

## ① 先看全貌：为什么用 FP8，为什么要 scale

FP8 的每个数只有 1 个字节，和 bf16 相比：权重和激活的显存、带宽都减半，Tensor Core 的 FP8 算力约是 bf16 的 2 倍。代价是**表示范围和精度都很有限**。

| | e4m3（本题用的） | bf16 |
|---|---|---|
| 位数 | 1 符号 + 4 指数 + 3 尾数 | 1 + 8 + 7 |
| 最大有限值 | **448** | 约 3.4 × 10³⁸ |
| 最小正规数 | 2⁻⁶ ≈ 0.0156 | 约 1.2 × 10⁻³⁸ |
| 相邻两个数的相对间隔 | 1/8 = 12.5% | 1/128 |

范围只有 448，权重和激活不能直接转成 FP8：大的会溢出，小的会变成 0。所以要先除以一个 **scale**，把一组数缩放到 [−448, 448] 里，存 FP8 的同时存下 scale，用的时候再乘回去。

::: flow 量化、存储、反量化
cfg cw=215 nw=170
n x 0 0 "一组 bf16 / fp32 值" "比如 128 个" k=acc
n s 1 0 "scale = amax / 448" "fp32，每组一个" k=warn
n q 2 0 "q = clamp(x / scale)" "转成 fp8 e4m3" k=vio
n d 2 1.3 "x ≈ q · scale" "GEMM 里乘回去" k=pass
e x -> s
e s -> q
e q -> d
cap: 每组的最大绝对值正好映射到 448，充分利用 FP8 的范围。scale 本身用 fp32 存，开销是每 128 个数多 4 个字节（约 3%）。
:::

## ② 一个能手算的量化例子

为了好算，假设组大小是 4（实际是 128）：x = [0.5, −2.0, 0.03, 1.2]。

| 步骤 | 0.5 | −2.0 | 0.03 | 1.2 |
|---|---|---|---|---|
| amax = 2.0，scale = 2.0 / 448 = 0.004464 | | | | |
| x / scale | 112 | −448 | 6.72 | 268.8 |
| 舍入到最近的 e4m3 | 112 | −448 | **6.5** | **256** |
| 反量化 q · scale | 0.500 | −2.000 | 0.0290 | 1.143 |
| 相对误差 | 0 | 0 | 3.3% | 4.8% |

e4m3 在 [4, 8) 之间的间隔是 0.5，在 [256, 448] 之间的间隔是 32，所以 6.72 只能舍入到 6.5，268.8 只能舍入到 256。相对误差最大约 6%（间隔 12.5% 的一半）。

### 为什么粒度要细：一个离群值

如果整个张量共用一个 scale（per-tensor），而张量里有一个离群值 1000：

| | per-tensor scale | 1×128 的组 scale（这一组的 amax 是 0.01） |
|---|---|---|
| scale | 1000 / 448 = 2.23 | 0.01 / 448 = 0.0000223 |
| 0.001 / scale | 0.00045 | 44.8 |
| 量化结果 | **远小于 e4m3 的最小正数（2⁻⁹ ≈ 0.002），舍入成 0** | 44.8 → 44，误差 1.8% |

LLM 的激活里经常有少数通道特别大。粒度越细，一个离群值影响的范围越小。DeepSeek-V3 的选择是：**激活每个 token 每 128 个通道一个 scale（1×128），权重每 128×128 一个 scale**。

::: callout analogy 一张照片的曝光
FP8 像一台动态范围很窄的相机，只能分辨 448 级亮度。per-tensor 量化是整张照片只用一个曝光参数：画面里有一盏大灯（离群值），为了不过曝只能把曝光调得很低，于是阴影里的细节全黑了（变成 0）。分块量化是把照片切成很多小块，每块单独测光：有灯的那块调暗，阴影那块调亮，每一块都在自己的 448 级里拍清楚。代价是每块要多记一个曝光参数（scale）。
:::

## ③ 激活量化 kernel：一个 program 一组

[[here:solutions/m3_fp8_block_gemm.py@L20-L32|per_token_group_quant_kernel（20–32 行）]] · host 端 [[here:solutions/m3_fp8_block_gemm.py@L35-L42|35–42 行]]

```python
row, g = tl.program_id(0), tl.program_id(1)            # grid = (M, K / 128)
offs = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
amax  = tl.maximum(tl.max(tl.abs(x), axis=0), eps)    # 全 0 的组得到一个很小的 scale，而不是 0
scale = amax / 448.0
q = tl.clamp(x / scale, -448.0, 448.0)
tl.store(q_ptr + ..., q.to(q_ptr.dtype.element_ty), mask=mask)
tl.store(s_ptr + row * cdiv(K, GROUP_SIZE) + g, scale)
```

- **eps**：一组全是 0 时 amax = 0，scale = 0，下一步 x / scale 就是 0/0 = NaN。取 max(amax, eps) 避免这种情况。
- **clamp**：x / scale 理论上不会超过 448，但浮点舍入可能让它略大一点；e4m3fn 没有 inf，超出范围的值转换后可能变成 NaN。clamp 一下更安全。
- 输出两个张量：x_q 是 `[M, K]` 的 FP8，scales 是 `[M, K/128]` 的 fp32。

**权重量化**（[[here:solutions/m3_fp8_block_gemm.py@L45-L55|block_quant_fp8，45–55 行]]）是离线做一次的，所以直接用 PyTorch：把 `[N, K]` 补齐到 128 的倍数，`view(nb, 128, kb, 128)`，对第 1、3 维取 amax，得到 `[N/128, K/128]` 的 scale。

## ④ GEMM：scale 为什么能从求和里提出来

[[here:solutions/m3_fp8_block_gemm.py@L58-L86|fp8_block_gemm_kernel（58–86 行）]]

把 K 按 128 切成若干块 kb。在第 kb 块里：

- A 的第 m 行这 128 个数共用一个 scale：`A[m, k] ≈ A_q[m, k] · sa[m, kb]`
- B 的这 128×128 个数共用一个 scale：`B[n, k] ≈ B_q[n, k] · sb[n // 128, kb]`

所以：

```text
C[m, n] = Σ_kb Σ_{k ∈ kb} A[m, k] · B[n, k]
        = Σ_kb  sa[m, kb] · sb[n//128, kb] · ( Σ_{k ∈ kb} A_q[m, k] · B_q[n, k] )
                \_________ 每块一个数 _______/   \______ 一次 fp8 tl.dot _____/
```

括号里是纯 FP8 的点积（Tensor Core 做，fp32 累加），算完之后乘上这一块的两个 scale，再累加到 acc 里。**scale 在每个 K 块都不同**，所以不能把所有 K 块的 FP8 乘积先加起来再统一乘 scale，必须每块乘一次。

```python
for kb in range(0, cdiv(K, BLOCK_K)):                 # BLOCK_K == 128 == 一个 scale 组
    a  = tl.load(a_ptrs, ...)                         # [BLOCK_M, 128] fp8
    b  = tl.load(b_ptrs, ...)                         # [128, BLOCK_N] fp8（Bᵀ）
    sa = tl.load(sa_ptr + offs_m * stride_sam + kb)   # [BLOCK_M]：每行一个
    sb = tl.load(sb_ptr + sb_row * stride_sbn + kb)   # 标量：整个 tile 共用一个
    acc += tl.dot(a, b) * (sa[:, None] * sb)
```

::: flow 一轮 K 块
cfg cw=225 nw=175
n a 0 0 "A_q 块 [BM, 128]" "fp8" k=acc
n b 0 1.3 "B_q 块 [128, BN]" "fp8" k=acc
n d 1 0.65 "tl.dot" "Tensor Core，fp32 累加" k=vio
n s 2 0 "sa[m, kb]" "BM 个，每行一个" k=warn
n sb 2 1.3 "sb[n//128, kb]" "1 个" k=warn
n acc 3 0.65 "acc += dot · sa · sb" "fp32" k=pass
e a -> d
e b -> d
e d -> acc
e s -> acc
e sb -> acc
cap: 每一轮多出来的开销是一次 [BM, BN] 的逐元素乘法。题目页的实测是 bf16 cuBLAS 的约 1.7 倍，没到理论上的 2 倍，差距主要就来自这里和流水线深度。
:::

### 两个约束

- **BLOCK_K 必须是 128**（[[here:solutions/m3_fp8_block_gemm.py@L75|75 行]]）：一个 K 块正好对应一个 scale 组。如果 BLOCK_K = 64，一个 scale 要用两轮；如果是 256，一轮里有两个 scale，就不能用一次 tl.dot 了。
- **BLOCK_N 必须整除 128**（[[here:solutions/m3_fp8_block_gemm.py@L71-L72|71–72 行]]，host 端 93 行 assert）：这样一个 tile 的所有列都落在同一个 128 行的权重块里，sb 只是一个标量。BLOCK_N = 256 时，tile 的左右两半要用不同的 sb，就得改成向量。vLLM 的实现允许这种情况（见第 ⑥ 节）。

## ⑤ 量化本身也要时间

题目页的实测（RTX 5090，TFLOP/s）：

| M × N × K | bf16 cuBLAS | FP8 block-scaled | 激活量化本身 |
|---|---|---|---|
| 128 × 7168 × 2048 | 108 | 151 | 6 us |
| 4096 × 7168 × 2048 | 219 | 364 | 35 us |
| 4096 × 4096 × 7168 | 218 | 372 | 112 us |

4096 × 7168 的激活量化要 112 us，占 GEMM 时间的 17%。所以实际系统不会单独做量化，而是把它融合进前一个算子：FlashInfer 的 [[flashinfer:flashinfer/norm/__init__.py@L1381|fused_add_rmsnorm_fp8_block_quant]] 把 F3 的“残差加 + RMSNorm”和这里的 1×128 量化合成一个 kernel。

::: callout analogy 打包发货
FP8 GEMM 像物流公司把货物压缩打包再运输：运费（带宽、算力）减半。但打包本身也要人工（量化 kernel）。如果在仓库出货的那一刻顺手就打好包（和 RMSNorm 融合），就不用专门安排一道打包工序。
:::

## ⑥ 对照真实实现

**vLLM 的 Triton 版**，和参考解几乎一样：

- 激活量化：[[vllm:vllm/model_executor/layers/quantization/utils/fp8_utils.py@L83|_per_token_group_quant_fp8]]，其中 [[vllm:vllm/model_executor/layers/quantization/utils/fp8_utils.py@L131|_absmax = max(max(abs(y)), eps)]]；Python 接口 [[vllm:vllm/model_executor/layers/quantization/utils/fp8_utils.py@L632|per_token_group_quant_fp8]]
- GEMM：[[vllm:vllm/model_executor/layers/quantization/utils/fp8_utils.py@L841|_w8a8_triton_block_scaled_mm]]，[[vllm:vllm/model_executor/layers/quantization/utils/fp8_utils.py@L892-L895|权重 scale 按 offs_bn // group_n 取]]（所以 BLOCK_N 可以大于 128，sb 是一个向量），[[vllm:vllm/model_executor/layers/quantization/utils/fp8_utils.py@L903-L906|accumulator += tl.dot(a, b) * a_s * b_s]]

**DeepGEMM**（DeepSeek 开源的 FP8 GEMM 库，Hopper / Blackwell 上接近 FP8 峰值）：[[deepgemm:README.md|README]]，SM90 的实现 [[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh|sm90_fp8_gemm_1d1d.cuh]]（wgmma + TMA + warp specialization）。它也有 grouped 版本，给 M2 那样的 MoE 用。

**Triton 的 block-scaled tutorial**：[[triton:python/tutorials/10-block-scaled-matmul.py|10-block-scaled-matmul]]（MXFP8 / NVFP4 这类硬件原生支持的 block scale 格式）。

## ⑦ 常见坑与自测

- **BLOCK_K ≠ 128**：scale 和 K 块对不上，结果系统性地错。
- **把 sa、sb 乘在循环外面**：每块的 scale 不同，提不出来。
- **用 e5m2 存激活**：e5m2 范围大（57344）但尾数只有 2 位，精度更差。DeepSeek-V3 的推理用 e4m3。
- **eps 太大**：比如 1e-3，真实的 amax 很小的组会被强行放大 scale，丢精度。
- **scale 用了 bf16**：scale 本身的误差会乘到整个组上。

::: answer 自测 1：M = 4096，K = 7168。激活量化输出的两个张量各多大？
x_q：4096 × 7168 × 1 字节 = 29.4 MB。scales：4096 × 56 × 4 字节 = 0.9 MB。和 bf16 的 58.7 MB 相比，总共省了约 48%。
:::

::: answer 自测 2：一组 128 个数里最大的是 0.9，最小的非零值是 0.0001。量化后 0.0001 还在吗？
scale = 0.9 / 448 = 0.00201，0.0001 / scale = 0.0498。它在 e4m3 的正规数范围里（≥ 2⁻⁶ ≈ 0.0156），落在 [0.03125, 0.0625) 这一段，间隔是 0.0039，舍入到 0.0508，误差约 2%，还在。如果一组里最大值和最小值相差超过约 448 / 2⁻⁹ ≈ 2.3 × 10⁵ 倍，小值就会低于 e4m3 的最小正数而变成 0。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/m3_fp8_block_gemm.py|solutions/m3_fp8_block_gemm.py]] · [[here:mine/m3_fp8_block_gemm.py|mine/m3_fp8_block_gemm.py]] · [[here:tests/test_m3_fp8_block_gemm.py|tests/test_m3_fp8_block_gemm.py]] · [[here:content/m3.md|content/m3.md]]

**实现**：[[vllm:vllm/model_executor/layers/quantization/utils/fp8_utils.py@L841|vLLM w8a8 block-scaled mm]] · [[deepgemm:README.md|DeepGEMM]] · [[flashinfer:flashinfer/norm/__init__.py@L1381|FlashInfer fused_add_rmsnorm_fp8_block_quant]] · [[triton:python/tutorials/10-block-scaled-matmul.py|Triton tutorial 10]]

**论文与规范**：[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)（第 3.3 节 FP8 训练）· [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)（e4m3 / e5m2 的定义）
