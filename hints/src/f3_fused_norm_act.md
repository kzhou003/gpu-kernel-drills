title: F3 讲解：fused_add_rmsnorm 与 silu_and_mul
lede: Transformer 每一层里最常见的两个融合小算子。这一页先画出 pre-norm 的残差流，说明 fused_add_rmsnorm 为什么要原地改两个张量；再逐行读参考解：view 和 reshape 的区别、为什么用 fp32 的 h 求 rstd（FlashInfer 也是这样）、SwiGLU 的 gate/up 布局和二维 grid，最后对照 FlashInfer 和 vLLM 的 CUDA 实现。

::: callout info 怎么读这一页
对照 [[here:solutions/f3_fused_norm_act.py|f3_fused_norm_act.py]] 读。RMSNorm 本身在 [T3 讲解](t3_layernorm.html) 讲过，这里只讲“融合残差加”和“原地”带来的新问题。silu_and_mul 的结构和 [T1 讲解](t1_add_gelu.html) 几乎一样。
:::

## ① 先看全貌：pre-norm 的一层

LLaMA、Qwen、DeepSeek 等模型的一层 decoder：

::: flow pre-norm decoder layer 里这两个算子在哪
cfg cw=200 nw=160
n r0 0 0 "residual" "残差流" k=vio
n n1 1 0 "fused_add_rmsnorm" "residual += x，x = norm" k=warn w=1.1
n at 2.1 0 "attention" "输出写进 x" k=acc
n n2 3.1 0 "fused_add_rmsnorm" "residual += x，x = norm" k=warn w=1.1
n gu 3.1 1.3 "gate_up_proj" "[T, 2I]" k=acc
n sm 2.1 1.3 "silu_and_mul" "[T, 2I] → [T, I]" k=pass
n dn 1 1.3 "down_proj" "输出写进 x" k=acc
e r0 -> n1
e n1 -> at
e at -> n2
e n2 -> gu
e gu -> sm
e sm -> dn
cap: 每个子层（attention、MLP）的输出 x 都要先加回残差流，再做 RMSNorm 交给下一个子层。fused_add_rmsnorm 把“加残差”和“归一化”合成一个 kernel：一层里出现两次。silu_and_mul 是 SwiGLU MLP 的门控，夹在两个 GEMM 之间。
:::

`fused_add_rmsnorm(x, residual, w)` 做三件事，**两个张量都原地修改**：

```text
h = x + residual                  # fp32 里算
residual ← h                      # 新的残差流，下一个子层还要用
x        ← h · rsqrt(mean(h²) + eps) · w     # 归一化后交给下一个子层
```

::: callout analogy 银行账户和每月的报表
residual 是你的银行账户余额（残差流），每个子层的输出 x 是这个月的收支。每到月底要做两件事：把收支记进账户（residual += x），再根据新的余额出一份标准化的报表给下一个部门（x = norm(residual)）。分开做要打开账本两次、抄写两次；融合之后，打开一次账本，把余额更新了，同时把报表也写好了。而且报表就写在收支单那张纸的背面（x 原地覆盖），不用再领一张新纸。
:::

## ② 省了什么：字节数和 kernel 数

| 做法 | 读 | 写 | kernel 数 |
|---|---|---|---|
| 分开：`h = x + r`，`r = h`，`x = rmsnorm(h)` | x、r、h（3 次） | h、x（2 次），r 若单独拷贝再加 | 2–3 |
| 融合 | x、r（2 次） | r、x（2 次） | **1** |

题目页的实测：T = 4096、H = 4096、bf16，每读写一遍是 32 MB，融合版 4 遍共 128 MB，88 us，1522 GB/s，已经到带宽上限。

decode 时（T = 16）时间都在 6 us 左右，这是一次 kernel launch 的下限。那时字节数已经不重要，**少一个 kernel 就是少 6 us**。一层里 RMSNorm × 2、残差加 × 2、RoPE、SwiGLU、量化等十来个小算子，不融合时它们加起来能和 GEMM 相当（题目页的面试追问）。

## ③ fused_add_rmsnorm 逐行讲

[[here:solutions/f3_fused_norm_act.py@L17-L28|fused_add_rmsnorm_kernel（17–28 行）]]

```python
x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
r = tl.load(r_ptr + row * r_stride + cols, mask=mask, other=0.0).to(tl.float32)
h = x + r
tl.store(r_ptr + row * r_stride + cols, h.to(r_ptr.dtype.element_ty), mask=mask)   # 新的残差
rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / N + eps)
w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
tl.store(x_ptr + row * x_stride + cols, (h * rstd * w).to(x_ptr.dtype.element_ty), mask=mask)
```

和 T3 的 rms_norm_kernel 相比只多了两行：读 residual、写回 residual。值得注意的细节：

- **rstd 用的是 fp32 的 h**，不是存回 residual 之后的 bf16 版本。h 存进 bf16 的 residual 时会被舍入，但这一行的归一化用的是舍入之前的值。FlashInfer 的 CUDA 版也是这样：它把 fp32 的 h 暂存在 shared memory 里（[[flashinfer:include/flashinfer/norm.cuh@L449-L458|residual_vec 存 bf16，x_vec 存 fp32 进 smem_x]]），第二遍从 smem_x 读 fp32 的值来归一化（[[flashinfer:include/flashinfer/norm.cuh@L492-L496|x_vec.load(smem_x …)]]）。测试要和 FlashInfer 对拍，所以这个细节要一致。
- **先写 residual 再写 x 没有冲突**：两个是不同的张量，x 写的是归一化后的值，residual 写的是 h。整行都在寄存器里，读完之后再写，不会读到自己写的东西。
- 和 T3 一样，padding 位置 load 的是 0，`h * h` 也是 0，不影响求和。

### host 端：为什么是 view 不是 reshape

[[here:solutions/f3_fused_norm_act.py@L42-L52|fused_add_rmsnorm（42–52 行）]]

```python
x2, r2 = x.view(-1, N), residual.view(-1, N)     # view, not reshape: we must write in place
```

`reshape` 在张量不能直接变形时会**悄悄拷贝**一份。kernel 写进拷贝里，原来的 x 和 residual 根本没变，函数返回 None，调用方什么都拿不到。`view` 在不能变形时直接报错，把问题暴露出来。原地修改的函数一定要保证写的就是调用方传进来的那块内存。

::: callout analogy 改合同要改原件
reshape 像是“如果原件格式不对，就复印一份再改”。可这份合同的作用是让你在原件上签字（原地修改），你在复印件上签了，原件还是空白的，而且没人告诉你。view 像是“格式不对就拒绝办理”，至少你会知道出了问题。
:::

## ④ silu_and_mul：SwiGLU 的门控

[[here:solutions/f3_fused_norm_act.py@L31-L39|silu_and_mul_kernel（31–39 行）]] · host 端 [[here:solutions/f3_fused_norm_act.py@L55-L63|55–63 行]]

SwiGLU MLP 是 `down( silu(gate(x)) * up(x) )`。gate 和 up 两个投影通常合成**一个** GEMM（gate_up_proj），输出 `[T, 2I]`：前一半是 gate，后一半是 up。

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 90px repeat(8, minmax(44px, 1fr)) 30px 90px repeat(4, minmax(44px, 1fr)); min-width: 760px">
<div class="lbl">x[row]</div><div class="c0">g0</div><div class="c0">g1</div><div class="c0">g2</div><div class="c0">g3</div><div class="c1">u0</div><div class="c1">u1</div><div class="c1">u2</div><div class="c1">u3</div><div class="lbl">→</div><div class="lbl">out[row]</div><div class="c2">o0</div><div class="c2">o1</div><div class="c2">o2</div><div class="c2">o3</div>
</div></div>
<p class="legend">d = 4 的例子：o_i = silu(g_i) · u_i。gate 在 offs，up 在 d + offs，所以一次读两段，各 d 个数，写一段 d 个数。</p>
:::

```python
row  = tl.program_id(0).to(tl.int64)
offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)      # 二维 grid：行 × 列块
gate = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
up   = tl.load(x_ptr + row * x_stride + d + offs, mask=mask, other=0.0).to(tl.float32)
out  = gate * tl.sigmoid(gate) * up                        # silu(g) = g · σ(g)
```

- **二维 grid**：`(T, cdiv(d, BLOCK))`，`BLOCK = min(1024, next_power_of_2(d))`。LLaMA-3 8B 的 I = 14336，一行的输出就有 14336 个数，一个 program 放不下，所以每行再切成 14 个列块，每个 program 处理 1024 个。和 T2、T3 不同，这里没有行内归约，所以行可以随便切。
- **silu(g) = g · σ(g)**，也叫 swish。用 `tl.sigmoid` 和 T1 的 tanh 用 sigmoid 表示是同一个原因。
- 输出是新张量 `[T, I]`，不是原地。

## ⑤ 对照真实实现

**FlashInfer**

- Python：[[flashinfer:flashinfer/norm/__init__.py@L1253|fused_add_rmsnorm]] · [[flashinfer:flashinfer/activation.py@L91|silu_and_mul]]
- CUDA：[[flashinfer:include/flashinfer/norm.cuh@L416|FusedAddRMSNormKernel]]，launcher [[flashinfer:include/flashinfer/norm.cuh@L509-L512|FusedAddRMSNorm]] 里的 `vec_size = gcd(16 / sizeof(T), d)`：每个线程一次读 16 字节（bf16 是 8 个数），d 不是 8 的倍数时退到能整除的最大值。行内归约是 C2 的两层结构（[[flashinfer:include/flashinfer/norm.cuh@L462-L477|warp shuffle + shared memory]]）。
- 激活：[[flashinfer:include/flashinfer/activation.cuh@L33-L47|act_and_mul_kernel]]，同一个模板通过函数指针参数支持 silu、gelu 等不同的激活。
- 题目页提到的 `griddepcontrol`（PDL）：[[flashinfer:include/flashinfer/norm.cuh@L79|griddepcontrol.wait]] 和 [[flashinfer:include/flashinfer/norm.cuh@L135|launch_dependents]]，让下一个 kernel 在这个 kernel 还没完全结束时就开始启动。

**vLLM**：[[vllm:csrc/layernorm_kernels.cu@L181-L196|fused_add_rms_norm_kernel]]（`z += residual[...]; residual[...] = z;`），[[vllm:csrc/activation_kernels.cu@L110|act_and_mul_kernel]]、[[vllm:csrc/activation_kernels.cu@L165|silu_kernel]]。

## ⑥ 常见坑与自测

- **用 reshape**：非连续输入时静默失效（见第 ③ 节）。
- **rstd 用了存回 residual 之后再读回来的值**：结果和 FlashInfer 差最后一位，严格对拍时会失败。
- **忘了写回 residual**：x 是对的，但下一层用的残差错了，几层之后结果完全偏掉。
- **silu_and_mul 把 gate 和 up 弄反**：`silu(up) * gate` 和 `silu(gate) * up` 不一样。约定是前一半过 silu。

::: answer 自测 1：T = 16，H = 4096，bf16。fused_add_rmsnorm 一共读写多少字节？按 1.5 TB/s 要多久？实测为什么是 6 us？
4 × 16 × 4096 × 2 字节 = 512 KB，按 1.5 TB/s 约 0.35 us。实测 6 us 是 kernel launch 和调度的固定开销，数据量太小，带宽根本用不上。这就是 decode 阶段要靠融合减少 kernel 数的原因。
:::

::: answer 自测 2：I = 14336，T = 4096，silu_and_mul 的 grid 是多少？每个 program 读写多少字节（bf16）？
grid = (4096, cdiv(14336, 1024)) = (4096, 14)，共 57344 个 program。每个 program 读 2 × 1024 个、写 1024 个 bf16，共 6 KB。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/f3_fused_norm_act.py|solutions/f3_fused_norm_act.py]] · [[here:mine/f3_fused_norm_act.py|mine/f3_fused_norm_act.py]] · [[here:tests/test_f3_fused_norm_act.py|tests/test_f3_fused_norm_act.py]] · [[here:content/f3.md|content/f3.md]]

**FlashInfer**：[[flashinfer:flashinfer/norm/__init__.py@L1253|fused_add_rmsnorm]] · [[flashinfer:include/flashinfer/norm.cuh@L416|FusedAddRMSNormKernel]] · [[flashinfer:flashinfer/activation.py@L91|silu_and_mul]] · [[flashinfer:include/flashinfer/activation.cuh@L33|act_and_mul_kernel]] · [[flashinfer:flashinfer/triton/kernels/activation.py@L8|Triton 版 silu_and_mul]]

**vLLM**：[[vllm:csrc/layernorm_kernels.cu@L181|fused_add_rms_norm_kernel]] · [[vllm:csrc/activation_kernels.cu@L110|act_and_mul_kernel]]

**论文**：[GLU Variants Improve Transformer（SwiGLU）](https://arxiv.org/abs/2002.05202) · [RMSNorm](https://arxiv.org/abs/1910.07467)
