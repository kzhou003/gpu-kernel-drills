title: T1 讲解：Triton 的六个基本动作
lede: 用 Triton 重写 C1 的融合激活。逐行读参考解：program_id、tl.arange、mask、指针加法、dtype 转换、grid 和 constexpr，再看编译器把“一块数据”拆给线程时实际做了什么，以及同一个 kernel 怎么在 CPU 上跑。

::: callout info 怎么读这一页
对照 [[here:solutions/t1_add_gelu.py|t1_add_gelu.py]] 读。如果你刚做完 C1，第 ① 节的对照表能帮你把 CUDA 的概念一一换成 Triton 的概念。
:::

## ① 先看全貌：从“一个线程”到“一块数据”

C1 里你写的是**一个线程**的代码：我是第几号，我处理哪个元素。Triton 里你写的是**一个 program** 的代码：我处理哪一**块**元素。program 相当于 CUDA 的 block，但 block 里的线程你看不到，由编译器负责。

| 做什么 | C1（CUDA） | T1（Triton） |
|---|---|---|
| 我是谁 | `blockIdx.x * blockDim.x + threadIdx.x`：一个下标 | `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`：一个**下标向量** |
| 越界 | `if (i < n)` 分支 | `mask = offs < n`，传给 load / store |
| 读 | `x[i]` | `tl.load(x_ptr + offs, mask=mask)` |
| 写 | `y[i] = v` | `tl.store(out_ptr + offs, v, mask=mask)` |
| 启动 | `kernel<<<blocks, threads>>>(...)` | `kernel[grid](...)` |
| 每个 block 几个线程 | 你定（256） | 编译器定，`num_warps × 32`，默认 4 × 32 = 128 |
| 编译 | 提前用 nvcc 编译 | 第一次调用时 JIT 编译，之后缓存 |

::: flow 一次 add_gelu 调用发生了什么
cfg cw=245 nw=175
n py 0 0 "add_gelu(x, y)" "Python host 函数" k=acc
n jit 1 0 "@triton.jit" "首次调用：编译并缓存" k=vio
n grid 2 0 "grid = (cdiv(n, 1024),)" "program 个数" k=acc w=1.1
n p 2 1.3 "每个 program" "处理 1024 个元素" k=pass
n th 1 1.3 "编译器拆给线程" "128 线程，每人 8 个" k=teal
e py -> jit
e jit -> grid
e grid -> p "启动"
e p -> th "layout"
cap: 你只写到“每个 program 处理 1024 个元素”这一层。1024 个元素怎么分给 4 个 warp、每个线程拿哪几个、用不用 128 位向量化读取，都是编译器根据 layout 决定的（G1 会专门讲 layout）。
:::

::: callout analogy 下单，而不是派活
CUDA 像是你亲自给每个快递员（线程）派活：“你送 3 号楼 201”。Triton 像是你在系统里给每个站点（program）下一张单：“这个站点负责 0–1023 号包裹”，站点内部怎么分给快递员由系统排。你能控制的是单子怎么拆（BLOCK_SIZE）和每个站点配几个人（num_warps），而不是某个人送哪一件。
:::

## ② kernel 逐行讲

[[here:solutions/t1_add_gelu.py@L16-L34|add_gelu_kernel（16–34 行）]]

### 签名：指针和 constexpr

```python
@triton.jit
def add_gelu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
```

- `@triton.jit`：这个函数不会被 Python 执行，而是被 Triton 解析、编译成 GPU 代码。
- host 端传进来的 torch 张量（`x`、`y`、`out`）在 kernel 里变成**指针**（`x_ptr`），指向张量的第一个元素，类型是 `pointer<fp32>` 之类。普通整数（`n`）就是标量。
- `BLOCK_SIZE: tl.constexpr`：**编译期常量**。每一个不同的值都会编译出一个独立的 kernel。`tl.arange` 的长度必须是编译期已知的 2 的幂，所以 BLOCK_SIZE 必须是 constexpr。

::: callout analogy 印刷的版次
constexpr 参数像书的“版次”：BLOCK_SIZE = 1024 印一版，改成 2048 就得重新排版再印一版（重新编译）。版次一旦印好就存起来（缓存），下次同样的版次直接拿。普通参数（n_elements）则像读者在书上填的空：同一版书，每次填的内容可以不同，不需要重印。
:::

### 我负责哪一块

[[here:solutions/t1_add_gelu.py@L18-L20|18–20 行]]

```python
pid  = tl.program_id(axis=0)                           # 第几个 program：0, 1, 2, ...
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)     # [BLOCK_SIZE] 个下标
mask = offs < n_elements                               # [BLOCK_SIZE] 个 True/False
```

`tl.arange(0, 1024)` 是一个长度 1024 的向量 `[0, 1, …, 1023]`，加上一个标量之后还是向量。所以 `offs` 和 `mask` 都是**一整块**。

n = 2500，BLOCK_SIZE = 1024 时：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 1024fr 1024fr 452fr 572fr; min-width: 520px">
<div class="c0">pid 0<small>offs 0 … 1023 · 全部 True</small></div>
<div class="c1">pid 1<small>offs 1024 … 2047 · 全部 True</small></div>
<div class="c2">pid 2<small>2048 … 2499</small></div>
<div class="cx">mask = False<small>2500 … 3071</small></div>
</div></div>
<p class="legend">grid = cdiv(2500, 1024) = 3。最后一个 program 的 1024 个下标里只有前 452 个是 True，后 572 个的 load 和 store 都被屏蔽。</p>
:::

和 C1 的 `if (i < n)` 是同一件事，区别是这里不写分支，而是把 mask 交给 load 和 store。

### 读、算、写

[[here:solutions/t1_add_gelu.py@L23-L34|23–34 行]]

```python
x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
z = x + y
inner = 0.7978845608028654 * (z + 0.044715 * z * z * z)
t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0          # = tanh(inner)
out = 0.5 * z * (1.0 + t)
tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)
```

- **`x_ptr + offs`**：指针加一个整数向量，得到一个**指针向量**（1024 个地址）。`tl.load` 一次把这 1024 个地址的值读成一个向量。
- **`mask=..., other=0.0`**：mask 为 False 的位置不访问内存，值填 `other`。这里填什么都无所谓（反正不会被存回去），但读越界本身必须避免。
- **`.to(tl.float32)`**：输入可能是 fp16 / bf16，中间计算统一用 fp32。fp16 的最大值是 65504，|z| 超过约 40 时 z³ 就溢出；bf16 的尾数只有 8 位，z³ 和小常数相加时精度损失明显。
- **`tl.sigmoid` 算 tanh**：恒等式 tanh(u) = 2σ(2u) − 1。注释里说明了原因：`tl.sigmoid` 在所有后端都有（包括 CPU 解释器），用它写更省心。
- **`out.to(out_ptr.dtype.element_ty)`**：`out_ptr.dtype` 是指针类型，`.element_ty` 是它指向的元素类型。存回去之前转回输入的 dtype。

所有运算（`+`、`*`、`tl.sigmoid`）都是对整个向量逐元素进行的，和 NumPy 一样。

## ③ host 端：grid 和启动语法

[[here:solutions/t1_add_gelu.py@L37-L49|add_gelu（37–49 行）]]

```python
x, y = x.contiguous(), y.contiguous()     # kernel 把内存当一维数组读，必须连续
out = torch.empty_like(x)
grid = (triton.cdiv(n, block_size),)       # 一维 grid，program 个数
add_gelu_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
```

- `triton.cdiv(a, b)` 就是 C1 里的 `(a + b - 1) / b`，向上取整。
- `kernel[grid](args...)`：方括号里是 grid（一个元组，最多三维），圆括号里是 kernel 参数。constexpr 参数用关键字传。
- grid 也可以写成函数：`grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)`。Triton 会把 constexpr 参数组成的字典 `meta` 传进去。这在 T4 用 autotune 时是必须的，因为那时 BLOCK_SIZE 由 autotuner 选，你在调用处还不知道它是多少。
- `x.contiguous()`：已经连续时不拷贝，直接返回自己；不连续时拷贝一份连续的。这和 C1 里 `is_contiguous()` 检查的目的一样，只是这里选择了“帮用户转”而不是“报错”。

## ④ 编译器替你做了什么

你写的是 1024 长的向量运算，GPU 上实际执行的是 `num_warps × 32` 个线程（默认 4 × 32 = 128）。编译器把 1024 个元素按 **layout** 分给这 128 个线程，每个线程 8 个。fp32 的典型分法是：每个线程一次拿**连续 4 个**（正好一次 128 位读取），128 个线程一轮覆盖 512 个元素，再绕一圈覆盖后 512 个：

::: flow BLOCK_SIZE = 1024 在硬件上的样子（num_warps = 4）
cfg cw=230 nw=175
n v 0 0.65 "一个 program" "1024 个元素的向量" k=acc
n w0 1 0 "warp 0" "0–127 和 512–639" k=teal
n w3 1 1.3 "warp 3" "384–511 和 896–1023" k=teal
n t0 2 0 "lane 0" "0–3 和 512–515" k=pass
n t1 2 1.3 "lane 1" "4–7 和 516–519" k=pass
e v -> w0
e v -> w3
e w0 -> t0
e w0 -> t1
cap: 这是 fp32 下的典型分法（G1 里叫 BlockedLayout，sizePerThread = 4）。实际 layout 由编译器根据 BLOCK_SIZE、num_warps、数据类型和指针对齐决定。每个线程拿连续 4 个 fp32，就能用 128 位的向量化读写；同一轮里 32 个 lane 读连续的 512 字节，合并访存也自然满足。
:::

这就是 Triton 省掉的工作：在 CUDA 里想做到这一点，你得自己写 `float4` 读取、自己算每个线程的偏移。代价是你看不到也很难精确控制这一层，G1 会讲怎么读懂它。

### 在 CPU 上跑：TRITON_INTERPRET=1

本仓库没有 GPU 时自动设 `TRITON_INTERPRET=1`（见 [[here:common/env.py|common/env.py]]）。这时 `@triton.jit` 不编译，而是用 NumPy **逐个 program** 解释执行：`tl.arange` 变成 NumPy 数组，`tl.load` 变成数组索引。所以正确性可以在笔记本上测，还能在 kernel 里直接 `print`。实现见 [[triton:python/triton/runtime/interpreter.py@L1674|InterpretedFunction]]。性能数字只有在 GPU 上才有意义。

## ⑤ 对照真实代码

- **官方 tutorial 01**，几乎逐行相同：[[triton:python/tutorials/01-vector-add.py@L30|add_kernel]]，其中 [[triton:python/tutorials/01-vector-add.py@L39-L54|program_id → arange → mask → load → store]]，以及 [[triton:python/tutorials/01-vector-add.py@L70|lambda grid]]。
- **FlashInfer 的 Triton silu_and_mul**：同样的结构，多了二维 grid（每行一个 program 维度，行内再分块）和可选的量化 scale：[[flashinfer:flashinfer/triton/kernels/activation.py@L8|silu_and_mul_kernel]]。注意它在 [[flashinfer:flashinfer/triton/kernels/activation.py@L41|41 行]]把 program_id 转成 int64，防止行号乘 stride 溢出；以及 [[flashinfer:flashinfer/triton/kernels/activation.py@L50-L58|先转 fp32 再算]]，和这题一样。F3 会用到这个算子。

## ⑥ 常见坑与自测

- **BLOCK_SIZE 不是 2 的幂**：`tl.arange` 要求长度是 2 的幂，1000 会直接报编译错误。
- **忘了 mask**：GPU 上越界读写可能不报错但结果错，或者报非法地址；解释器模式下通常会直接报 IndexError。
- **忘了转回 dtype**：fp32 的 out 存进 fp16 的指针，Triton 会报类型不匹配，或者隐式转换（取决于版本）。
- **在 kernel 里用 Python 的 `if` 判断一个张量**：`if mask:` 不行，mask 是向量。用 `tl.where(cond, a, b)` 做逐元素选择。

::: answer 自测 1：n = 1,000,000，block_size = 1024，一共几个 program？最后一个 program 有几个有效元素？
cdiv(1,000,000, 1024) = 977。前 976 个 program 覆盖 976 × 1024 = 999,424 个元素，最后一个有 1,000,000 − 999,424 = 576 个有效元素。
:::

::: answer 自测 2：block_size 从 1024 改成 4096，program 数和每个线程处理的元素数怎么变？
program 数变成 1/4（245 个）。num_warps 不变时，每个线程从 8 个元素变成 32 个，寄存器占用变大。block 太大时一般要同时调大 num_warps，题目页“测完之后”建议的扫参就是在看这个权衡。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/t1_add_gelu.py|solutions/t1_add_gelu.py]] · [[here:mine/t1_add_gelu.py|mine/t1_add_gelu.py]] · [[here:tests/test_t1_add_gelu.py|tests/test_t1_add_gelu.py]] · [[here:content/t1.md|content/t1.md]] · [[here:content/basics_triton.md|content/basics_triton.md（Triton 基础）]]

**Triton**：[[triton:python/tutorials/01-vector-add.py|tutorial 01-vector-add]] · [[triton:python/triton/runtime/interpreter.py@L1674|解释器]] · API：[tl.program_id](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html) · [tl.arange](https://triton-lang.org/main/python-api/generated/triton.language.arange.html) · [tl.load](https://triton-lang.org/main/python-api/generated/triton.language.load.html) · [tl.store](https://triton-lang.org/main/python-api/generated/triton.language.store.html) · [tl.sigmoid](https://triton-lang.org/main/python-api/generated/triton.language.sigmoid.html) · [triton.language 全部 API](https://triton-lang.org/main/python-api/triton.language.html)

**FlashInfer**：[[flashinfer:flashinfer/triton/kernels/activation.py@L8|silu_and_mul_kernel]] · [[flashinfer:flashinfer/triton/activation.py|host 端 wrapper]]
