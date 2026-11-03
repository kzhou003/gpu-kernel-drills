lede: Triton 的心智模型是“一个 program 处理一块数据”：你写 block 级的向量代码，编译器负责把它拆给线程、安排 shared memory 和流水线。
# Triton 语法速查（附 Gluon 速览）

## 与 CUDA 的对应

| | CUDA | Triton |
|---|---|---|
| 编程单位 | 一个**线程**处理一个元素 | 一个 **program** 处理一个 **block**（向量 / 矩阵块） |
| 编号 | `blockIdx` + `threadIdx` | `tl.program_id(axis)`；没有 threadIdx |
| 一块数据的偏移 | 自己算 | `pid * BLOCK + tl.arange(0, BLOCK)` |
| 越界 | `if (i < n)` | `mask=offs < n` 传给 load / store |
| shared memory、同步、合并访存、向量化 | 手写 | 编译器自动处理 |
| warp 数 | `blockDim` | launch 参数 `num_warps` |
| 矩阵乘 | WMMA / mma PTX | `tl.dot` |

## 最小的 kernel

```python
import torch, triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)      # [BLOCK] 个 int32 偏移
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)                 # program 数；也可以是 lambda meta: (...)
    add_kernel[grid](x, y, out, n, BLOCK=1024)     # torch.Tensor 自动变成指针
    return out
```

## 规则与常用 API

- **`tl.constexpr`**：编译期常量。block 大小、head_dim、是否 causal 这类决定代码形状的参数都要标成 constexpr；每种取值编译出一个独立的 kernel。
- **block 形状必须是 2 的幂**：`tl.arange(0, BLOCK)` 的 BLOCK 必须是 2 的幂。行长 N 不是 2 的幂时，用 `triton.next_power_of_2(N)` 取上界，再用 mask 屏蔽多出的部分。
- **指针算术 + 广播**造二维块：`ptrs = base + rows[:, None] * stride + cols[None, :]` 得到 `[R, C]` 的指针块，`tl.load(ptrs, mask=...)` 读出 `[R, C]` 的值。
- **dtype**：`tl.load` 返回指针指向的类型；`.to(tl.float32)` 转换；存储前转回 `out_ptr.dtype.element_ty`。
- **masked load 的 `other`**：被屏蔽位置的值。做 max 时填 `-inf`，做 sum 时填 `0`。

| 类别 | API |
|---|---|
| 归约 | `tl.sum(x, axis)`、`tl.max`、`tl.min`、`tl.argmax` |
| 逐元素 | `tl.exp`、`tl.log`、`tl.sqrt`、`tl.sigmoid`、`tl.maximum`、`tl.where(cond, a, b)` |
| 构造 | `tl.zeros(shape, dtype)`、`tl.full(shape, value, dtype)`、`tl.arange` |
| 矩阵乘 | `tl.dot(a, b, acc)`：`[M,K] @ [K,N]`，M、N、K 都至少 16；fp16 / bf16 输入，fp32 累加 |
| 形状 | `x[:, None]`、`tl.trans(x)`、`tl.reshape`、`tl.broadcast_to` |
| 工具 | `tl.cdiv`、`triton.cdiv`、`triton.next_power_of_2`、`tl.static_assert`、`tl.device_print` |

## 性能相关的 launch 参数

- **`num_warps`**：每个 program 的 warp 数（默认 4）。block 越大，需要越多 warp 来分摊。
- **`num_stages`**：软件流水线深度。循环里的 `tl.load` 会被提前若干轮发出，经 shared memory 缓冲，计算和访存重叠。主要影响 GEMM / attention 这类带 K 循环的 kernel。
- **`@triton.autotune(configs=[triton.Config({...}, num_warps=.., num_stages=..)], key=[...])`**：对每个新的 `key`（比如 M、N、K）把所有 config 各跑一遍，缓存最快的那个。

## 调试

- **`TRITON_INTERPRET=1`**：在 CPU 上逐 program 解释执行，可以在 kernel 里 `print`、打断点。本仓库没有 GPU 时自动打开。
- `tl.device_print("x", x)`：GPU 上打印；`tl.static_print`：编译期打印 constexpr 和 shape。
- 看编译产物：`kernel[grid](...)` 返回的对象有 `.asm["ttir"]`、`["ttgir"]`、`["ptx"]`。`ttgir` 里能看到编译器给每个张量选的 layout，比如 `#blocked`、`#mma`、`#shared`，这正是 G1 要讲的东西。

## Gluon 速览

Gluon 是 Triton 下面一层的语言（`triton.experimental.gluon`）：语法和 Triton 很像，但**layout 由你显式指定**，shared memory 的分配、异步拷贝、warp specialization 也由你控制。Triton 编译器替你做的那些决定，在 Gluon 里由你自己写。

```python
# 示意：Gluon API 仍是 experimental，参数名以你所装版本的官方 tutorial 为准
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def copy_kernel(src, dst, n, BLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)   # 偏移张量带着 layout
    mask = offs < n
    gl.store(dst + offs, gl.load(src + offs, mask=mask), mask=mask)

layout = gl.BlockedLayout(size_per_thread=[4], threads_per_warp=[32], warps_per_cta=[4], order=[0])
copy_kernel[(triton.cdiv(n, 512),)](src, dst, n, BLOCK=512, layout=layout, num_warps=4)
```

`BlockedLayout(size_per_thread=[4], threads_per_warp=[32], warps_per_cta=[4])` 的意思是：每个线程拿 4 个连续元素（正好一次 16 字节的向量化 load），一个 warp 覆盖 128 个元素，4 个 warp 覆盖 512 个。**G1** 用纯 Python 把这个映射算清楚：哪个元素在哪个 warp、哪个 lane、第几个寄存器，以及它对合并访存和归约代价的影响。理解了这一层，读 `ttgir` 和写 Gluon 都会容易很多。

## 图解与类比

::: flow Triton 的编译管线
n src 0 0 "@triton.jit 函数" "Python AST" k=acc
n ttir 1 0 "TTIR" "块级张量运算" k=vio
n ttgir 2 0 "TTGIR" "+ layout · smem · 流水线" k=vio
n llir 3 0 "LLVM IR" "" k=vio
n ptx 3 1 "PTX → cubin" "ptxas" k=warn
n run 2 1 "launch" "grid × num_warps" k=pass
n interp 0 1 "TRITON_INTERPRET=1" "NumPy 逐 program 执行" k=ghost w=1.2
e src -> ttir
e ttir -> ttgir "编译器决定 layout"
e ttgir -> llir
e llir -> ptx
e ptx -> run
e src -> interp "CPU 调试" dash
cap: 你只写到 TTIR 这一层（块级运算）；TTGIR 里出现的 `#blocked`、`#mma`、`#shared` 就是编译器替你做的决定，G1 会教你读懂它们。
:::
::: flow 一个 program 怎么变成线程
n prog 0 0.5 "program pid" "offs = pid·B + arange(B)" k=acc w=1.25
n w0 1.6 0 "warp 0" "lane 0–31" k=vio
n w1 1.6 1 "warp 1 …" "num_warps 个" k=vio
n regs 2.8 0.5 "每线程若干寄存器" "sizePerThread" k=pass w=1.1
e prog -> w0 "按 layout 切分"
e prog -> w1
e w0 -> regs
e w1 -> regs
:::
::: callout analogy CUDA 写给工人，Triton 写给工头
CUDA 里你给每个工人（线程）写说明书：“你是第几号，你负责第几个格子”。Triton 里你只给工头（program）写：“把这一箱（block）处理完”，工头自己决定怎么分给手下（num_warps × 32 个线程）、谁拿几个、怎么搬得最快。
`mask` 就是最后一箱不满，空格子贴上“跳过”标签；`tl.constexpr` 是模具尺寸：换尺寸就要重新开模（重新编译）。
:::

<!-- wave2 -->
::: flow autotune：第一次调用时把每个 config 都试一遍
n call 0 0.5 "kernel[grid](M, N, K)" "" k=acc w=1.15
n key 1.35 0.5 "这个 key 见过吗？" "key = (M, N, K)" k=warn w=1.05
n try 2.6 0 "每个 config 跑一遍计时" "BLOCK · num_warps · num_stages" k=vio w=1.35
n best 4.1 0 "缓存最快的 config" "" k=teal
n run 2.6 1 "直接用缓存的 config" "" k=pass w=1.25
e call -> key
e key -> try "没见过"
e try -> best
e best:b -> run:r
e key -> run "见过"
:::
::: callout analogy num_stages 是提前备料
厨师做第 k 道菜时，帮工已经在洗第 k+1、k+2 道的菜（num_stages = 3）：计算和搬运同时进行。备得越多越不会断档，但每一份备料都要占一块料理台（shared memory），台子放满了，能同时开工的厨师（occupancy）就少了。
:::

## 源码与资料

**文档**：[Triton 文档首页](https://triton-lang.org/main/index.html) · [Tutorials 目录](https://triton-lang.org/main/getting-started/tutorials/index.html) · [triton.language API](https://triton-lang.org/main/python-api/triton.language.html)

**源码**：[[triton:python/tutorials/01-vector-add.py]] · [[triton:python/triton/runtime/interpreter.py|interpreter.py（TRITON_INTERPRET）]] · [[triton:python/triton/runtime/autotuner.py@Autotuner]] · [[triton:python/tutorials/gluon/01-intro.py|Gluon 01-intro]] · [[triton:python/tutorials/gluon/02-layouts.py|Gluon 02-layouts]] · [[triton:python/tutorials/gluon/|全部 Gluon tutorials]] · [[puzzles|Triton-Puzzles]]

