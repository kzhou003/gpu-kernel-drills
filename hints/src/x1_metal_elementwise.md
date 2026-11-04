title: X1 讲解：MLX 的第一个自定义 Metal kernel
lede: 用 mx.fast.metal_kernel 重写 C1 / T1 的 add + GELU。这一页把 MLX 实际生成的 Metal 源码贴出来（在本机用 verbose=True 抓的），逐行对照你写的函数体，讲清楚签名是怎么来的、模板参数 T 怎么变成 half、Metal 的 grid 为什么是“线程总数”而不是 block 数，以及 4 元素版本和 MLX 自己的 unary kernel 是什么关系。

::: callout info 怎么读这一页
对照 [[here:solutions/x1_metal_elementwise.py|x1_metal_elementwise.py]] 读。如果你做过 C1 和 T1，这一题的 kernel 本身没有新东西，新的是 MLX 的“只写函数体”这种接口，以及 Metal 和 CUDA 在 launch 语义上的差别。
:::

## ① 先看全貌：你写函数体，MLX 写其余部分

::: flow 一次 metal_kernel 调用
cfg cw=260 nw=170
n src 0 0 "source + header" "你写的函数体" k=acc
n mk 1 0 "mx.fast.metal_kernel" "input_names、output_names" k=vio w=1.05
n sig 2 0 "生成完整的 Metal 源码" "签名 + 模板实例化" k=warn w=1.05
n run 2 1.3 "调用时" "template、grid、threadgroup" k=pass w=1.1
n disp 1 1.3 "dispatchThreads" "grid = 线程总数" k=pass
e src -> mk
e mk -> sig "首次调用"
e sig -> run
e run -> disp
cap: kernel 在第一次以某组模板参数调用时编译，之后缓存。和 T1 的 @triton.jit 一样是 JIT，和 C1 的 load_inline 也类似，只是这里连函数签名都不用写。
:::

在本机（MLX 0.32.4）用 `verbose=True` 调用 fp16 版本，MLX 打印出的完整源码是：

```cpp
inline float gelu_tanh(float z) {                       // ← 你的 header
    return 0.5f * z * (1.0f + metal::precise::tanh(0.7978845608028654f * (z + 0.044715f * z * z * z)));
}
template <typename T>
[[kernel]] void custom_kernel_add_gelu__float16_t_float16_t_float16_t_int32_ts_float16_t(
  const device float16_t* x [[buffer(0)]],              // ← input_names[0]
  const device float16_t* y [[buffer(1)]],              // ← input_names[1]
  const constant int32_t& n [[buffer(2)]],              // ← Python int 变成 constant 标量
  device float16_t* out [[buffer(3)]],                  // ← output_names[0]
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {   // ← 因为函数体里用到了它
    uint i = thread_position_in_grid.x;                 // ← 你的 source
    if (i < n) {
        out[i] = T(gelu_tanh(float(x[i]) + float(y[i])));
    }
}
template [[host_name("custom_kernel_add_gelu__…")]] [[kernel]]
decltype(custom_kernel_add_gelu__…<float16_t>) custom_kernel_add_gelu__…<float16_t>;   // ← 用 T = float16_t 实例化
```

逐项说明：

| 生成的部分 | 从哪来 |
|---|---|
| `const device float16_t* x [[buffer(0)]]` | `input_names=["x", "y", "n"]` 的前两个是数组，变成只读的 device 指针；`[[buffer(i)]]` 是 Metal 的参数槽位 |
| `const constant int32_t& n` | 第三个输入是 Python 的 int，MLX 把它当作标量，放进 `constant` 地址空间 |
| `device float16_t* out` | `output_names=["out"]`，可写的 device 指针 |
| `uint3 thread_position_in_grid [[…]]` | MLX 扫描函数体，发现用到了这个 Metal 属性，才把它加进签名（[[mlx:mlx/backend/common/custom_kernel.cpp@L255-L279|支持的属性列表]]） |
| 函数名里的 `float16_t_float16_t_…` | 由输入、输出的 dtype 拼出来，不同 dtype 是不同的 kernel |
| 最后的 `template [[host_name(…)]]` | 用 `template=[("T", x.dtype)]` 给的 T 显式实例化 |

生成签名的代码在 [[mlx:mlx/backend/common/custom_kernel.cpp@L53|write_signature]]，[[mlx:mlx/backend/common/custom_kernel.cpp@L92|kernel void 的开头]] 和 [[mlx:mlx/backend/common/custom_kernel.cpp@L350-L352|host_name 实例化]] 都是在这里拼出来的字符串。

::: callout analogy 填申请表
CUDA 像让你从一张白纸开始写申请书：抬头、称呼、正文、落款全部自己来（`__global__ void k(const half* x, ...)`）。MLX 的 metal_kernel 像一张印好的申请表：抬头和落款（签名、参数槽位、模板实例化）已经印好了，你只需要在“正文”那一栏填内容（source），并在表头勾选“申请人姓名”“附件清单”（input_names、output_names）。你在正文里提到了“位置”（thread_position_in_grid），表格就自动多出一栏让系统填。
:::

## ② CUDA ↔ Metal 对照

| CUDA（C1） | Metal / MLX（X1） |
|---|---|
| block / thread | threadgroup / thread |
| warp（32） | simdgroup（Apple GPU 上也是 32） |
| `blockIdx.x * blockDim.x + threadIdx.x` | `thread_position_in_grid.x`：直接就是全局编号 |
| `__shared__` | `threadgroup` 内存（X2 会用） |
| `const float* __restrict__ x` | `const device T* x` |
| 显存，要拷贝 | **统一内存**：CPU 和 GPU 共用同一块，不需要拷贝 |
| nvcc / Nsight | Metal Shading Language（C++14 方言）/ Xcode Instruments |

## ③ grid 的语义：线程总数，不是 block 数

[[here:solutions/x1_metal_elementwise.py@L43-L49|add_gelu（43–49 行）]]

```python
_kernel(inputs=[x, y, n], template=[("T", x.dtype)],
        grid=(max(n, 1), 1, 1), threadgroup=(threadgroup, 1, 1), ...)
```

这是从 CUDA 转过来最容易错的地方。MLX 的 `grid` 对应 Metal 的 `dispatchThreads`（[[mlx:mlx/backend/metal/custom_kernel.cpp@L99|compute_encoder.dispatch_threads]]），它的单位是**线程**：

| | CUDA（C1） | Metal（X1） |
|---|---|---|
| n = 1000，每组 256 | `<<<4, 256>>>`：你自己算 ⌈1000/256⌉ = 4 个 block | `grid=(1000,1,1), threadgroup=(256,1,1)`：你给线程总数 |
| 实际线程数 | 4 × 256 = 1024，多出 24 个 | 1000：最后一组只有 232 个线程 |
| 越界检查 | 必须 `if (i < n)` | 一个元素一个线程时，严格说不需要 |

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 256fr 256fr 256fr 232fr 24fr; min-width: 520px">
<div class="c0">group 0<small>256</small></div><div class="c1">group 1<small>256</small></div><div class="c2">group 2<small>256</small></div><div class="c3">group 3<small>232 个线程</small></div><div class="lbl" style="text-align:left">不存在</div>
</div></div>
<p class="legend">和 C1 那张图对比：CUDA 的最后一个 block 有 24 个空转线程，Metal 的最后一个 threadgroup 直接变小（非均匀 threadgroup，Apple GPU 支持）。</p>
:::

参考解仍然写了 `if (i < n)`：这是防御性的写法，而且 4 元素版本里确实需要（下一节）。MLX 的文档也专门提醒了这一点：[[mlx:docs/src/dev/custom_metal_kernels.rst@L89-L91|grid 和 threadgroup 是 dispatchThreads 的参数]]。

::: callout analogy 订车 vs 报人数
CUDA 让你直接订车：“要 4 辆 256 座的大巴”，多出来的座位空着，你自己确保空座位上的人什么都不做。Metal 让你报人数：“一共 1000 人，每辆车最多 256 座”，调度员自己算出要 4 辆车，最后一辆只派 232 座的。你报错了单位（把车数当人数报），就只会来 4 个人。
:::

## ④ 函数体：模板、fp32 计算、header

[[here:solutions/x1_metal_elementwise.py@L13-L26|13–26 行]]

```cpp
uint i = thread_position_in_grid.x;
if (i < n) {
    out[i] = T(gelu_tanh(float(x[i]) + float(y[i])));
}
```

- **T 是模板参数**：同一份 source 通过 `template=[("T", mx.float16)]` 实例化出 fp16、bf16、fp32 三个版本。
- **在 fp32 里算**：和 T1、C1 一样，`float(x[i])` 先升精度，算完再 `T(...)` 降回去。
- **header** 放辅助函数。`metal::precise::tanh` 是精确版本，`metal::fast::tanh` 是快速近似版本，后者更快但误差更大。这和 CUDA 的 `tanhf` 与 `__tanhf` 是同一类选择。
- **ensure_row_contiguous**（[[here:solutions/x1_metal_elementwise.py@L46|46 行]]注释）：默认为 True，输入不连续时 MLX 先复制一份连续的。这和 T1 的 `.contiguous()` 是同一个目的。

## ⑤ 4 元素版本

[[here:solutions/x1_metal_elementwise.py@L28-L35|_SRC_VEC（28–35 行）]] · [[here:solutions/x1_metal_elementwise.py@L52-L58|add_gelu_vec（52–58 行）]]

```cpp
uint base = thread_position_in_grid.x * N_PER;
for (uint j = 0; j < N_PER; ++j) {
    uint i = base + j;
    if (i < n) out[i] = T(...);
}
```

每个线程处理连续的 N_PER 个元素，线程数变成 ⌈n / N_PER⌉。这里 `if (i < n)` 是**必须**的：n = 1000、N_PER = 4 时有 250 个线程，正好；但 n = 1001 时有 251 个线程，最后一个线程的 j = 1、2、3 都越界。

`N_PER` 也是模板参数（`template=[("T", …), ("N_PER", n_per)]`），所以循环次数是编译期常量，编译器可以展开。

MLX 自己的逐元素 kernel 就是这个结构：[[mlx:mlx/backend/metal/kernels/unary.h@L3-L19|unary_v]]，每个线程处理 `WorkPerThread<T>::n` 个元素，并且专门处理了最后一段不满 N 的情况。

题目页的实测（M2，fp16，n = 2²⁴）：单元素版 42 GB/s，4 元素版 43 GB/s，MLX 自带操作 46 GB/s，同一时刻 `x + 1` 实测 59 GB/s。差别不大，因为它们都被带宽限制住了。memory-bound kernel 要和**这台机器实测的带宽**比，而不是和标称值比。

## ⑥ 常见坑与自测

- **把 grid 当 block 数**：`grid=(cdiv(n, 256), 1, 1)` 只会启动 ⌈n/256⌉ 个线程，结果大部分元素没被算。
- **标量用 Python float 传 int 的地方**：n 是 int，会变成 `int32_t`；如果传了 float，类型就对不上。
- **忘了 template**：source 里用了 T，调用时没给 `template=[("T", ...)]`，编译失败。
- **在 fp16 里算 z³**：和 T1 一样，|z| > 40 左右就溢出。

::: answer 自测 1：n = 2²⁴，threadgroup = 256，单元素版本启动多少个线程？多少个 threadgroup？
grid = 2²⁴ = 16,777,216 个线程，2²⁴ / 256 = 65,536 个 threadgroup，正好整除，没有不满的组。
:::

::: answer 自测 2：用 CUDA 的思路写成 grid=((n + 255) // 256, 1, 1)，n = 1000 时会发生什么？
只启动 4 个线程（grid 是线程总数），算出 out[0..3]，其余 996 个元素保持未初始化。测试会失败，但不会报错。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/x1_metal_elementwise.py|solutions/x1_metal_elementwise.py]] · [[here:mine/x1_metal_elementwise.py|mine/x1_metal_elementwise.py]] · [[here:tests/test_x1_metal_elementwise.py|tests/test_x1_metal_elementwise.py]] · [[here:content/x1.md|content/x1.md]] · 对照：[C1](c1_elementwise.html) · [T1](t1_add_gelu.html)

**MLX**：[[mlx:docs/src/dev/custom_metal_kernels.rst|custom_metal_kernels.rst]] · [[mlx:mlx/backend/common/custom_kernel.cpp@L53|write_signature]] · [[mlx:mlx/backend/metal/custom_kernel.cpp@L99|dispatch_threads]] · [[mlx:mlx/backend/metal/kernels/unary.h@L3|unary_v]] · [[mlx:docs/src/dev/metal_debugger.rst|Metal debugger]]

**文档**：[MLX · Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) · [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
