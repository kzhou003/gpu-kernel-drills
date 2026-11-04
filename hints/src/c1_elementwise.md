title: C1 答疑：逐行读懂 c1_elementwise.cu
lede: 把读 C1 参考解时卡住的 8 个问题讲透：TORCH_CHECK 和 at::kFloat、blockDim 从哪来、at/c10/torch、launch 配置、__device__ __forceinline__、grid-stride loop、block 数上限、bias[i % cols]。每节都有图、类比、算例和源码链接。

::: callout info 怎么读这一页
每一节对应一个你问过的问题，开头标出它在 [[here:solutions/c1_elementwise.cu|c1_elementwise.cu]] 里的行号（点灰色小标签可以跳到 GitHub 上的那一行）。图里的节点可以悬停，相连的边会高亮。PyTorch 的链接都钉在固定 commit 上，行号不会漂移。
:::

## ① TORCH_CHECK 在干什么？at::kFloat 从哪来？

<p class="q">你问的是 <b>第 13–16 行</b>：<code>TORCH_CHECK((t).scalar_type() == at::kFloat, #t " must be float32")</code> · [[here:solutions/c1_elementwise.cu@L13-L16|c1_elementwise.cu:13–16]]</p>

一句话：**在 kernel 启动之前，确认输入张量满足 kernel 的假设；不满足就抛异常，Python 那边看到一个写清楚原因的 `RuntimeError`。**

这三行在文件里是一个宏的一部分，三项检查一起做：

```cpp
#define CHECK_F32_CUDA(t)                                                   \
  TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor");                  \
  TORCH_CHECK((t).is_contiguous(), #t " must be contiguous");               \
  TORCH_CHECK((t).scalar_type() == at::kFloat, #t " must be float32")
```

| 检查 | 为什么 kernel 需要它 | 不检查会怎样 |
|---|---|---|
| `is_cuda()` | kernel 只能读显存 | 把 CPU 内存地址交给 GPU：非法地址错误 |
| `is_contiguous()` | kernel 用 `x[i]` 按连续下标读 | 转置后的视图会被按错误顺序读，结果错但不报错 |
| `scalar_type() == at::kFloat` | kernel 写的是 `data_ptr<float>()`，按 4 字节 float 解读 | fp16 张量的字节被当成 float 读：**静默的垃圾值** |

::: flow TORCH_CHECK 失败时，错误怎么一路回到 Python
cfg cw=190 nw=160
n py 0 0 "Python" "ext.vector_add(a, b)" k=acc
n cpp 1 0 "vector_add(a, b)" "C++ host 函数" k=acc
n chk 2 0 "TORCH_CHECK" "条件为假？" k=warn
n ok 3 0 "继续往下执行" "检查通过" k=pass
n fail 2 1.3 "torchCheckFail" "拼消息，抛 c10::Error" k=todo w=1.1
n pyb 1 1.3 "pybind11" "翻译 C++ 异常" k=vio
n err 0 1.3 "RuntimeError" "a must be float32" k=todo
e py -> cpp
e cpp -> chk
e chk -> ok "否" k=pass
e chk -> fail "是" k=todo
e fail -> pyb k=todo anim
e pyb -> err k=todo anim
cap: TORCH_CHECK 和 assert 不同：release 构建里照样生效，而且错误能完整地传回 Python，消息里还带着 C++ 的文件名和行号。
:::

#### 拆开这一行

- **`TORCH_CHECK(cond, msg...)`**：PyTorch 的断言宏。展开后大致是 `if (!(cond)) torchCheckFail(__func__, __FILE__, __LINE__, msg)`，见 [[pytorch:c10/util/Exception.h@L627|Exception.h · TORCH_CHECK]]。
- **`(t).scalar_type()`**：返回张量的元素类型，类型是枚举 `ScalarType`，相当于 Python 的 `t.dtype`。定义见 [[pytorch:aten/src/ATen/core/TensorBase.h@L363|TensorBase.h · scalar_type()]]。
- **`#t`**：预处理器的**字符串化**。`CHECK_F32_CUDA(bias)` 里的 `#t` 变成字面量 `"bias"`。
- **`"bias" " must be float32"`**：相邻的两个字符串字面量会被编译器自动拼成一个，所以最终消息是 `"bias must be float32"`，直接告诉你是哪个参数出了问题。

```cpp
CHECK_F32_CUDA(bias);
// 预处理之后，第三行变成：
TORCH_CHECK((bias).scalar_type() == at::kFloat, "bias" " must be float32");
//                                              ^^^^^^ #t     ^^^^^^^^^^^^^^^^ 自动拼接
```

#### at::kFloat 的来历

`kFloat` 不是某个文件里手写的 `const`，而是用宏**批量生成**的，再经过两次 `using namespace` 才能写成 `at::kFloat`：

::: flow at::kFloat 是怎么来的
cfg cw=200 nw=160
n enum 0 0 "enum class ScalarType" "Byte … Float … BFloat16" k=vio w=1.2
n def 0 1.2 "DEFINE_CONSTANT 宏" "kFloat = ScalarType::Float" k=vio w=1.2
n c10 0 2.4 "c10::kFloat" "真正定义在这里" k=acc
n at 1.4 2.4 "at::kFloat" "using namespace c10" k=acc
n torch 2.8 2.4 "torch::kFloat" "using namespace at" k=acc
n you 1.4 1.2 "你的代码" "== at::kFloat" k=pass
e enum -> def "对每个类型展开一次"
e def -> c10
e c10 -> at "导入"
e at -> torch "导入"
e at -> you "查到"
cap: 三个名字指向同一个常量。枚举在 torch/headeronly/core/ScalarType.h，k 前缀常量在 c10/core/ScalarType.h，at 导入 c10 在 torch/headeronly/macros/Macros.h，torch 导入 at 在 torch/types.h。
:::

对应源码（点开就是那一行）：

1. 枚举本体：[[pytorch:torch/headeronly/core/ScalarType.h@L263|enum class ScalarType]]；`Float` 是第 6 号，见 [[pytorch:torch/headeronly/core/ScalarType.h@L112|_(float, Float) /* 6 */]]
2. 生成 `k` 前缀常量：[[pytorch:c10/core/ScalarType.h@L39-L43|DEFINE_CONSTANT → constexpr kFloat]]。`k##name` 是宏的**拼接**操作，`k` 和 `Float` 拼成 `kFloat`
3. `at` 导入 `c10`：[[pytorch:torch/headeronly/macros/Macros.h@L172-L174|namespace at { using namespace c10; }]]
4. `torch` 导入 `at`：[[pytorch:torch/csrc/api/include/torch/types.h@L36|namespace torch { using namespace at; }]]

C++ 常量和 Python dtype 的对应：

| C++ | Python | 字节 |
|---|---|---|
| `at::kFloat` | `torch.float32` | 4 |
| `at::kHalf` | `torch.float16` | 2 |
| `at::kBFloat16` | `torch.bfloat16` | 2 |
| `at::kDouble` | `torch.float64` | 8 |
| `at::kInt` / `at::kLong` | `torch.int32` / `torch.int64` | 4 / 8 |

::: callout analogy 登机口查票
`CHECK_F32_CUDA` 是登机口的工作人员：设备（是不是这趟航班）、连续性（座位号是不是连号）、dtype（是不是这个舱位的票）。查票只花一秒，但如果不查，乘客会被送到错误的城市，而且没人告诉他（kernel 静默地算出垃圾值）。`#t` 就是工作人员在广播里念出名字：“**bias** 旅客的票不对”，而不是只说“有人的票不对”。
:::

::: answer 自测：传一个 fp16 张量进来，报错信息是什么？
`a must be float32`（如果是第一个参数 `a`）。前两项检查都通过了（在 CUDA 上，也是连续的），第三项失败，`#t` 展开成 `"a"`。Python 端看到的是 `RuntimeError: a must be float32`。
:::

## ② blockDim.x 从哪来？为什么点不进去？

<p class="q">你问的是 kernel 里的 <code>blockDim.x</code>：它也是 torch 头文件带进来的吗？为什么在编辑器里 click 不到定义？· [[here:solutions/c1_elementwise.cu@L27|c1_elementwise.cu:27]]</p>

**不是 torch 带来的。** `blockDim`、`threadIdx`、`blockIdx`、`gridDim`、`warpSize` 是 **CUDA 语言自带的内置变量**，由 nvcc 提供，你不需要 `#include` 任何东西。

| 内置变量 | 含义 | 本文件里等于 | 读的是哪个 PTX 特殊寄存器 |
|---|---|---|---|
| `threadIdx.x` | 我在 block 里排第几 | 0 … 255 | `%tid.x` |
| `blockIdx.x` | 我的 block 在 grid 里排第几 | 0 … blocks−1 | `%ctaid.x` |
| `blockDim.x` | 每个 block 有几个线程 | 256 | `%ntid.x` |
| `gridDim.x` | grid 有几个 block | `blocks` | `%nctaid.x` |

::: flow blockDim.x 的一生：从 <<<>>> 到寄存器
cfg cw=210 nw=170
n launch 0 0 "<<<blocks, 256>>>" "host 端的启动参数" k=acc
n hw 1 0 "驱动 + 硬件" "建 block 时写入 256" k=vio
n reg 2 0 "特殊寄存器 %ntid.x" "只读，值 = 256" k=warn
n hdr 0 1.4 "device_launch_parameters.h" "extern const dim3 blockDim" k=ghost w=1.25
n kern 2 1.4 "kernel 里的 blockDim.x" "编译成 mov %ntid.x" k=pass
e launch -> hw "启动"
e hw -> reg "写入"
e hdr -> kern "只提供名字（声明）" dash
e reg -> kern "运行时读" k=pass anim
cap: 头文件里的 extern const 只是让编译器“认识这个名字”，它没有 C++ 实现：nvcc 把 blockDim.x 直接翻译成一条读特殊寄存器的指令，值由你 <<<>>> 里传的 256 决定。
:::

细节：

- nvcc 编译 `.cu` 时会**隐式 include** `cuda_runtime.h`，后者又 include 了 `device_launch_parameters.h`，里面声明了 `extern const dim3 blockDim;` 等几行。CUDA Toolkit 的头文件不在 GitHub 上，装好 Toolkit 后在 `$CUDA_HOME/include/device_launch_parameters.h`。
- clang 也能编 CUDA，它的同等定义是开源的，读起来最直观：[[llvm:clang/lib/Headers/__clang_cuda_builtin_vars.h@L79-L83|struct __cuda_builtin_blockDim_t]]，每个分量 `x` 就是调用 `__nvvm_read_ptx_sreg_ntid_x()`，也就是读 `%ntid.x`；变量本身在 [[llvm:clang/lib/Headers/__clang_cuda_builtin_vars.h@L109|blockDim]]，[[llvm:clang/lib/Headers/__clang_cuda_builtin_vars.h@L115|warpSize = 32]]。
- 官方文档：[CUDA Programming Guide · Built-in Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#built-in-variables)；PTX 寄存器：[%ntid](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-ntid)、[%tid](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-tid)、[%ctaid](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-ctaid)。

#### 为什么在 Mac 上点不进去

::: flow 编辑器为什么找不到 blockDim
cfg cw=250 nw=200
n ide 0 0 "编辑器 / clangd" "打开 c1_elementwise.cu" k=acc
n cpp 1 0 "当成普通 C++ 解析" "不知道这是 CUDA" k=warn
n look 2 0 "找 blockDim 的声明" "没有隐式 include" k=warn
n miss 2 1.3 "找不到" "macOS 上没有 CUDA 头文件" k=todo
n fix 0 1.3 ".clangd 配成 CUDA 模式" "-xcuda -nocudainc …" k=pass
n clang 1 1.3 "clang 自带的定义" "__clang_cuda_builtin_vars.h" k=pass
e ide -> cpp
e cpp -> look
e look -> miss "结果" k=todo
e fix -> clang "能跳到" k=pass
cap: 两层原因：NVIDIA 从 CUDA 10.2 起不再支持 macOS，本机根本没有这些头文件；编辑器默认也不会像 nvcc 那样做隐式 include。
:::

解决办法：在仓库根目录运行一次 `./setup_ide.sh`（之后在 VS Code 里 Reload Window）。它会：

- 从 NVIDIA 官方的 pip wheel 里取出 CUDA 头文件，放到 `.refs/cuda/`；
- 生成 `.clangd`，让 clangd 按 CUDA 模式解析 `.cu`，并加上 torch、CUDA、FlashInfer 的 include 路径；
- 让 VS Code 用 clangd 代替 C/C++ 扩展的 IntelliSense。

在这台 Mac 上验证过：`blockDim` 跳到 clang 的 [[llvm:clang/lib/Headers/__clang_cuda_builtin_vars.h@L109|__clang_cuda_builtin_vars.h]]（本机是 `/Library/Developer/CommandLineTools/usr/lib/clang/16/include/` 下的同一个文件），`TORCH_CHECK`、`at::kFloat`、`getCurrentCUDAStream`、`C10_CUDA_KERNEL_LAUNCH_CHECK`、`atomicAdd` 都能跳到 torch / CUDA 头文件里的定义。在 Linux GPU 机器上用 Remote-SSH 打开项目也可以，那里有完整的 CUDA toolkit。


::: callout analogy 工牌上印好的“本组人数”
`blockDim.x` 不是你声明的变量，而像是公司发给每个员工的工牌上印着的一行字：“本组共 256 人”。组建团队的时候（kernel 启动），人事部（硬件）就把数字印上去了，员工只能读，不能改。你在代码里“找它的定义”，就像在公司规章里找“你工牌上的 256 是哪一条规定的”：答案不在规章里，而在组建团队的那张申请单（`<<<blocks, 256>>>`）上。
:::

## ③ at 是什么？at / c10 / torch 什么关系？

<p class="q">你问的是 <code>at::kFloat</code>、<code>at::cuda::getCurrentCUDAStream()</code> 里的 <b>at</b> · [[here:solutions/c1_elementwise.cu@L44|c1_elementwise.cu:44]]</p>

`at` 是 **ATen**（“A Tensor Library”）的命名空间：PyTorch 底层的 C++ 张量库。PyTorch 的 C++ 部分分三层：

::: flow PyTorch C++ 的三层命名空间
cfg cw=200 nw=170
n torch 0 0 "torch::" "C++ 前端：nn、optim" k=teal w=1.2
n at 0 1.2 "at::（ATen）" "Tensor + 几千个算子" k=acc w=1.2
n c10 0 2.4 "c10::" "ScalarType、Device、异常" k=vio w=1.2
n atc 1.6 1.2 "at::cuda::" "getCurrentCUDAStream …" k=acc w=1.2
n c10c 1.6 2.4 "c10::cuda::" "CUDAStream、检查宏" k=vio w=1.2
e torch -> at "using namespace at"
e at -> c10 "using namespace c10"
e atc -> c10c "using namespace c10::cuda"
cap: 箭头是“导入”：上层能直接用下层的名字。所以 torch::Tensor 就是 at::Tensor，at::kFloat 就是 c10::kFloat，at::cuda::getCurrentCUDAStream 就是 c10::cuda::getCurrentCUDAStream。
:::

| 命名空间 | 名字来源 | 放什么 | 本文件里用到的 |
|---|---|---|---|
| `c10` | “Caffe2” + “ATen” 的谐音（ca-ten → c-10） | 最底层的基础设施 | `TORCH_CHECK`、`C10_CUDA_KERNEL_LAUNCH_CHECK`、`kFloat` 的真身 |
| `at` | ATen | 张量和算子 | `at::kFloat`、`at::cuda::getCurrentCUDAStream`、`at::cuda::getCurrentDeviceProperties` |
| `torch` | C++ 前端 | 面向用户的 API | `torch::Tensor`、`torch::empty_like` |

源码：[[pytorch:torch/headeronly/macros/Macros.h@L172-L177|at 导入 c10、at::cuda 导入 c10::cuda]] · [[pytorch:torch/csrc/api/include/torch/types.h@L12-L36|torch 导入 at（附长注释解释为什么）]] · 你 include 的 [[pytorch:torch/extension.h|torch/extension.h]] 把这一切都带进来。

::: callout analogy 一栋楼的三层
c10 是地下室的配电和水管（类型、设备、报错机制），at 是中间的工厂车间（张量和算子都在这里生产），torch 是一楼的门店（给顾客用的 nn、optim）。`using namespace` 相当于在楼上开了一个通往楼下的门：站在 at 层就能直接拿到 c10 的东西，所以 `at::kFloat` 和 `c10::kFloat` 是同一件货。写 kernel 的人习惯写 `at::`，因为我们干的就是车间里的活。
:::

## ④ vector_add：threads = 256 和 ceil 除法

<p class="q">你问的是 <b>第 41–42 行</b>：<code>threads = 256</code> 和 <code>blocks = (n + threads - 1) / threads</code> · [[here:solutions/c1_elementwise.cu@L33-L49|c1_elementwise.cu:33–49]]</p>

先看整个 host 函数做了什么。它跑在 CPU 上，自己不做加法，只负责把 GPU 上的活安排好：

::: flow vector_add 这个 host 函数的流程
cfg cw=200 nw=168
n chk 0 0 "CHECK_F32_CUDA ×2" "设备、连续、dtype" k=warn
n alloc 1 0 "empty_like(a)" "分配 c，不清零" k=acc
n calc 2 0 "算 launch 配置" "threads, blocks" k=acc
n launch 3 0 "<<<…, stream>>>" "排进队列，立刻返回" k=vio
n lchk 4 0 "LAUNCH_CHECK" "cudaGetLastError()" k=warn
n gpu 3 1.4 "GPU 稍后执行" "c[i] = a[i] + b[i]" k=pass
n ret 4 1.4 "return c" "回到 Python" k=pass
e chk -> alloc
e alloc -> calc
e calc -> launch
e launch -> lchk
e launch -> gpu "异步" k=pass anim
e lchk -> ret
cap: kernel 启动是异步的：<<<>>> 只是把任务放进 stream 的队列，CPU 马上往下走。return c 的时候 GPU 可能还在算，但 c 后续的使用会排在同一个 stream 上，所以拿到的一定是算完的结果。
:::

#### 核心想法：一个线程负责一个元素

CUDA 不让你直接说“给我 n 个线程”，只能说“给我 `blocks` 个 block，每个 block `threads` 个线程”。所以要把 n 个元素按 256 个一组切开，算出要几组。

**`threads = 256`：每个 block 的线程数。**

- **必须是 32 的倍数。** GPU 按 warp（32 个线程）调度。写 100 的话，硬件照样分配 4 个 warp = 128 个线程的位置，最后一个 warp 只有 4 个线程在干活。
- **上限 1024。** 超过会让启动失败（下面的 `LAUNCH_CHECK` 就是为了抓这种错误）。
- **为什么是 256：** 够大，一个 block 有 8 个 warp，SM 可以在它们之间切换来掩盖访存延迟；又不太大，一个 SM 可以同时驻留好几个 block。对逐元素 kernel，128 到 512 之间差别通常很小。

**`blocks = (n + threads - 1) / threads`：向上取整的整数除法。**

整数除法 `n / threads` 是向下取整，会漏掉尾巴。先加上 `threads - 1` 再除，就变成 ⌈n / threads⌉。

| n | n / 256（向下） | (n + 255) / 256（向上） | 启动的线程数 | 空转的线程 |
|---|---|---|---|---|
| 256 | 1 | 1 | 256 | 0 |
| 257 | 1，漏 1 个 | **2** | 512 | 255 |
| 1000 | 3，漏 232 个 | **4** | 1024 | 24 |
| 1 | 0，什么都没做 | **1** | 256 | 255 |

n = 1000 时，4 个 block 怎么覆盖这 1000 个元素：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 256fr 256fr 256fr 232fr minmax(30px, 24fr); min-width: 520px">
<div class="c0">block 0<small>i = 0 … 255</small></div>
<div class="c1">block 1<small>i = 256 … 511</small></div>
<div class="c2">block 2<small>i = 512 … 767</small></div>
<div class="c3">block 3<small>i = 768 … 999</small></div>
<div class="cx">✗<small>24</small></div>
</div></div>
<p class="legend">block 3 的最后 24 个线程（i = 1000 … 1023）没有对应的元素，被 <code>if (i &lt; n)</code> 挡掉，什么都不做。</p>
:::

所以 `blocks` 的计算和 kernel 里的边界检查是**配套**的：前者宁可多发线程也不能漏元素，后者把多发的线程挡掉。

```cpp
// kernel 里（第 27–29 行）
int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;   // 全局编号
if (i < n) c[i] = a[i] + b[i];                                // 多出来的线程直接跳过
// 例：block 3 的第 231 号线程：i = 3 * 256 + 231 = 999 < 1000，干活
//     block 3 的第 232 号线程：i = 3 * 256 + 232 = 1000，不满足，跳过
```

**为什么到处都是 `int64_t`？** n 可能超过 2³¹（约 21 亿）。`blockIdx.x * blockDim.x` 两个都是 32 位无符号数，相乘会在 2³² 处回绕，所以第 27 行先把 `blockIdx.x` 转成 `int64_t` 再乘。

#### 剩下几行

- `if (n == 0) return c;`：n = 0 时 blocks 会算出 0，而用 0 个 block 启动 kernel 是 CUDA 错误。
- `at::cuda::getCurrentCUDAStream()`：拿 PyTorch 当前的 stream，让这个 kernel 和前后的 torch 操作按顺序执行。返回的 `CUDAStream` 能隐式转换成 `cudaStream_t`：[[pytorch:c10/cuda/CUDAStream.h@L84|operator cudaStream_t()]]，函数声明见 [[pytorch:c10/cuda/CUDAStream.h@L233|getCurrentCUDAStream]]。
- `<<<blocks, threads, 0, stream>>>`：四个参数依次是 grid 大小、block 大小、动态 shared memory 字节数（这里不用）、stream。见 [Programming Guide · Launching Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#launching-kernels)。
- `C10_CUDA_KERNEL_LAUNCH_CHECK()`：`<<<>>>` 本身不返回错误。这个宏立刻调用 `cudaGetLastError()`，把非法的启动配置（例如 threads > 1024）当场报出来：[[pytorch:c10/cuda/CUDAException.h@L85|C10_CUDA_KERNEL_LAUNCH_CHECK]]。它抓不到 kernel **运行中**的错误（比如越界访问），那些要等之后同步时才会暴露。

::: callout analogy 坐大巴
1000 个人去春游，每辆大巴 256 个座位。1000 / 256 = 3.9，向下取整订 3 辆车会落下 232 个人，所以必须订 4 辆（向上取整）。第 4 辆车有 24 个空座：这就是 `if (i < n)`，没有乘客的座位什么都不做。为什么座位数是 256 而不是 250？因为大巴按 32 座一排造（warp），订 250 座的车，最后一排照样占着一整排的位置。
:::

::: answer 自测：n = 513，blocks 是多少？会有几个线程空转？
(513 + 255) / 256 = 768 / 256 = 3。启动 3 × 256 = 768 个线程，768 − 513 = 255 个空转。
:::

## ⑤ `__device__ __forceinline__` 是什么意思？

<p class="q">你问的是 <b>第 54 行</b>：<code>__device__ __forceinline__ float gelu_tanh(float x)</code> · [[here:solutions/c1_elementwise.cu@L54-L58|c1_elementwise.cu:54–58]]</p>

两个修饰词各管一件事：`__device__` 说“这个函数在 GPU 上运行，给 kernel 调用”；`__forceinline__` 说“别生成函数调用，把函数体直接贴到调用处”。

#### `__device__`：执行空间

::: flow 本文件里三种函数分别跑在哪
cfg cw=290 nw=200
n host 0 0 "bias_gelu(x, bias)" "__host__（默认）· CPU" k=acc
n kern 1 0 "bias_gelu_kernel" "__global__ · GPU 入口" k=vio
n dev 2 0 "gelu_tanh(x)" "__device__ · GPU 辅助" k=pass
n inl 2 1.4 "内联之后" "函数体嵌进 kernel" k=pass
e host -> kern "<<<>>>"
e kern -> dev "普通调用"
e dev -> inl "__forceinline__" dash k=pass
cap: CPU 不能直接调 `__device__` 函数，kernel 也不能用普通调用语法去调另一个 `__global__`。`__global__` 必须返回 void，`__device__` 可以有返回值（这里返回 float）。
:::

| 限定符 | 在哪运行 | 谁能调用 | 例子 |
|---|---|---|---|
| `__global__` | GPU | CPU 用 `<<<>>>` 启动 | `bias_gelu_kernel`（第 63 行） |
| `__device__` | GPU | 只有 GPU 代码 | `gelu_tanh`（第 54 行） |
| `__host__`（不写就是它） | CPU | CPU | `bias_gelu`（第 75 行） |

如果把 `__device__` 删掉，`gelu_tanh` 就成了普通 CPU 函数，kernel 里调用它会直接编译报错。

#### `__forceinline__`：强制内联

内联之后，第 71 行在编译器眼里相当于：

```cpp
float v = x[i] + bias[i % cols];
y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
```

为什么在 GPU 上这很重要：

- **省掉调用开销。** GPU 上真正的函数调用要按调用约定传参、保存寄存器，可能还要用栈，而栈放在 local memory 里，很慢。对只有几条算术指令的函数，这些开销可能比函数本身还大。
- **让编译器继续优化。** 展开以后，编译器能把它和周围代码一起分配寄存器、重排指令、合并成 FMA（乘加）。

nvcc 默认就会积极内联小的 `__device__` 函数，`__forceinline__` 是把“大概率内联”变成“一定内联”，也写明了作者的意图。反过来的 `__noinline__` 用于很大的函数，避免代码膨胀。文档：[Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#execution-space-specifiers) · [Inline Specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#inline-specifiers)。

对照 PyTorch 自己的实现：`F.gelu(approximate="tanh")` 的 CUDA kernel 用的是同一个公式，`kBeta` 就是这里的 `k0 = sqrt(2/π)`，`kKappa` 就是 `k1 = 0.044715`：[[pytorch:aten/src/ATen/native/cuda/ActivationGeluKernel.cu@L26-L30|ActivationGeluKernel.cu · tanh GELU]]。

::: callout analogy 抄菜谱 vs 打电话问
厨师（kernel）做菜做到一半需要调一个酱汁（gelu_tanh）。不内联就像每做一份都放下手里的活，打电话问酱汁大师傅怎么调，听完再回来接着做（保存现场、跳转、返回）。内联就是把酱汁的配方直接抄进自己的菜谱里，一气呵成。配方只有三行时，打电话的时间比调酱汁还长，当然应该抄进来。`__device__` 则说明这本菜谱只在后厨（GPU）用，前台（CPU）拿到也看不懂。
:::

## ⑥ grid-stride loop：一个线程处理好几个元素

<p class="q">你问的是 <b>第 68–72 行</b>的 for 循环 · [[here:solutions/c1_elementwise.cu@L68-L72|c1_elementwise.cu:68–72]]</p>

```cpp
const int64_t stride = (int64_t)blockDim.x * gridDim.x;      // 整个 grid 的线程总数
for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
  y[i] = gelu_tanh(x[i] + bias[i % cols]);
}
```

和 vector_add 的区别：vector_add 启动“刚好够”的线程，每个线程只做一次；这里 block 数**有上限**（见第 ⑦ 节），线程可能比元素少，所以每个线程**循环**：做完一个元素，往后跳 `stride` 个再做一个。

- `stride = blockDim.x * gridDim.x` = 每个 block 的线程数 × block 数 = **整个 grid 的线程总数**
- 起点 `i` 和 vector_add 一样，是这个线程的全局编号
- `i < total` 同时是边界检查

#### 小数字走一遍：2 个 block × 4 个线程，total = 20

stride = 4 × 2 = 8。颜色表示 block，格子里写的是元素下标和处理它的线程（全局编号 g）：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 64px repeat(8, minmax(44px, 1fr)); min-width: 470px">
<div class="lbl">第 1 轮</div>
<div class="c0">0<small>g0</small></div><div class="c0">1<small>g1</small></div><div class="c0">2<small>g2</small></div><div class="c0">3<small>g3</small></div>
<div class="c1">4<small>g4</small></div><div class="c1">5<small>g5</small></div><div class="c1">6<small>g6</small></div><div class="c1">7<small>g7</small></div>
<div class="lbl">第 2 轮</div>
<div class="c0">8<small>g0</small></div><div class="c0">9<small>g1</small></div><div class="c0">10<small>g2</small></div><div class="c0">11<small>g3</small></div>
<div class="c1">12<small>g4</small></div><div class="c1">13<small>g5</small></div><div class="c1">14<small>g6</small></div><div class="c1">15<small>g7</small></div>
<div class="lbl">第 3 轮</div>
<div class="c0">16<small>g0</small></div><div class="c0">17<small>g1</small></div><div class="c0">18<small>g2</small></div><div class="c0">19<small>g3</small></div>
<div class="cx">20<small>退出</small></div><div class="cx">21<small>退出</small></div><div class="cx">22<small>退出</small></div><div class="cx">23<small>退出</small></div>
</div></div>
<p class="legend"><span class="c0"></span>block 0（g0–g3）<span class="c1"></span>block 1（g4–g7）<span class="cx"></span>i ≥ total，循环结束</p>
:::

- 线程 g0 处理 0 → 8 → 16，下一个是 24，退出
- 线程 g5 处理 5 → 13，下一个是 21，不满足 `21 < 20`，退出

整个 grid 像一把梳子，每轮整体往前平移 stride 个元素，直到扫完数组。

#### 为什么跨步，而不是每个线程分一段连续的？

另一种直觉是“线程 0 做 0–2，线程 1 做 3–5……”。这样性能很差，原因是**合并访存（coalescing）**。

::: flow 同一个 warp、同一条 load 指令，两种分法
cfg cw=250 nw=170
n warp 0 0.6 "1 个 warp 的 32 个线程" "同时执行同一条 load" k=vio w=1.1
n good 1 0 "grid-stride" "地址 i, i+1, …, i+31" k=pass
n bad 1 1.2 "每线程一段" "地址相隔一整段" k=todo
n t1 2 0 "4 个 32B sector" "= 1 条 128B cache line" k=pass
n t2 2 1.2 "最多 32 个 sector" "每个只用 4B，浪费 7/8" k=todo
e warp -> good
e warp -> bad
e good -> t1 "连续 128B" k=pass anim
e bad -> t2 "分散" k=todo anim
cap: 显存按 32 字节的 sector 搬运。32 个相邻的 float 正好是连续的 128 字节，一次就搬完；如果 32 个线程各读一个分散的地址，每个 sector 只用到 4 字节。
:::

关键点：grid-stride 是在**同一个线程的不同轮次之间**跨步；**同一轮里，相邻线程读的永远是相邻地址**。延伸阅读：[NVIDIA blog · Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) · [Programming Guide · Coalesced Global Memory Access](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access)。

::: callout analogy 发牌，而不是切牌
8 个人打牌，荷官发 20 张牌：一人一张、转一圈，再一人一张……这就是 grid-stride：第 0 个人拿到第 0、8、16 张。另一种做法是把牌切成 8 叠，每人拿一叠连续的，这就是“每线程一段”。
为什么发牌更快？想象牌在一条传送带上，每一轮 8 个人**同时**伸手，正好拿走传送带上**挨着的 8 张**，一次抓取就够了（合并访存）。切牌的话，8 个人同时伸手去抓 8 个相隔很远的位置，传送带要停 8 次。
:::

## ⑦ block 数上限 sms × 32 在哪里？为什么要限？

<p class="q">你问的是“为什么要限制 block 数（sms × 32），我没看到”：它在 host 函数 <code>bias_gelu</code> 里，<b>第 84–88 行</b> · [[here:solutions/c1_elementwise.cu@L84-L88|c1_elementwise.cu:84–88]]</p>

```cpp
const int threads = 256;
const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
const int64_t blocks = std::min<int64_t>((total + threads - 1) / threads, (int64_t)sms * 32);
```

kernel 里看不到，是因为 **kernel 不决定启动多少个 block**。block 数是 host 端在 `<<<blocks, threads>>>` 里传进去的，kernel 只能通过 `gridDim.x` 读到它。

::: flow block 数怎么算出来，又怎么变成 kernel 里的 stride
cfg cw=200 nw=160
n total 0 0 "total 个元素" "x.numel()" k=acc
n need 1 0 "ceil(total / 256)" "一线程一元素要的 block" k=acc w=1.3
n cap 1 1.3 "sms × 32" "RTX 5090：170 × 32" k=warn w=1.3
n min 2.35 0.65 "std::min" "取较小者 = blocks" k=vio
n grid 3.4 0.65 "<<<blocks, 256>>>" "kernel 里是 gridDim.x" k=vio w=1.1
n stride 3.4 2 "stride" "256 × gridDim.x" k=pass w=1.1
e total -> need
e need -> min
e cap -> min
e min -> grid
e grid -> stride "kernel 内读取" k=pass
cap: host 端封顶 block 数，kernel 端必须用 grid-stride 循环把剩下的元素扫完。两边必须配套：如果 kernel 还是 vector_add 那种只做一次的写法，超出线程总数的元素就没人处理了。
:::

- `getCurrentDeviceProperties()` 返回当前 GPU 的 `cudaDeviceProp`，`multiProcessorCount` 是 SM（流式多处理器）个数。源码：[[pytorch:aten/src/ATen/cuda/CUDAContextLight.h@L72|getCurrentDeviceProperties]]。
- 常见 GPU 的 SM 数：A100 是 108，H100 SXM 是 132，RTX 5090（本仓库实测用的卡）是 170。

#### 用 RTX 5090 算几个例子

sms × 32 = 170 × 32 = 5440 个 block，stride = 5440 × 256 = 1,392,640。

| total | ceil(total / 256) | blocks（取 min） | 每个线程最多循环几轮 |
|---|---|---|---|
| 100,000 | 391 | 391 | 1（和 vector_add 完全一样） |
| 1,000,000 | 3,907 | 3,907 | 1 |
| 10,000,000 | 39,063 | **5,440**（封顶） | 8 |
| 1,000,000,000 | 3,906,250 | **5,440**（封顶） | 719 |

（轮数 = ⌈total / stride⌉，例如 10⁷ / 1,392,640 ≈ 7.2，所以最多 8 轮。）

#### 为什么要封顶

- **数据少时没区别。** `min` 取的是 ceil(total/256)，每个线程只循环一次。
- **数据多时少调度开销。** 与其发几百万个只干一点活就退出的 block，不如让数量合适的 block 常驻在 SM 上循环干活。每个 SM 32 个 block，足够让所有 SM 都忙起来，并有足够多的 warp 轮换来掩盖访存延迟。
- **一个 kernel 适配任意大小。** 不管 total 多大，grid 大小都是有限的。

::: callout analogy 按桌子雇服务员，不按客人数
餐厅里来了 10,000 位客人。你不会雇 10,000 个服务员，每人服务一位就下班（每个 block 只干一点活），光是招人和交接就累死。合理的做法是按桌子数（SM 数）雇人，比如每张桌子配 32 个服务员，每个服务员轮流服务很多位客人（grid-stride 循环）。客人很少的时候，来几位就派几位服务员，和原来没区别。
:::

## ⑧ bias[i % cols]：扁平下标怎么找到列号

<p class="q">你问的是 <b>第 70–71 行</b>：<code>y[i] = gelu_tanh(x[i] + bias[i % cols]);</code> · [[here:solutions/c1_elementwise.cu@L70-L71|c1_elementwise.cu:70–71]]</p>

这一行等价于 PyTorch 的：

```python
y = F.gelu(x + bias, approximate="tanh")   # x: [rows, cols], bias: [cols]
```

关键是 `x + bias` 的**广播**：bias 只有一行，要加到 x 的**每一行**上。可 kernel 里没有“行”和“列”，只有一个扁平下标 `i`，所以要用 `i % cols` 算出当前元素在哪一列。

#### 第一步：二维张量在显存里是一条一维数组

设 rows = 3，cols = 4。颜色表示**列**：

::: raw
<p class="legend">逻辑上的二维 x（3 行 × 4 列）和 bias（4 个数）：</p>
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 56px repeat(4, minmax(52px, 1fr)); max-width: 420px">
<div class="lbl"></div><div class="lbl" style="text-align:center">列 0</div><div class="lbl" style="text-align:center">列 1</div><div class="lbl" style="text-align:center">列 2</div><div class="lbl" style="text-align:center">列 3</div>
<div class="lbl">行 0</div><div class="c0">x00</div><div class="c1">x01</div><div class="c2">x02</div><div class="c3">x03</div>
<div class="lbl">行 1</div><div class="c0">x10</div><div class="c1">x11</div><div class="c2">x12</div><div class="c3">x13</div>
<div class="lbl">行 2</div><div class="c0">x20</div><div class="c1">x21</div><div class="c2">x22</div><div class="c3">x23</div>
<div class="lbl">bias</div><div class="c0">b0</div><div class="c1">b1</div><div class="c2">b2</div><div class="c3">b3</div>
</div></div>
<p class="legend">显存里实际的样子（行优先：先存完第 0 行，再存第 1 行……）：</p>
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 74px repeat(12, minmax(40px, 1fr)); min-width: 600px">
<div class="lbl">i</div><div>0</div><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>7</div><div>8</div><div>9</div><div>10</div><div>11</div>
<div class="lbl">x[i]</div><div class="c0">x00</div><div class="c1">x01</div><div class="c2">x02</div><div class="c3">x03</div><div class="c0">x10</div><div class="c1">x11</div><div class="c2">x12</div><div class="c3">x13</div><div class="c0">x20</div><div class="c1">x21</div><div class="c2">x22</div><div class="c3">x23</div>
<div class="lbl">i % 4</div><div>0</div><div>1</div><div>2</div><div>3</div><div>0</div><div>1</div><div>2</div><div>3</div><div>0</div><div>1</div><div>2</div><div>3</div>
<div class="lbl">bias[i%4]</div><div class="c0">b0</div><div class="c1">b1</div><div class="c2">b2</div><div class="c3">b3</div><div class="c0">b0</div><div class="c1">b1</div><div class="c2">b2</div><div class="c3">b3</div><div class="c0">b0</div><div class="c1">b1</div><div class="c2">b2</div><div class="c3">b3</div>
</div></div>
<p class="legend">x[i] 和 bias[i % 4] 的颜色每一格都对得上：每到新的一行，i % cols 就回到 0，bias 从头再读一遍。这就是广播。</p>
:::

#### 第二步：从 i 反推行号和列号

每行有 `cols` 个元素，所以**行号 = `i / cols`**（前面有几个完整的行），**列号 = `i % cols`**（在当前行里排第几）。

| i | i / 4（行） | i % 4（列） | 元素 | 加的 bias |
|---|---|---|---|---|
| 0 | 0 | 0 | x00 | b0 |
| 3 | 0 | 3 | x03 | b3 |
| 4 | 1 | **0** | x10 | **b0**，换行后列号回到 0 |
| 6 | 1 | 2 | x12 | b2 |
| 11 | 2 | 3 | x23 | b3 |

只需要列号，所以代码里没有算行号。

#### 第三步：这一行按顺序做了什么，以及“融合”省在哪

1. 读 `x[i]`（显存）
2. 读 `bias[i % cols]`（几乎总是命中缓存，见下）
3. 相加，结果在寄存器里
4. `gelu_tanh`，结果还在寄存器里
5. 写 `y[i]`（显存）

::: flow 未融合 vs 融合：数据进出显存几次
cfg cw=180 nw=150
n x1 0 0 "x" "显存" k=todo
n k1 1 0 "kernel 1" "t = x + bias" k=acc
n t 2 0 "t" "显存里的中间结果" k=todo
n k2 3 0 "kernel 2" "y = gelu(t)" k=acc
n y1 4 0 "y" "显存" k=todo
n x2 0 1.5 "x" "显存" k=todo
n kf 2 1.5 "融合 kernel" "+ 和 GELU 都在寄存器" k=pass w=1.15
n y2 4 1.5 "y" "显存" k=todo
g 0 0 4 0 "未融合：读 x、写 t、读 t、写 y = 4 次" k=todo
g 0 1.5 4 1.5 "融合：读 x、写 y = 2 次" k=pass
e x1 -> k1 "读"
e k1 -> t "写"
e t -> k2 "读"
e k2 -> y1 "写"
e x2 -> kf "读" k=pass
e kf -> y2 "写" k=pass
cap: 逐元素运算的瓶颈是显存带宽，不是算力。显存流量减半，速度就接近翻倍。中间结果 t 从来没有写进显存，这就是 fusion 省下的东西。
:::

**为什么注释说 bias “stays in L1/L2”：** x 的每个元素只读一次，实打实地消耗显存带宽；bias 很小（cols = 4096 时只有 16 KB），却被每一行重复读一遍。第一次读之后它就留在缓存里（L2 有几十 MB），之后几乎全部命中，不占显存带宽。所以这个 kernel 的显存流量就是读 x 一次加写 y 一次。

**一个小代价：** `%` 是 64 位整数取模，GPU 上要用几十条指令模拟。这个 kernel 受访存限制，取模的时间通常被访存延迟掩盖掉。想进一步优化，可以改成二维 grid，让 block 直接按行列索引，就不需要取模了。

**前提：x 必须是连续的。** “列号 = i % cols”只在 x 真的是紧密的行优先布局时成立。传一个 `x.t()` 进来，逻辑形状没变，内存顺序却变了，算出的列号就错了，而且不会报错。这正是第 ① 节里 `is_contiguous()` 检查存在的原因。

::: callout analogy 日历和每周课表
把一年的日子从 0 开始编号：第 i 天。不管过了多少周，“今天星期几”就是 `i % 7`。bias 就像一张每周固定的课表：周一体育、周二美术……第 i 天上什么课 = `课表[i % 7]`。x 是 365 天的日历（按周一行排开，行优先），你不需要知道今天是第几周（行号），只需要知道星期几（列号），就能查到当天的课。
:::

::: answer 自测：cols = 4096，i = 10,000，这个元素在哪一行哪一列？加的是 bias 的第几个？
行 = 10000 / 4096 = 2，列 = 10000 % 4096 = 10000 − 8192 = 1808。所以是第 2 行第 1808 列，加的是 `bias[1808]`。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/c1_elementwise.cu|solutions/c1_elementwise.cu]] · [[here:mine/c1_elementwise.cu|mine/c1_elementwise.cu（你写的）]] · [[here:content/c1.md|content/c1.md（题目页）]]

**PyTorch（钉在固定 commit）**

- 检查与报错：[[pytorch:c10/util/Exception.h@L627|TORCH_CHECK]] · [[pytorch:c10/cuda/CUDAException.h@L39|C10_CUDA_CHECK]] · [[pytorch:c10/cuda/CUDAException.h@L85|C10_CUDA_KERNEL_LAUNCH_CHECK]]
- dtype：[[pytorch:torch/headeronly/core/ScalarType.h@L263|enum ScalarType]] · [[pytorch:c10/core/ScalarType.h@L39-L43|kFloat 等常量]]
- 命名空间：[[pytorch:torch/headeronly/macros/Macros.h@L172-L177|at ← c10]] · [[pytorch:torch/csrc/api/include/torch/types.h@L36|torch ← at]]
- Tensor 方法：[[pytorch:aten/src/ATen/core/TensorBase.h@L455|is_cuda]] · [[pytorch:aten/src/ATen/core/TensorBase.h@L268|is_contiguous]] · [[pytorch:aten/src/ATen/core/TensorBase.h@L337|numel]] · [[pytorch:aten/src/ATen/core/TensorBase.h@L363|scalar_type]] · [[pytorch:aten/src/ATen/core/TensorBase.h@L623|data_ptr&lt;T&gt;]]
- CUDA 上下文：[[pytorch:c10/cuda/CUDAStream.h@L233|getCurrentCUDAStream]] · [[pytorch:aten/src/ATen/cuda/CUDAContextLight.h@L72|getCurrentDeviceProperties]]
- 对照实现：[[pytorch:aten/src/ATen/native/cuda/ActivationGeluKernel.cu@L26-L30|PyTorch 的 tanh-GELU kernel]]

**CUDA 内置变量（clang 的开源实现）**：[[llvm:clang/lib/Headers/__clang_cuda_builtin_vars.h@L79-L83|blockDim → %ntid]] · [[llvm:clang/lib/Headers/__clang_cuda_builtin_vars.h@L109|blockDim 变量]]

**文档**

- [Programming Guide · Built-in Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#built-in-variables) · [Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#execution-space-specifiers) · [Inline Specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#inline-specifiers) · [`__restrict__`](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#restrict-pointers)
- [Programming Guide · Launching Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#launching-kernels) · [Error Checking](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#error-checking-in-cuda) · [Coalesced Access](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access)
- [PTX · special registers](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers)
- [NVIDIA blog · Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) · [How to Access Global Memory Efficiently](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [PyTorch · Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) · [PyTorch C++ API](https://docs.pytorch.org/cppdocs/)
