title: F5 讲解：读懂 FlashInfer 的分层，准备第一个 PR
lede: 这一题没有参考解，要动手的是 FlashInfer 仓库本身。这一页把 rmsnorm 从 Python 到 kernel 的每一层都钉到具体的源码行，讲清楚每一层为什么存在（框架无关、TVM-FFI、JIT 缓存键），再把“第一个 PR”拆成可执行的步骤，对照项目的 PR 模板逐节说明要写什么。

::: callout info 怎么读这一页
这一题的内容以题目页 [[here:content/f5.md|content/f5.md]] 的四个 Lab 为主，这一页是配套的“源码导读”：Lab 2 追调用链时开着这一页，每一层都能点到对应的行。FlashInfer 的链接都钉在本仓库固定的 commit 上；实际动手时以你 clone 下来的最新代码为准，行号可能会变。
:::

## ① 先看全貌：一个 API 的五层

拿 `flashinfer.rmsnorm(x, w)` 举例，一次调用依次经过：

::: flow rmsnorm 的五层，每一层只做一件事
cfg cw=250 nw=180
n py 0 0 "① Python API" "参数检查、选后端" k=acc
n mod 1 0 "② 模块缓存" "@functools.cache" k=acc
n jit 2 0 "③ JIT 生成器" "列出源文件和编译参数" k=vio
n csrc 2 1.3 "④ csrc launcher" "张量 → 指针，按 dtype 分发" k=warn w=1.05
n inc 1 1.3 "⑤ include kernel" "只认裸指针" k=teal
n so 0 1.3 "~/.cache/flashinfer" "编译好的 .so" k=ghost
e py -> mod
e mod -> jit "首次"
e jit -> csrc "ninja"
e csrc -> inc "实例化"
e inc -> so dash
e so -> py "以后直接加载" dash
cap: 第 ①② 层是 Python，第 ③ 层生成编译任务，第 ④⑤ 层是 C++/CUDA。第一次调用会走完整条链路编译出 .so，之后 @functools.cache 和磁盘缓存让调用直接从 ① 跳到 .so 里的函数。
:::

| 层 | 文件 | 这一层的源码 | 职责 |
|---|---|---|---|
| ① | `flashinfer/norm/__init__.py` | [[flashinfer:flashinfer/norm/__init__.py@L1125-L1126|@flashinfer_api + def rmsnorm]] → [[flashinfer:flashinfer/norm/__init__.py@L1173|get_norm_module().rmsnorm(…)]] | 检查形状、分配输出、决定走哪个后端 |
| ② | 同上 | [[flashinfer:flashinfer/norm/__init__.py@L158-L165|@functools.cache def get_norm_module]] | 进程内只编译 / 加载一次 |
| ③ | `flashinfer/jit/norm.py` | [[flashinfer:flashinfer/jit/norm.py@L21-L34|gen_norm_module]]：两个源文件 + `-DENABLE_BF16 -DENABLE_FP8` | 描述“要编译什么”，返回 JitSpec |
| ④ | `csrc/norm.cu` + 绑定 | [[flashinfer:csrc/norm.cu@L24|void rmsnorm(TensorView …)]]，[[flashinfer:csrc/norm.cu@L44-L50|按 dtype 分发并调用 norm::RMSNorm]]；导出 [[flashinfer:csrc/flashinfer_norm_binding.cu@L61|TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm, …)]] | 把框架的张量换成裸指针，选模板实例 |
| ⑤ | `include/flashinfer/norm.cuh` | launcher [[flashinfer:include/flashinfer/norm.cuh@L140-L145|RMSNorm（算 vec_size 和 block_size）]]，kernel [[flashinfer:include/flashinfer/norm.cuh@RMSNormKernel|RMSNormKernel]] | 真正的 CUDA kernel |

这条链和你在本套题里写的东西一一对应：第 ⑤ 层就是 T3 / F3 的 kernel，第 ④ 层相当于 C1 里的 host 函数 `vector_add`，第 ① 层相当于每道题 `.py` 文件里的 Python 包装函数。

::: callout analogy 电器和插头转换器
include 里的 kernel 是一台只认“裸电压”（裸指针）的电器，它不关心电是从哪个国家的插座来的（PyTorch、JAX 还是别的框架）。csrc 里的 launcher 和 TVM-FFI 绑定是插头转换器：把 PyTorch 张量这种“国标插头”转换成电器要的裸电压。所以项目有一条硬规定：**include 里不能 include Torch 的头文件**。电器里不能焊死某一国的插头，否则换个国家就用不了。
:::

## ② 为什么要这样分层

**框架无关**：项目的 CLAUDE.md 写明“Torch headers MUST NOT be included in `include/` directory files”。kernel 只收裸指针和整数，同一份 kernel 可以被不同框架复用。

**TVM-FFI**：绑定层用的不是 pybind11，而是 Apache TVM 的 FFI。`TensorView` 是一个与框架无关的张量视图（基于 DLPack），`TVM_FFI_DLL_EXPORT_TYPED_FUNC` 把 C++ 函数导出成可以从 Python 直接调用的符号。你在 C1 里用的 `torch/extension.h` + `TORCH_CHECK` 是 PyTorch 专用的做法；FlashInfer 的 [[flashinfer:csrc/norm.cu@L36-L40|TVM_FFI_ICHECK_EQ]] 是它的对应物。

**JIT**：kernel 的模板参数组合太多（dtype × head_dim × 架构 × 各种开关），全部预编译不现实，所以用到哪个组合才编译哪个。

## ③ JIT：缓存键里有什么

[[flashinfer:flashinfer/jit/core.py@L739|gen_jit_spec]] 返回一个 [[flashinfer:flashinfer/jit/core.py@L292|JitSpec]]，调用它的 [[flashinfer:flashinfer/jit/core.py@L372|build_and_load()]] 时：

::: flow build_and_load 的判断
cfg cw=230 nw=180
n s 0 0.65 "JitSpec" "名字 + 源文件 + 编译参数" k=vio
n k 1 0.65 "算缓存键" "参数 + 源码哈希 + flags + 架构" k=acc w=1.1
n hit 2.15 0 "命中" "直接加载 .so" k=pass
n miss 2.15 1.3 "未命中 / 源码改过" "写 build.ninja，编译" k=todo w=1.05
e s -> k
e k -> hit "有"
e k -> miss "没有"
e miss -> hit "编译完" dash
cap: 缓存键包含源文件的 SHA256，所以你改了 include 里的 .cuh，下一次调用会自动重新编译，不需要重新 pip install。这就是 Lab 2 里“改一行注释，再运行一次就会看到重新编译”的原理。
:::

常用的环境变量（都在项目 CLAUDE.md 里列着）：

| 变量 | 作用 |
|---|---|
| `FLASHINFER_JIT_VERBOSE=1` | 打印编译过程，Lab 2 用它看到 JIT 被触发 |
| `FLASHINFER_JIT_DEBUG=1` | 带调试符号、`-O0` 编译，配合 cuda-gdb |
| `FLASHINFER_CUDA_ARCH_LIST="12.0"` | 只为指定架构编译，省时间 |
| `FLASHINFER_LOGLEVEL=3` | API 级日志：kernel 崩溃前先把输入打出来 |
| `rm -rf ~/.cache/flashinfer/` | 清掉所有编译缓存 |

::: callout analogy 打印店的底版
打印店（JIT）收到一份订单（API 调用 + 参数），先查有没有现成的底版（缓存键匹配的 .so）。底版的编号里记着原稿的指纹（源码哈希）：原稿改了一个字，指纹就变了，旧底版作废，重新制版。所以你只管改原稿，不用操心去通知打印店。
:::

## ④ 第一个 PR：拆成可执行的步骤

题目页 Lab 4 列了几个方向。不管选哪个，流程是一样的：

::: flow 从找方向到合并
cfg cw=200 nw=165
n a 0 0 "确认没人在做" "搜 issues / 开 issue 说一声" k=acc w=1.12
n b 1 0 "改代码 + 测试" "tests/<领域>/" k=vio
n c 2 0 "前后 benchmark" "绝对数字 + GPU 型号" k=warn
n d 3 0 "pre-commit run -a" "" k=teal
n e 3 1.3 "开 PR" "保留默认模板" k=pass
n f 1.6 1.3 "@flashinfer-bot run tests/…" "只跑相关目录" k=pass w=1.3
n g 0 1.3 "review → squash merge" "" k=pass
e a -> b
e b -> c
e c -> d
e d -> e
e e -> f
e f -> g
cap: 文档和测试类的 PR 可以跳过 benchmark 这一步。性能类的 PR 没有前后数字基本不会被合并。
:::

一个最小的例子：给 `rmsnorm` 补一个边界测试（hidden size 不是 8 的倍数，这时 `vec_size = gcd(8, d)` 会退到更小的值）。

```bash
git checkout -b test-rmsnorm-odd-hidden
# 1. 在 tests/norm/ 里找到 rmsnorm 的参数化测试，加上 hidden_size = 1004（gcd(8, 1004) = 4）之类的值
pytest tests/norm -k "rmsnorm and not cake" -x -q
# 2. 在你的 GPU 上全部通过，再跑格式检查
pre-commit run -a
git commit -am "test(norm): cover rmsnorm hidden sizes not divisible by 8"
git push -u origin test-rmsnorm-odd-hidden
# 3. 开 PR，用默认模板；CI 里评论：@flashinfer-bot run tests/norm/
```

### PR 模板逐节填写

项目的 [[flashinfer:.github/pull_request_template.md|PR 模板]]：

| 小节 | 写什么 |
|---|---|
| [[flashinfer:.github/pull_request_template.md@L3|Description]] | 改了什么、为什么。squash merge 之后这段就是 commit message，几个月后有人 `git bisect` 到这里，读的就是它 |
| [[flashinfer:.github/pull_request_template.md@L7|Related Issues]] | 对应的 issue 编号 |
| [[flashinfer:.github/pull_request_template.md@L11-L19|Checklist · Pre-commit]] | 确认装了并运行过 pre-commit |
| [[flashinfer:.github/pull_request_template.md@L23-L27|Tests]] | 加了哪些测试；是否需要**全量** CI（默认不需要，需要的话要说明理由） |
| [[flashinfer:.github/pull_request_template.md@L33-L38|Experimental Track]] | 如果改动在 `flashinfer/experimental/` 下，填 tracking issue |
| [[flashinfer:.github/pull_request_template.md@L61|Reviewer Notes]] | 希望 reviewer 重点看什么；性能 PR 的 benchmark 表格放这里 |

**性能 PR 的数字怎么给**：GPU 型号、问题规模、改动前后的**绝对时间**（不只是加速比）、用的哪个 benchmark 命令。例如：

```text
RTX 5090, bf16, T=4096, H=4096, benchmarks/flashinfer_benchmark.py --routine <对应的 routine> ...
before: 92.1 us   after: 88.3 us
```

::: callout analogy 病历要写给下一个医生看
PR 描述不是写给现在的 reviewer 看的，而是写给半年后排查回归的人看的：他从 `git log` 里找到这个 commit，手里只有这段描述。“优化了 norm”是一句没用的病历；“RTX 5090 上 hidden=4096 时 rmsnorm 从 92 us 降到 88 us，原因是改用 16 字节向量化读取”才能让下一个医生对症下药。
:::

## ⑤ 读源码时的几个入口

- 项目给新人的三个参考实现（由易到难）：[[flashinfer:flashinfer/norm/__init__.py|norm/__init__.py]]（不用 Jinja）、[[flashinfer:flashinfer/sampling.py|sampling.py]]（用了 Jinja 模板）、[[flashinfer:flashinfer/decode.py@L1354|decode.py]]（plan / run 和 workspace，F1 讲过）。
- 你在本套题里接触过的 FlashInfer 代码：F1 的 [[flashinfer:include/flashinfer/attention/scheduler.cuh@L150|decode 调度]]、F2 的 [[flashinfer:include/flashinfer/attention/cascade.cuh@L45|MergeStateKernel]]、F3 的 [[flashinfer:include/flashinfer/norm.cuh@L416|FusedAddRMSNormKernel]]、F4 的 [[flashinfer:include/flashinfer/sampling.cuh@L1709|TopPRenormProbKernel]]、T1 的 [[flashinfer:flashinfer/triton/kernels/activation.py@L8|Triton silu_and_mul]]。从熟悉的算子开始读，最容易看出门道。
- API 日志装饰器：[[flashinfer:flashinfer/api_logging.py@flashinfer_api|flashinfer_api]]；实验性 API：[[flashinfer:flashinfer/api_logging.py@flashinfer_experimental_api|flashinfer_experimental_api]]。

## ⑥ 常见坑与自查

- **`pip install -e .` 没加 `--no-build-isolation`**：pip 可能另装一个 CUDA 版本不同的 torch，后面全乱。
- **clone 没加 `--recursive`**：`3rdparty/cutlass` 是空的，编译报找不到头文件。补救：`git submodule update --init --recursive`。
- **在 include 里 `#include <torch/...>`**：违反分层规则，review 一定会被打回。
- **CI 评论不带路径**：`@flashinfer-bot run` 不带路径会跑全部测试，代价很高。
- **自己发明 PR 描述格式**：项目明确要求保留默认模板。

::: answer 自测：改了 include/flashinfer/norm.cuh 里的一行，需要重新 pip install 吗？为什么？
不需要。JIT 的缓存键包含源文件的 SHA256，下次调用 rmsnorm 时 FlashInfer 发现哈希变了，会自动重新生成源码、重新编译。前提是用 `pip install -e .`（可编辑安装），这样 Python 包里的 `data/include` 是指向你仓库目录的符号链接。
:::

## 源码与资料索引

**本仓库**：[[here:content/f5.md|content/f5.md（四个 Lab）]]

**FlashInfer 规范**：[[flashinfer:CONTRIBUTING.md|CONTRIBUTING.md]] · [[flashinfer:CLAUDE.md|CLAUDE.md]] · [[flashinfer:.github/pull_request_template.md|PR 模板]] · [[flashinfer:docs/code_review_guidance.md|code_review_guidance.md]]

**rmsnorm 调用链**：[[flashinfer:flashinfer/norm/__init__.py@L1126|rmsnorm]] → [[flashinfer:flashinfer/norm/__init__.py@L159|get_norm_module]] → [[flashinfer:flashinfer/jit/norm.py@L21|gen_norm_module]] → [[flashinfer:flashinfer/jit/core.py@L739|gen_jit_spec]] → [[flashinfer:csrc/norm.cu@L24|csrc/norm.cu]] → [[flashinfer:csrc/flashinfer_norm_binding.cu@L61|绑定]] → [[flashinfer:include/flashinfer/norm.cuh@L140|norm.cuh]]

**文档**：[FlashInfer 文档](https://docs.flashinfer.ai) · [TVM-FFI](https://tvm.apache.org/ffi/) · [FlashInfer 论文](https://arxiv.org/abs/2501.01005)
