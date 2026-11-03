lede: 先看这一节：整套路线怎么排、每道题怎么做、在哪台机器上跑，以及做完之后你应该能做到什么。
# 路线与用法

## 目标

做完这套题，你应该能做到三件事：

1. **读懂并写出主流 kernel**：用 CUDA C++、Triton、CuTe 和 MLX / Metal 各写过一遍 elementwise、reduction、GEMM、attention、MoE，知道每一层优化在解决什么问题，并能用 roofline 说清楚离硬件上限还有多远。
2. **对付面试**：说得清 FlashAttention、paged attention、split-KV、GQA、MoE grouped GEMM、FP8 block scaling 的原理和取舍，以及**不同模型、不同硬件上哪个 kernel 是当前的最佳选择**、为什么。
3. **给 FlashInfer 提 PR**：知道它的源码结构、plan/run 模式、JIT 系统、测试和 benchmark 规范，并且已经亲手实现过它的四个 API（F1–F4），能和官方实现逐数值对拍、比较性能。

## 路线：核心 → 进阶

```text
Part 0  基础 + 全景地图（模型 × 硬件 × 当前最优 kernel）
   │
核心 ├─ C  CUDA C++      C1–C4  线程映射、合并访存、warp 原语、GEMM tiling
     ├─ T  Triton        T1–T5  block 级编程、融合、tl.dot、FlashAttention
     ├─ L  推理 attention L1–L3  paged KV、decode + GQA、split-KV（从零手写）
     └─ F  FlashInfer    F1–F5  CSR 分页格式、merge_state、融合 norm、采样、贡献实战
   │
进阶 ├─ G  Layout / CuTe G1–G2  BlockedLayout / SliceLayout，CuTe 布局代数
     ├─ M  MoE 与量化     M1–M3  路由对齐、fused MoE grouped GEMM、FP8 block scaling（DeepGEMM 配方）
     └─ X  MLX / Metal   X1–X3  Apple GPU 上的 elementwise、simdgroup 归约、simdgroup_matrix GEMM
   │
面试题库 · 附录（下一步、阅读路线）
```

**为什么这样排**：FlashInfer 的 kernel 本身就是 CUDA C++（以及越来越多的 CuTe DSL），它的 Python 生态、vLLM / SGLang 的大量 kernel 又是 Triton。所以先用 C、T 打基础，再用 L 从零写出 paged attention，最后在 F 里对着 FlashInfer 的真实接口实现并对拍。进阶部分覆盖面试和实际工作中同样常见的其他方向：CuTe（CUTLASS 3/4、FlashAttention-4、DeepGEMM 都基于它）、MoE 与低精度、Apple 设备上的 MLX。

## 每道题的结构

| 小节 | 作用 |
|---|---|
| **背景** | 这个 kernel 在真实系统里出现在哪，为什么值得写 |
| **要用到的知识** | 只讲这题需要的新语法和硬件概念 |
| **题目要求** | 函数签名、shape / dtype、边界条件、验收标准 |
| **题干** | `mine/` 里的骨架文件全文：你只填 `TODO` |
| **验收** | 测试命令和测试代码。F 题额外和 FlashInfer 官方实现对拍 |
| **提示 L1 / L2 / L3** | 卡住 10–15 分钟再按级别展开；L3 基本就是答案 |
| **参考解** | 每一行都有注释的完整实现，写完再看 |
| **测完之后** | 实测数字（RTX 5090 / Apple M2）、可以做的实验、面试追问 |

## 仓库与命令

```bash
./check.sh f2                # 测你写的 mine/f2_merge_state.py
./check.sh f2 --solutions    # 测参考解（确认环境没问题）
./check.sh f2 --bench        # 性能：你的实现 vs FlashInfer / PyTorch（需要 NVIDIA GPU）
./check.sh x2                # X 题自动使用 .venv-mlx（Apple silicon）
./check.sh all               # 全部
```

## 在哪跑

| 环境 | 能做什么 |
|---|---|
| **只有 CPU** | T、L、F1–F4、G、M1–M2 的正确性都能验收：没有 GPU 时自动打开 `TRITON_INTERPRET=1`。macOS 上没有 triton wheel，可以用 Linux 容器。 |
| **自己的 NVIDIA GPU**（比如 RTX 5090） | 全部 C / T / L / F / M 题和 benchmark。已有装了 torch + triton 的 Python 即可，不需要系统级 CUDA toolkit：`./setup_env.sh /path/to/python`，以后每个新 shell 先 `source env.sh`。FlashInfer：`pip install flashinfer-python`。 |
| **Apple silicon Mac** | X 题：`python3 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx numpy pytest`（必须是 arm64 的 Python）。 |
| **Colab T4 / 云 GPU** | T4 是 sm_75：bf16 和 FP8 测试会自动跳过（FP8 需要 sm_89+）。H100 / B200 上才能跑 FlashAttention-3/4、DeepGEMM、wgmma / tcgen05 相关内容。 |

页面里的“实测”数字来自：RTX 5090（sm_120）+ torch 2.13（CUDA 13.0）+ Triton 3.7 + FlashInfer 0.7.2；Apple M2（24 GB）+ MLX 0.32。

## 速成路线（约 24 小时）

| 次 | 内容 | 时长 |
|---|---|---|
| 1 | Part 0 基础 + 全景地图 | 2 h |
| 2 | C1 + C2 + C3 | 2 h |
| 3 | C4 | 1.5 h |
| 4 | T1 + T2 + T3 | 1.5 h |
| 5 | T4 + T5 | 2.5 h |
| 6 | L1 + L2 + L3 | 2.5 h |
| 7 | F1 + F2 + F3 | 2 h |
| 8 | F4 + F5（贡献实战，可以拆成多次） | 3 h |
| 9 | G1 + G2 | 1.5 h |
| 10 | M1 + M2 + M3 | 3 h |
| 11 | X1 + X2 + X3 | 2.5 h |

**只有一周时**：Part 0 → T1 → T2 → T5 → L2 → L3 → F1 → F2 → F4 → M3，再读一遍面试题库。
