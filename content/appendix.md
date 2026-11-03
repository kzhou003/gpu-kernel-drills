lede: 做完这些题之后：继续深入的方向、源码阅读路线和参考资料。

## 继续深入的方向

| 方向 | 建议的下一步 |
|---|---|
| Tensor Core（CUDA） | 用 `nvcuda::wmma` 改写 C4，在 FP16 下对比 cuBLAS；再读 CUTLASS 的 `cute` 入门例子（MMA atom、TiledMMA） |
| CuTe DSL | 在 RTX 5090 上跑通 CUTLASS `examples/python/CuTeDSL/` 中的 elementwise 和 GEMM 示例，然后读 FlashInfer 的 `flashinfer/cute_dsl/` 中的一个 kernel |
| Hopper / Blackwell 专用指令 | 有 H100 / B200 时：读 FlashAttention-3（`hopper/`）和 FlashAttention-4（`flash_attn/cute/`）、DeepGEMM 的主循环，理解 TMA + wgmma / tcgen05 + warp specialization 的分工 |
| Gluon | Triton 官方的 Gluon tutorial：从 layout 和 shared memory 开始，最后一课是 warp-specialized GEMM。把 T4 改写成 Gluon，对比生成的 `ttgir` |
| FlashAttention backward | 在 T5 中保存 lse，按 FlashAttention-2 论文的 Algorithm 2 实现 dQ / dK / dV |
| 线性 attention | 读 flash-linear-attention 中 GDN 的 chunked kernel，再看 FlashInfer 0.7 的 `gdn_prefill` / `gdn_fused_decode_step` |
| 通信 | DeepEP 的 dispatch / combine，NCCL 的 all-reduce 算法，以及 FlashInfer `comm/` 中把 all-reduce 和 RMSNorm 融合在一起的 kernel |
| MLX | 读 `mlx/backend/metal/kernels/steel/gemm/` 和 `sdpa_vector.h`；给 MLX 写一个它还没有的融合算子（例如带量化的 RMSNorm），用 `mx.fast.metal_kernel` 先实现原型 |

## 源码阅读路线（按本套题的顺序）

| 题目 | FlashInfer | 其他 |
|---|---|---|
| T2、T3、F3 | `include/flashinfer/norm.cuh`、`activation.cuh`；`flashinfer/triton/norm.py` | vLLM `csrc/layernorm_kernels.cu` |
| T5 | `include/flashinfer/attention/prefill.cuh` | Triton 官方 `06-fused-attention.py`；flash-attention 仓库 |
| L1、F1 | `include/flashinfer/page.cuh`；`flashinfer/page.py`（`append_paged_kv_cache`） | vLLM `csrc/cache_kernels.cu`（`reshape_and_cache_flash`） |
| L2、L3、F1 | `include/flashinfer/attention/decode.cuh`、`scheduler.cuh`；`flashinfer/decode.py` 的 `plan` | vLLM `csrc/attention/paged_attention_v2.cu` |
| F2 | `include/flashinfer/attention/cascade.cuh`；`flashinfer/cascade.py` | |
| F4 | `include/flashinfer/sampling.cuh`、`topk.cuh`；`flashinfer/sampling.py` | |
| M1、M2 | `flashinfer/fused_moe/`、`cutlass_fused_moe` | vLLM `model_executor/layers/fused_moe/fused_moe.py`、`csrc/moe/moe_align_sum_kernels.cu` |
| M3 | `flashinfer/deep_gemm.py`、`gemm/` 中的 FP8 groupwise GEMM | DeepGEMM 仓库；vLLM `w8a8_block_fp8_matmul` |
| G2 | `flashinfer/cute_dsl/` | CUTLASS `python/pycute/`、`include/cute/layout.hpp` |
| X1–X3 | — | MLX `mlx/backend/metal/kernels/`（`softmax.h`、`steel/`、`quantized.h`、`sdpa_vector.h`） |

## 参考资料

| 资料 | 用在哪 |
|---|---|
| CUDA C++ Programming Guide：Programming Model、Memory Hierarchy、Performance Guidelines | Part 0、C 系列 |
| Mark Harris, “Optimizing Parallel Reduction in CUDA” | C2 |
| Simon Boehm, “How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance” | C4 |
| Triton 官方 tutorials（Vector Addition、Fused Softmax、Layer Norm、Matrix Multiplication、Fused Attention）、Triton Puzzles | T 系列 |
| FlashAttention 1 / 2 / 3 论文，FlashAttention-4（arXiv 2603.05451，MLSys 2026） | T5、全景地图 |
| Kwon et al., PagedAttention（vLLM，SOSP 2023）；Flash-Decoding（2023 博客） | L 系列 |
| FlashInfer 论文（MLSys 2025）、文档、仓库中的 `CONTRIBUTING.md` 与 `CLAUDE.md` | F 系列 |
| DeepSeek-V3 技术报告（FP8 训练与推理）；DeepGEMM、DeepEP 仓库 | M 系列 |
| CuTe 文档（CUTLASS 仓库 `media/docs/cpp/cute/`）、CuTe DSL 文档 | G2 |
| MLX 文档 “Custom Metal Kernels”；Metal Shading Language Specification | X 系列 |
| SageAttention 1–3 论文；SVDQuant；xDiT | 全景地图：扩散 |
| PMPP（Programming Massively Parallel Processors） | 随用随查 |

## 每题完成后的记录模板

```markdown
## F2 · merge_state
- 环境：RTX 5090 / CUDA 13.0 / torch 2.13 / triton 3.7 / flashinfer 0.7.2
- 正确性：与 flashinfer.merge_state 对拍，max abs err = …
- 性能：S=8192, H=32, D=128 fp16：yours … us（… GB/s），flashinfer … us
- 瓶颈：memory-bound；ncu Memory Throughput …%
- 实验：每个 program 处理 2 个 head：…
- 结论 / 下一步：…
```
