lede: 在写任何一道题之前先建立地图：一个模型的时间花在哪类 kernel 上，这类 kernel 在每种硬件上当前用什么实现最好，以及为什么。
# 全景地图：模型 × 硬件 × 当前最优 kernel

## 先把 kernel 分类

几乎所有推理 / 训练的时间都落在下面几类 kernel 上。后面所有“谁最优”的判断，都是在问：**这个模型、这个阶段，时间主要花在哪一类上？这一类在这块硬件上是 compute-bound 还是 memory-bound？**

| 类别 | 例子 | 通常的瓶颈 | 本套题 |
|---|---|---|---|
| **稠密 GEMM** | QKV / O / MLP 投影、LM head | prefill 和大 batch 时 compute-bound；小 batch decode 时读权重，memory-bound | C4、T4 |
| **低精度 GEMM** | FP8（per-tensor / block-scaled）、NVFP4 / MXFP4、W4A16 | 同上，但每次 MMA 的吞吐翻倍、权重字节减半 | M3 |
| **grouped GEMM（MoE）** | 每个 expert 各算一小块 | 每个 expert 的 M 很小、不均匀；路由和排序本身也有开销 | M1、M2 |
| **attention：prefill** | FlashAttention 类 | compute-bound（S² 的 FLOPs） | T5 |
| **attention：decode** | paged decode、split-KV、MLA decode | memory-bound（读 KV cache） | L1–L3、F1、F2 |
| **线性 attention / SSM** | Mamba、GDN（Qwen3-Next）、KDA（Kimi） | chunked scan；decode 是状态更新 | 附录 |
| **融合的小算子** | RMSNorm + 残差、RoPE、SwiGLU、量化、adaLN modulate | memory-bound，次数多；不融合就被 launch 开销和多次读写拖慢 | T1–T3、F3 |
| **采样** | top-k / top-p / min-p、投机解码验证 | 词表 15 万：排序是瓶颈 | F4 |
| **通信** | all-reduce、MoE 的 all-to-all（DeepEP） | 互联带宽；要和计算重叠 | 附录 |

## 一个 LLM 推理 step 的时间去了哪

- **prefill**（一次处理整段 prompt）：GEMM 占大头，attention 随 S² 增长，长 prompt 时 attention 可以超过一半。两者都是 compute-bound，此时比拼 Tensor Core 的利用率。
- **decode**（每步每个序列 1 个 token）：
  - 小 batch：GEMM 退化成矩阵乘向量，时间 ≈ 读一遍全部权重。量化权重（W4A16、FP8、NVFP4）直接按比例省时间。
  - 大 batch、长上下文：KV cache 的读取量超过权重，**decode attention** 变成主要开销。paged attention、GQA / MLA、KV 量化、split-KV 都在解决这件事。
  - batch = 1 的极端情况（比如 VLA、端侧）：每个 kernel 只跑几微秒，**launch 开销和 CPU 调度**反而成为瓶颈，所以要靠 CUDA graph 和更激进的融合。
- **MoE 模型**：GEMM 被拆成几十上百个 expert 的小 GEMM。专家并行（EP）时还要加上 all-to-all 通信。

## 按硬件：先看硬件给了什么

| 硬件 | 关键特性 | 这一代的代表性 kernel |
|---|---|---|
| **A100**（sm_80） | `mma.sync` Tensor Core、`cp.async`；没有 FP8 | FlashAttention-2、FlashInfer（FA2 后端）、cuBLAS / CUTLASS 2.x、Triton |
| **RTX 4090 / L40S**（sm_89） | 加入 FP8 `mma.sync`；没有 wgmma / TMA | FA2 类 kernel；FP8 GEMM |
| **H100 / H200**（sm_90a） | **wgmma**（warpgroup 级异步 MMA）、**TMA**（张量内存加速器）、线程块集群、warp specialization | **FlashAttention-3**、FlashInfer 的 FA3 后端、**DeepGEMM**、CUTLASS 3（CuTe）、FlashMLA |
| **B200 / GB200 / B300**（sm_100 / sm_103） | **tcgen05** MMA + **TMEM**（张量内存）、2-CTA MMA、硬件 block-scaled FP8 / FP6 / FP4（MXFP8、NVFP4） | **FlashAttention-4**（CuTe DSL）、TRT-LLM 生成的 kernel（FlashInfer 的 `trtllm-gen` 后端）、cuDNN attention、DeepGEMM（sm_100）、CUTLASS 4 / CuTe DSL |
| **RTX 5090 / RTX PRO 6000**（sm_120，消费级 Blackwell） | **没有** tcgen05 / TMEM / wgmma；`mma.sync` 扩展支持 FP8 / FP6 / FP4（含 block scale） | FA2 风格的 kernel、Triton、**SageAttention3**（FP4 attention）、FlashInfer 的 sm_120 专用 kernel（如 `nvfp4_attention_sm120`，在 FlashInfer 里属于 experimental 后端）；DeepGEMM 不支持 |
| **Jetson Thor**（sm_110）/ **Orin**（sm_87） | 端侧，功耗受限；统一内存 | TensorRT；FlashInfer 已有 sm_110 的 GQA decode kernel |
| **AMD MI300X / MI325X / MI355X**（gfx942 / gfx950） | MFMA 矩阵指令；MI355X 加入 FP4 / FP6 | **AITER**（Triton + Composable Kernel + 手写汇编：attention、fused MoE、MLA、GEMM）、CK、HipKittens；vLLM / SGLang 通过 `VLLM_ROCM_USE_AITER=1` 等开关启用 |
| **Apple M 系列** | 统一内存（带宽约 100–800 GB/s）；simdgroup（32 线程）、`simdgroup_matrix` 8×8 MMA；M5 起 GPU 内有 neural accelerator | **MLX**：steel GEMM / attention、decode 专用的 `sdpa_vector`、分组量化的 `qmv / qmm`、MoE 用的 `gather_mm / gather_qmm`，以及面向 M5 neural accelerator 的 `*_nax` kernel；自定义 kernel 用 `mx.fast.metal_kernel`（X 系列） |
| **TPU / Trainium** | 脉动阵列 | Pallas（JAX）、NKI：本套题不覆盖 |

**两个容易记错的点**：

- RTX 5090 叫 Blackwell，但它的指令集（sm_120）和 B200（sm_100）**不一样**：没有 tcgen05 和 TMEM，所以 FlashAttention-4、DeepGEMM 这类为 sm_100 写的 kernel 不能直接跑。在 5090 上写 kernel，更接近“带 FP4 的 Ada”。FlashInfer 的贡献指南也把 SM12x 归为 experimental 后端。
- 越新的硬件，最优 kernel 越依赖专门的指令（wgmma、TMA、tcgen05）和 warp specialization。这也是 CUTLASS / CuTe DSL、Gluon 这类“能显式控制硬件”的工具变得重要的原因：Triton 的自动化在 Hopper / Blackwell 上更难追平手写 kernel。

## 按类别：当前的常见最优选择

这张表描述的是 2026 年下半年 vLLM / SGLang 这类主流 serving 框架的常见默认或推荐选择，变化很快，用之前请查对应框架的文档。

| 类别 | H100 | B200 | RTX 5090 | AMD MI3xx | Apple |
|---|---|---|---|---|---|
| prefill attention | FA3；FlashInfer（FA3 后端） | FA4；trtllm-gen / cuDNN（经 FlashInfer） | FA2 类；FlashInfer FA2 后端；SageAttention | AITER / CK | MLX steel attention |
| decode attention（GQA，分页） | FlashInfer decode / FA3 | trtllm-gen（经 FlashInfer）；XQA | FlashInfer FA2 后端 | AITER paged attention | MLX `sdpa_vector` |
| MLA（DeepSeek-V3 类） | FlashMLA；FlashInfer MLA | FlashInfer（CUTLASS / trtllm-gen MLA） | 通用实现 | AITER MLA | — |
| 稠密 GEMM | cuBLAS / CUTLASS；DeepGEMM（FP8 block） | cuBLAS / CUTLASS（NVFP4 / MXFP8）；DeepGEMM | cuBLAS；Triton | hipBLASLt；AITER | MLX steel GEMM |
| MoE grouped GEMM | Triton fused_moe；DeepGEMM（FP8 grouped） | FlashInfer 的 TRT-LLM / CUTLASS / CuTe-DSL MoE | Triton fused_moe | AITER fused MoE | MLX `gather_mm` / `gather_qmm` |
| 小算子融合 | FlashInfer norm / RoPE / activation；Triton | 同左 | 同左 | AITER；Triton | MLX `mx.fast.*` |
| 采样 | FlashInfer sampling | 同左 | 同左 | — | MLX |

## 按模型领域

### Dense LLM（GQA）

- prefill：attention 和 GEMM 都是 compute-bound，选这块硬件上最好的 FlashAttention 代际（A100 用 FA2，H100 用 FA3，B200 用 FA4 或 trtllm-gen）。
- decode：paged decode attention（L 系列、F1），长上下文配 split-KV（L3），GQA 打包让每个 KV 头只读一次（L2）。
- 周边：融合 RMSNorm + 残差（F3）、融合 RoPE + 写 KV cache（FlashInfer 有 `fused_qk_rmsnorm_rope`）、SwiGLU（F3）、不排序的采样（F4）。
- 量化：Hopper 上常用 FP8（per-tensor 或 block-scaled），Blackwell 上 NVFP4 / MXFP4；小 batch decode 时权重量化收益最大。

### MLA 模型（DeepSeek-V3/R1、Kimi K2 等）

MLA 把 K、V 压缩到一个低秩 latent 里，KV cache 小很多。decode 时，可以把 latent 的上投影吸收进 Q，attention 直接在 latent 上做（head_dim 512 + 64 的 RoPE 部分）。这使 decode attention 的算术强度比 GQA 高，需要专门的 kernel：FlashMLA（Hopper）、FlashInfer 的 `BatchMLAPagedAttentionWrapper`。

### MoE 模型（DeepSeek-V3、Qwen3-MoE、gpt-oss、Kimi）

- **路由与排列**：top-k 选专家、把 token 按专家分组并补齐到 tile 大小（M1）。
- **grouped GEMM**：一次 launch 算完所有专家（M2）。prefill 用 contiguous 布局（各专家的 token 连续排列）；decode 要配合 CUDA graph，用 masked 布局（预分配好每个专家的最大行数，按实际数量屏蔽）。DeepGEMM 同时提供这两种布局。
- **低精度**：DeepSeek-V3 的 FP8 训练和推理使用 1×128（激活）和 128×128（权重）的 block scale（M3），NVFP4 MoE 则是 Blackwell 上的主流。
- **通信**：专家并行时，dispatch / combine 的 all-to-all 用 DeepEP（基于 NVSHMEM）这类库，并尽量与计算重叠。DeepGEMM 2026 年 4 月加入了把 dispatch、两层 FP8×FP4 GEMM、combine 融合在一起的 Mega MoE kernel。

### 线性 attention / 混合架构（Qwen3-Next、Kimi Linear、Mamba 类）

把 attention 换成状态大小固定的递推（GDN、KDA、Mamba2），长上下文时 decode 的开销不再随长度增长。kernel 主要来自 flash-linear-attention（Triton）项目，FlashInfer 0.7 也已经加入 GDN / KDA 的 prefill 与 decode kernel。训练和 prefill 用分块并行的 chunked scan，decode 是逐 token 的状态更新，同样需要融合。

### 扩散模型（DiT 图像 / 视频）

- 没有 KV cache，也没有 decode：每个去噪步都是对全部 token 的一次 **compute-bound** 前向，视频模型的序列长度动辄数万到数十万，**attention 往往占一半以上的时间**。
- 当前最有效的 kernel 方向：
  - 最好的 FlashAttention 代际（H100 用 FA3，B200 用 FA4）；
  - **量化 attention**：SageAttention 系列（INT8 / FP8；SageAttention3 在 RTX 5090 上用 FP4，作者报告 attention kernel 约是 FlashAttention2 的 5 倍，HunyuanVideo 端到端约 3 倍）；
  - **稀疏 attention**：滑动 tile、radial、VSA 一类利用视频时空局部性的方法；
  - **融合 adaLN**：DiT 每层都有 `norm → scale/shift → gate → residual`，FlashInfer 已有 `fused_dit_*` 系列算子；
  - 低比特 GEMM：SVDQuant / Nunchaku 的 W4A4；
  - 多卡：xDiT 的 Ulysses / ring 序列并行；VAE 解码中的 3D 卷积依赖 cuDNN。
- 注意端到端收益：去噪步数的蒸馏、特征缓存（TeaCache 等）往往比任何单个 kernel 的提升都大。也有实际工作流中 attention 只占很小一部分的例子：kernel 快 4 倍，整体只快 2%。**先 profile，再决定优化哪个 kernel。**

### Omni / 语音 / 多模态（Qwen3-Omni 类）

- 多个阶段串联：视觉 / 音频编码器（ViT、卷积 + Transformer）→ LLM（thinker）→ 语音生成（talker）→ codec 解码。
- 编码器是 compute-bound 的变长 attention（FlashAttention varlen），文本主干就是普通的 LLM 路径，外加多维 RoPE（M-RoPE）。
- talker 和 codec 每步只生成很少的数据，**小 kernel、batch 小，以 launch 开销为主**：CUDA graph、流式因果卷积、跨阶段的 chunk 流水才是关键。

### VLA / 机器人

- batch = 1、对延迟有硬性要求（几十 Hz 的控制频率）。π0 这类模型由 VLM 主干和一个用 flow matching 多步去噪生成动作块的 action expert 组成。
- 瓶颈通常不是某一个大 kernel，而是**上百个小 kernel 的 launch 开销和 CPU 调度**。有效手段几乎总是同一个组合：用 Triton 融合小算子、预分配静态 buffer，把整条前向（包括去噪循环）捕获进一个 CUDA graph。有研究报告，仅去掉 CPU 开销就能带来约 2 倍的加速。
- 端侧部署（Jetson Thor / Orin）再加上 TensorRT 和低比特量化。

## 选型的几条经验法则

1. **先判断 compute-bound 还是 memory-bound**（Part 0 的 roofline）：前者比拼 Tensor Core 利用率和新指令；后者比拼少读字节（融合、量化、GQA / MLA、KV 压缩）。
2. **batch 很小时先看 launch 开销**：kernel 本身只要几微秒时，CUDA graph 和融合比再快 10% 的 kernel 更重要。
3. **硬件代际决定可选项**：B200 上最好的 kernel 在 5090 上可能根本跑不了，反过来也一样。看 sm 版本，而不是看产品名。
4. **库的选择大于手写**：FlashInfer、cuBLAS / CUTLASS、DeepGEMM、AITER、MLX 已经覆盖了绝大多数热点。值得手写的是**新模型带来的新算子**（新的 attention 变体、新的量化格式、新的融合模式），这也是向 FlashInfer 这类库贡献的主要方向。

## 来源

- FlashAttention-4（Blackwell，CuTe DSL）：[论文 arXiv 2603.05451](https://arxiv.org/pdf/2603.05451.pdf) · [MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf)
- DeepGEMM（SM90 / SM100，FP8 / FP4，grouped，Mega MoE）：[文档](https://docsearch.algolia.com/mcp/docs/repo/deepseek-ai/deepgemm)
- SageAttention3（FP4 attention，RTX 5090 实测）：[论文 arXiv 2505.11594](https://huggingface.co/papers/2505.11594) · [GitHub](https://github.com/thu-ml/sageattention) · [一个端到端收益很小的反例](https://ai-muninn.com/en/blog/minimax-h3-rtx5090-config-stack)
- AMD AITER 与 vLLM / SGLang：[ROCm blog: vLLM on MI355X](https://rocm.blogs.amd.com/artificial-intelligence/scaling-ai-inference/README.html) · [MLPerf v5.1 技术解读](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html)
- VLA 实时推理：[Running VLAs at Real-time Speed（arXiv 2510.26742）](https://arxiv.org/pdf/2510.26742) · [FluxVLA 推理加速文档](https://fluxvla.limxdynamics.com/md_source/tutorials/inference/triton_inference.html)
- FlashInfer 的 API 和后端名称（trtllm-gen、cudnn、cutlass、cute_dsl、sm_110 / sm_120 kernel、GDN / KDA、fused DiT norm）：直接来自 `flashinfer-python 0.7.2` 的 Python API 和 2026-11-02 的源码。
