lede: 按主题整理的高频面试题。每题先自己口头回答一遍（限时 2 分钟），再展开参考答案；括号里是本套题中对应的练习。
# 面试题库

## 硬件与性能模型

::: answer 1. 怎么判断一个 kernel 是 compute-bound 还是 memory-bound？（Part 0）
算算术强度 AI = FLOPs / 必须读写的 HBM 字节数，和硬件拐点（峰值算力 ÷ 峰值带宽，H100 FP16 约 295 FLOP/B，RTX 5090 约 117 FLOP/B）比较。低于拐点是 memory-bound，用有效带宽（GB/s）衡量；高于拐点是 compute-bound，用 TFLOP/s 衡量。再用 ncu 验证：看 Memory Throughput 和 SM Throughput 哪个接近 100%，以及 warp 在等什么（Long Scoreboard 表示在等访存）。
:::

::: answer 2. 什么是合并访存？什么是 bank conflict？各自怎么解决？（C3）
一个 warp 的 32 次访存会被合并成尽量少的 32 字节 sector：相邻线程访问相邻地址时效率最高，跨步访问会让事务数成倍增加。解决办法是调整线程到数据的映射，或者通过 shared memory 中转（转置）。shared memory 有 32 个 bank，同一 warp 访问同一 bank 的不同地址时会串行化；解决办法是加 padding（`[32][33]`）或对地址做 swizzle（异或），访问同一地址则是广播。
:::

::: answer 3. occupancy 越高越好吗？
不一定。occupancy 高可以隐藏延迟，但 GEMM、FlashAttention 这类 kernel 往往故意用很多寄存器和 shared memory 来提高数据复用（每个线程算更大的 tile），occupancy 只有 1–2 个 block / SM，靠指令级并行和软件流水线（cp.async / TMA 预取）来隐藏延迟。判断标准是实际吞吐，不是 occupancy 数字。
:::

::: answer 4. H100、B200、RTX 5090 在写 kernel 时最大的区别？（全景地图）
H100（sm_90a）引入 wgmma（warpgroup 级异步 MMA）、TMA、线程块集群，最优 kernel 普遍使用 warp specialization（生产者 warp 用 TMA 搬数据，消费者 warp 做 wgmma）。B200（sm_100）换成 tcgen05 MMA，累加器放在专门的 TMEM 里，支持 2-CTA MMA 和硬件 block-scaled FP4 / FP8。RTX 5090（sm_120）虽然也叫 Blackwell，但没有 tcgen05 / TMEM / wgmma，仍是 `mma.sync`（扩展了 FP4 / FP6 / FP8），所以为 sm_100 写的 FA4、DeepGEMM 不能直接运行。
:::

## CUDA 与 Triton

::: answer 5. 讲一下 reduction 的优化过程。（C2）
从 shared memory 树形归约开始（每轮 `__syncthreads`）；用 warp shuffle 取代 warp 内的那几轮（不需要同步、不经过 shared memory）；每个线程先用 grid-stride loop 在寄存器里累加多个元素，减少 block 数；跨 block 用一次 atomicAdd，或者用两趟 kernel 保证确定性。Triton 里一个 `tl.sum` 就会被编译成线程内累加、shuffle、shared memory 这三层。
:::

::: answer 6. GEMM 从 naive 到 cuBLAS，每一步优化了什么？（C4、T4）
naive 每次乘加都从 HBM 读两个数。shared memory tiling 让每个数在 block 内复用 tile 宽度次；register tiling 让每个线程计算一个小块，shared memory 的读取再减少 TM/TN 倍；然后是向量化加载、双缓冲 / 多级流水线（cp.async、TMA）、Tensor Core（mma / wgmma / tcgen05）、swizzle 消除 bank conflict、grouped ordering 提高 L2 命中，以及按形状 autotune。在 RTX 5090 上，本套题的 Triton GEMM（T4）从 1024³ 开始就和 cuBLAS 持平。
:::

::: answer 7. Triton 和 CUDA 怎么选？Triton 的局限是什么？
Triton 让你在 block 层面编程：shared memory、合并访存、流水线、Tensor Core 指令都由编译器决定，几十行就能写出接近 cuBLAS 的 GEMM，或者带宽跑满的融合算子，非常适合新算子和融合。局限在于对硬件的控制力：warp specialization、复杂的 TMA / 集群用法、精确的寄存器分配、非常规的数据布局，在 Hopper / Blackwell 上很难让编译器自动做到最好。这时就用 CUDA / CUTLASS / CuTe DSL，或者 Triton 的底层方言 Gluon（显式控制 layout 和数据搬运）。
:::

::: answer 8. Triton 的 num_warps 和 num_stages 分别控制什么？（T4）
num_warps 是每个 program（CTA）的 warp 数，决定一个块级张量被分给多少线程，以及每个线程持有多少寄存器。num_stages 是软件流水线深度：K 循环里的 load 会被提前几轮发出，经 shared memory 缓冲，计算和访存因此重叠。stage 越多预取越深，但每个 stage 都要占一份 shared memory，可能降低 occupancy，甚至超出容量。最优值依赖形状和硬件，所以用 autotune 来选。
:::

## Attention

::: answer 9. FlashAttention 为什么快？它的核心技巧？（T5）
它没有减少 FLOPs，减少的是 HBM 读写：通过 tiling 和 online softmax（维护行最大值 m 和分母 l，新块到来时用 exp(m_old − m_new) 重新缩放旧的累加值），S×S 的分数矩阵从不写回 HBM，IO 从 O(S²) 降到 O(S·D)。FA2 改进了并行划分（沿序列维度并行、减少非矩阵运算）；FA3 在 Hopper 上用 wgmma、TMA、warp specialization 和 FP8；FA4 在 Blackwell 上用 tcgen05 / TMEM，并用 CuTe DSL 实现。
:::

::: answer 10. decode attention 和 prefill attention 的瓶颈有什么不同？（L2、T5）
prefill 时 Q 有很多行，QKᵀ 是真正的矩阵乘，compute-bound，比拼 Tensor Core 利用率。decode 时每个序列只有一个 query，每个 KV token 只做约 4D 次运算却要读 2D 个数，是 memory-bound 的，时间约等于 KV cache 字节数 / 带宽。所以 decode 的优化方向是少读、快读：paged 布局、GQA 打包（每个 KV head 只读一次）、KV 量化（FP8 / FP4）、split-KV（让足够多的 SM 一起读）。
:::

::: answer 11. 什么是 split-KV / Flash-Decoding？什么时候用？（L3、F1）
batch 小、上下文长时，按 (序列, KV head) 划分的 program 数远少于 SM 数。split-KV 沿上下文维度再切成几段并行计算，每段输出 (o, lse)，最后用 LSE 加权精确合并。RTX 5090 上 B = 1、ctx = 32K 时，分 64 段快了 19 倍。batch 大时分段反而更慢（额外的中间结果写回和合并 kernel 成为纯开销），所以分几段要按形状决定：这正是 FlashInfer `plan()` 的工作。
:::

::: answer 12. 只给每段的输出 o_s，不给 lse，能合并吗？FlashInfer 的 LSE 有什么特别之处？（F2）
不能：合并需要每段 softmax 分母的相对大小，只有 o_s 时这个信息已经丢失了。合并公式是 w_s = exp(lse_s − max)，o = Σ w_s o_s / Σ w_s。FlashInfer 的 LSE 以 2 为底（log₂Σexp），kernel 内部全部使用 exp2 / log2，和自然对数版本混用会算出错误的权重。
:::

::: answer 13. GQA、MQA、MLA 对 KV cache 和 decode kernel 有什么影响？（L2、全景地图）
GQA / MQA 让多个 query head 共享一组 KV，KV cache 缩小 group 倍；decode kernel 让一个 program 处理同一 KV head 下的全部 query head，KV 只读一次，并可以把这些 query 组成矩阵、用上 Tensor Core。MLA（DeepSeek）把 KV 压缩成低秩 latent，cache 更小；decode 时把上投影吸收进 Q，直接在 latent 上做 attention（head_dim 512 + 64 的 RoPE 部分），算术强度更高，需要 FlashMLA、FlashInfer MLA 这样的专用 kernel。
:::

::: answer 14. PagedAttention 解决了什么问题？代价是什么？（L1、F1）
它把 KV cache 按固定大小的 page 分配，消除了按最大长度预留带来的碎片和浪费，让 KV 可以动态增长、在序列之间共享前缀（copy-on-write），并发量因此提高数倍。代价是 attention kernel 每读一个 page 都要先查一次页表（间接寻址），物理地址不连续。vLLM 用二维 block table，FlashInfer 用 CSR（indptr、indices、last_page_len），后者不需要按最长序列补齐。
:::

## FlashInfer 与 serving

::: answer 15. FlashInfer 的 plan / run 模式是什么？为什么这样设计？（F1）
plan() 在 CPU 上根据这一批序列的长度做调度：是否分段、分几段、每个 CTA 处理哪些 page、中间结果放在 workspace 的哪里，然后把元数据拷到 GPU。run() 只启动 kernel。一个 batch 的形状确定后 plan 只做一次，所有层复用；run() 形状固定、没有 CPU 同步，可以被 CUDA graph 捕获。
:::

::: answer 16. 为什么采样 kernel 不排序？（F4）
top-k / top-p 只需要一个阈值。FlashInfer 反复对“大于 pivot 的元素”做 (质量, 个数) 的归约来移动 pivot，每轮一次全词表扫描，几轮就收敛，还能按请求使用不同的参数，并直接在同一个 kernel 里完成采样。RTX 5090 上 64 个请求、15 万词表时，`torch.sort` 本身就要 1 ms，FlashInfer 的 top-k renorm 只要 88 us。
:::

::: answer 17. decode 阶段为什么要融合 RMSNorm、残差加、SwiGLU 这些小算子？（F3）
decode 时每个 kernel 的工作量都很小，几微秒的固定开销（launch、调度、一次读写）就占了大头。一个 decoder layer 有十来个这样的小算子，不融合时它们的总开销可以和 GEMM 相当。融合（fused_add_rmsnorm、silu_and_mul、norm + 量化、RoPE + 写 KV cache）、CUDA graph 和 PDL 都是在减少这部分开销。
:::

::: answer 18. 你会怎么给 FlashInfer 贡献一个新算子？（F5）
kernel 写在 include/（只用裸指针，不 include Torch），launcher 写在 csrc/，加 TVM-FFI 绑定，在 flashinfer/jit/ 中写 JIT 生成器，再写 Python API（@functools.cache 缓存模块）、测试、AOT 注册、导出和 trace 模板。性能 PR 要给出改动前后的绝对数字，并写明 GPU 和问题规模；保留默认的 PR 模板；CI 用 `@flashinfer-bot run tests/<路径>` 只跑相关测试。消费级 GPU 的 kernel 走 experimental 路线。
:::

## MoE 与低精度

::: answer 19. fused MoE kernel 是怎么工作的？（M1、M2）
先路由（top-k 专家及其权重），再把 (token, slot) 对按专家排序，并把每个专家的那一组补齐到 BLOCK_M 的整数倍（moe_align_block_size）。然后一次 grouped GEMM 的 launch：每个 M 块属于一个专家，A 的行通过 sorted_token_ids 间接读取，B 用这个专家的权重；路由权重乘进 epilogue；最后把 k 个专家的结果求和。prefill 时是 compute-bound 的，decode 时是读权重的 memory-bound。
:::

::: answer 20. DeepSeek-V3 的 FP8 block scaling 是怎么回事？为什么不用 per-tensor？（M3）
FP8 e4m3 的动态范围很小，per-tensor 的 scale 会让离群值把其他数压没。DeepSeek-V3 对激活每 1×128 取一个 scale，对权重每 128×128 取一个；每个 128 宽的 K 块做一次 FP8 Tensor Core 矩阵乘，再乘上这一块的 sa·sb，fp32 累加。粒度和 K 块一致，所以几乎没有额外开销。本套题在 RTX 5090 上的 Triton 实现达到 bf16 cuBLAS 的 1.7 倍。Blackwell 的 MXFP8 / NVFP4 把这种块缩放做进了硬件指令。
:::

::: answer 21. DeepGEMM 的 contiguous 和 masked 两种 grouped 布局分别给什么场景用？
contiguous：各专家的 token 连续排列（按专家补齐到对齐边界），适合 prefill 和训练，token 数在 CPU 上已知。masked：为每个专家预留固定的最大行数，用一个张量记录实际行数，kernel 内部屏蔽多出的部分，形状固定，适合 decode 阶段配合 CUDA graph（每步每个专家的 token 数在 GPU 上才知道）。
:::

## 不同模型、不同硬件

::: answer 22. 扩散模型（视频 DiT）的推理瓶颈和 LLM 有什么不同？（全景地图）
没有 KV cache 和 decode，每个去噪步都是对全部 token 的一次 compute-bound 前向；视频的序列长度可达数万到数十万，attention 往往占一半以上的时间。有效手段是最好的 FlashAttention 代际、量化 attention（SageAttention：INT8 / FP8，SageAttention3 在 RTX 5090 上用 FP4）、利用时空局部性的稀疏 attention、融合 adaLN，以及序列并行（Ulysses / ring）。端到端收益要算全：步数蒸馏和特征缓存的影响通常更大。
:::

::: answer 23. VLA 模型（比如 π0）在机器人上部署，kernel 层面最该优化什么？
batch = 1、对延迟有硬性要求，瓶颈往往不是某一个大 kernel，而是上百个小 kernel 的 launch 开销和 CPU 调度（包括 flow matching 的多步去噪循环）。最有效的组合是：用 Triton 融合小算子、预分配静态 buffer、把整条前向（含去噪循环）捕获进一个 CUDA graph；再根据端侧硬件（Jetson Thor / Orin）用 TensorRT 和低比特量化。
:::

::: answer 24. 在 Mac 上用 MLX 跑 LLM，性能主要由什么决定？（X1–X3）
decode 是 memory-bound 的：每个 token 读一遍权重，tok/s ≈ 带宽 ÷ 模型字节数，所以 4-bit 分组量化（MLX 的 qmv / qmm）是最有效的手段。prefill 受 GPU 矩阵吞吐限制（simdgroup_matrix 8×8；M5 起还有 neural accelerator）。统一内存免去了拷贝，但带宽只有数据中心 GPU 的几分之一到几十分之一。
:::

::: answer 25. 同一个模型在 H100 和 AMD MI300X 上，kernel 栈有什么不同？
NVIDIA 上：FlashAttention-3 / FlashInfer、cuBLAS / CUTLASS / DeepGEMM、Triton。AMD 上：AITER（Triton + Composable Kernel + 手写汇编，提供 attention、fused MoE、MLA、GEMM），配合 vLLM / SGLang 的 ROCm 后端（如 `VLLM_ROCM_USE_AITER=1`）；Triton 的 ROCm 后端让同一份 Triton kernel 也能运行，但最优配置（tile 大小、wave 数）不同。MI300X 显存大（192 GB），单卡能放下更大的模型或更多 KV。
:::
