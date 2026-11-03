"""Second wave: one more diagram (aimed at the hardest detail) + analogies per page.
Appended to the end of each page's '## 图解与类比' section; intro gets a new '## 路线图' section."""
from pathlib import Path

C = Path(__file__).resolve().parent.parent / "content"
MARK = "<!-- wave2 -->"


def flow(title, body):
    return f"::: flow {title}\n{body.strip()}\n:::\n"


def analogy(title, body):
    return f"::: callout analogy {title}\n{body.strip()}\n:::\n"


F = {}

F["intro"] = flow("整套路线", """
cfg cw=176 nw=146
n p0 0 1 "Part 0" "基础 + 全景地图" k=dark
n c 1 0 "C · CUDA C++" "C1–C4" k=acc
n t 2 0 "T · Triton" "T1–T5" k=acc
n l 3 0 "L · 推理 attention" "L1–L3" k=vio
n f 4 0 "F · FlashInfer" "F1–F5 · 对拍 + 贡献" k=pass w=1.15
n g 2 2 "G · Layout / CuTe" "G1–G2" k=teal
n m 3 2 "M · MoE / FP8" "M1–M3" k=warn
n x 4 2 "X · MLX / Metal" "X1–X3" k=todo
n iv 5.2 1 "面试题库" "+ 附录" k=dark
g 1 0 4 0 "核心"
g 2 2 4 2 "进阶"
e p0 -> c
e c -> t
e t -> l
e l -> f
e t -> g dash
e g -> m dash
e m -> x dash
e f -> iv
e x -> iv dash
cap: 实线是建议的主线顺序；虚线的进阶部分可以穿插在核心之后的任何时间做。
""")

F["basics_hw"] = flow("occupancy：资源决定能同时驻留多少 warp", """
n reg 0 0 "每线程寄存器数" "ncu: Registers" k=acc
n smem 0 1 "每 block shared memory" "ncu: Shared Memory" k=acc w=1.1
n bs 0 2 "block 大小" "线程数" k=acc
n lim 1.4 1 "每个 SM 能放几个 block" "取三者最紧的那个" k=warn w=1.2
n warps 2.8 1 "驻留 warp 数" "= occupancy × 上限" k=vio
n hide 4 1 "隐藏访存延迟" "warp 等数据时换人" k=pass
e reg -> lim
e smem -> lim
e bs -> lim
e lim -> warps
e warps -> hide
""") + analogy("合并访存像拼车", """
一个 warp 发出一次 load，就像 32 个人一起叫车：如果大家住同一栋楼（地址连续），4 辆车（4 个 32 字节 sector）就能全部拉走；如果 32 个人散住在城市各处（跨步访问），就要派 32 辆车，每辆只坐一个人。车（带宽）的数量是固定的，坐不满就是浪费。
""")

F["basics_cuda"] = flow("shared memory 的标准节奏", """
n ld 0 0 "协作加载 tile" "每个线程搬一部分" k=acc
n s1 1.15 0 "__syncthreads()" "等所有人搬完" k=todo
n cmp 2.3 0 "计算" "读别人搬来的数据" k=pass
n s2 3.45 0 "__syncthreads()" "等所有人用完" k=todo
e ld -> s1
e s1 -> cmp
e cmp -> s2
e s2:b -> ld:b "下一个 tile" anim
cap: 第二个 sync 很容易漏：没有它，跑得快的线程会在别人还在读的时候，把下一个 tile 写进同一块 shared memory。
""") + analogy("atomicAdd 像只有一个窗口的银行", """
很多人往同一个账户存钱，只有一个柜员窗口（同一地址上的原子操作会串行）：人越多队越长。所以先让每个小组把钱凑齐（block 内归约），每组只派一个人去排队。
浮点数的“零头”跟存钱顺序有关，谁先到窗口不确定，所以每次结果的最后几位可能不同。
""")

F["basics_triton"] = flow("autotune：第一次调用时把每个 config 都试一遍", """
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
""") + analogy("num_stages 是提前备料", """
厨师做第 k 道菜时，帮工已经在洗第 k+1、k+2 道的菜（num_stages = 3）：计算和搬运同时进行。备得越多越不会断档，但每一份备料都要占一块料理台（shared memory），台子放满了，能同时开工的厨师（occupancy）就少了。
""")

F["basics_workflow"] = flow("读 ncu 报告：三个数字定方向", """
n mem 0 0 "Memory Throughput 高" "> 80% 峰值" k=warn w=1.05
n sm 0 1 "SM / Tensor 吞吐高" "" k=acc w=1.05
n low 0 2 "两个都不高" "" k=todo
n mb 1.4 0 "已到带宽上限" "只能少读字节" k=warn
n cb 1.4 1 "已到算力上限" "换更快的指令 / 精度" k=acc w=1.1
n stall 1.4 2 "看 Warp State" "warp 在等什么？" k=todo
n ls 2.8 1.6 "Long Scoreboard" "等 global memory" k=vio w=1.05
n bar 2.8 2.6 "Barrier / MIO" "sync 或 smem 排队" k=vio
e mem -> mb
e sm -> cb
e low -> stall
e stall -> ls "加并行 / 预取"
e stall -> bar "减少同步 / 冲突"
""") + analogy("ncu 是体检报告", """
不要一上来就“吃补药”（盲目调参数）。先看体检报告：带宽已经满了（血压到顶），再快的计算也没用；两项都不高，说明问题在“等待”上，就去看 warp 在等什么：是在等内存（Long Scoreboard），还是在等同伴（Barrier）。
""")

F["landscape"] = flow("NVIDIA 架构代际：哪些能力是新加的", """
cfg cw=186 nw=156
n a100 0 0.5 "sm_80 · A100" "mma.sync · cp.async" k=acc
n ada 1 0.5 "sm_89 · 4090 / L40S" "+ FP8 mma.sync" k=acc
n h100 2 0 "sm_90a · H100" "+ wgmma · TMA · 集群" k=vio
n b200 3 0 "sm_100 · B200" "+ tcgen05 · TMEM · FP4" k=pass
n r5090 3 1 "sm_120 · RTX 5090" "mma.sync + FP4 / FP6" k=warn
n fa3 4.1 0 "FA3 · FA4 · DeepGEMM" "依赖 wgmma / tcgen05" k=pass w=1.1
n fa2 4.1 1 "FA2 类 · Triton" "SageAttention3" k=warn w=1.1
e a100 -> ada
e ada -> h100
e h100 -> b200
e ada -> r5090 "没有 wgmma / tcgen05" dash
e b200 -> fa3
e r5090 -> fa2
""") + analogy("同名不同厂", """
“Blackwell”这个名字同时用在 B200 和 RTX 5090 上，就像同一个汽车品牌既有赛车也有家用车：品牌一样，发动机不同。B200 有 tcgen05 这台“涡轮”，为它调校的 kernel 拿到 5090 上点不着火；5090 上能用的是更接近上一代（Ada）的 `mma.sync`，只是多了 FP4。
""")

F["c1"] = flow("launch 配置：n = 1000、每 block 256 线程", """
n n 0 0.5 "n = 1000" "" k=acc
n b 1 0.5 "blocks = ceil(1000 / 256)" "= 4" k=vio w=1.3
n b0 2.6 0 "block 0–2" "768 个线程全部有效" k=pass
n b3 2.6 1 "block 3" "只有 232 个有效" k=warn
n chk 3.8 1 "i ≥ 1000 的 24 个线程" "if (i < n) 跳过" k=todo w=1.2
e n -> b
e b -> b0
e b -> b3
e b3 -> chk
""") + analogy("float4 是一次搬一箱", """
每个线程一次读一个 float，就像一瓶一瓶地搬水；用 `float4`（一次 16 字节）就是一次搬一箱。总重量（字节数）不变，但搬运的次数（指令数）少了 4 倍。前提是箱子摆得整齐：地址要 16 字节对齐，长度是 4 的倍数。
""")

F["c2"] = flow("__shfl_down_sync 的 5 轮（只看 lane 0 的视角）", """
n r0 0 0 "第 1 轮 offset 16" "lane i += lane i+16" k=teal
n r1 1 0 "第 2 轮 offset 8" "" k=teal
n r2 2 0 "第 3 轮 offset 4" "" k=teal
n r3 3 0 "第 4 轮 offset 2" "" k=teal
n r4 4 0 "第 5 轮 offset 1" "" k=teal
n out 4 1 "lane 0 = 32 个数之和" "其余 lane 是部分和" k=pass w=1.15
e r0 -> r1 "16 个有效"
e r1 -> r2 "8 个"
e r2 -> r3 "4 个"
e r3 -> r4 "2 个"
e r4 -> out
""") + analogy("淘汰赛", """
32 名选手打淘汰赛：第一轮 1 号对 17 号、2 号对 18 号……（offset 16），赢家把对手的分数加到自己身上；每轮人数减半，5 轮之后冠军（lane 0）手里就是全部分数之和。整场比赛在一张桌上完成（寄存器之间直接交换），不用换场地（shared memory）。
""")

F["c3"] = flow("block 坐标在写回时对调", """
n rd 0 0.5 "block (bx, by) 读 x" "行 by·32…, 列 bx·32…" k=acc w=1.2
n tile 1.5 0.5 "tile[ty + j][tx]" "" k=teal
n wr 2.8 0.5 "写到 y 的 block (by, bx)" "行 bx·32…, 列 by·32…" k=pass w=1.25
n rule 2.8 1.5 "threadIdx.x 仍走 y 的列" "→ 写入也合并" k=warn w=1.25
e rd -> tile
e tile -> wr "读 tile[tx][ty + j]"
e wr -> rule dash
""") + analogy("换座位", """
转置就是让 (行, 列) 互换座位。如果让每个人直接走到对面的新座位，写入是跨步的（每个人都要穿过整个礼堂）；先在小会议室（shared memory）里按新的排法站好队，再整排整排地走进新礼堂，每一排都是挨着走的。
""")

F["c4"] = flow("register tiling：256 个线程怎么分 64×64 的 C 块", """
n blk 0 0.5 "block：C 的 64×64" "256 线程" k=acc
n tid 1.2 0.5 "tid → (ty, tx)" "tx = tid % 16，ty = tid / 16" k=vio w=1.25
n patch 2.6 0.5 "每线程 4×4 个输出" "acc[4][4] 在寄存器" k=pass w=1.05
n af 3.85 0 "a_frag：As[k][ty·4 …]" "连续 4 个（A 转置存）" k=teal w=1.25
n bf 3.85 1 "b_frag：Bs[k][tx·4 …]" "连续 4 个" k=teal w=1.25
e blk -> tid
e tid -> patch
e patch -> af "每个 k"
e patch -> bf
""") + analogy("外积就是九九乘法表", """
手里有一列 4 个数（a_frag）和一行 4 个数（b_frag），把它们两两相乘就是一张 4×4 的乘法表：读 8 个数，做 16 次乘加。tile 越大，表越大，每读一个数能做的乘法越多，但每个人（线程）手里要拿的东西（寄存器）也越多，拿不下就会掉到地上（寄存器溢出到 local memory）。
""")

F["t1"] = flow("dtype 的一进一出", """
n in 0 0.5 "x, y：fp16" "HBM 里" k=acc
n up 1.1 0.5 ".to(tl.float32)" "加载后立刻升精度" k=vio
n cmp 2.2 0.5 "gelu(x + y)" "fp32 计算：z³ 不溢出" k=teal w=1.05
n down 3.35 0.5 ".to(out.dtype.element_ty)" "存回前降精度" k=vio w=1.25
n outn 4.6 0.5 "out：fp16" "" k=pass
e in -> up
e up -> cmp
e cmp -> down
e down -> outn
""") + analogy("用大锅炒，用小碗装", """
fp16 是小碗（最大约 65504，精度 3 位），fp32 是大锅。食材（数据）用小碗运进厨房省地方，但炒的时候（z³ 这种会变大的中间结果）要倒进大锅，否则会溢出；炒完再盛回小碗端出去。
""")

F["t2"] = flow("行太长时：online softmax 分段扫描（T5 的预告）", """
n c0 0 0.5 "第 1 段" "m = max₀，l = Σexp(x − m)" k=acc w=1.1
n c1 1.35 0.5 "第 2 段" "m′ = max(m, max₁)" k=acc
n up 2.55 0.5 "旧的 l 打折" "l = l·exp(m − m′) + Σ…" k=warn w=1.15
n end 3.9 0.5 "最后一段" "得到 (m, l)" k=pass
n w 3.9 1.5 "再扫一遍写出" "exp(x − m) / l" k=pass
e c0 -> c1
e c1 -> up
e up -> end
e up:t -> c1:t "下一段" anim
e end -> w
""") + analogy("换汇率", """
你在不同国家记账，一开始用美元当基准（当前最大值 m），后来发现日元更大（出现了更大的 m′），就要把之前所有账目按新汇率折算一次：旧的总和乘以 exp(m − m′)。折算只需要一次乘法，不需要回去重新翻每一笔账。
""")

F["t3"] = flow("为什么用两遍法求方差", """
n x 0 0.5 "x ≈ 10000 + 小波动" "" k=acc w=1.1
n one 1.35 0 "E[x²] − E[x]²" "两个 1e8 量级的数相减" k=todo w=1.15
n two 1.35 1 "先减均值再平方" "Σ(x − μ)² / N" k=pass w=1.1
n bad 2.75 0 "灾难性抵消" "有效数字几乎全丢" k=todo
n ok 2.75 1 "数值稳定" "只和波动本身有关" k=pass
e x -> one
e x -> two
e one -> bad
e two -> ok
""") + analogy("称体重差", """
想知道两个人体重差多少，最差的办法是各自连同一辆 10 吨卡车一起称，再相减：秤的误差比他们的体重差还大（E[x²] − E[x]² 就是这样）。两遍法是先把卡车开走（减去均值），再称剩下的差别。
""")

F["t4"] = flow("num_stages = 3：搬运和计算重叠", """
n l0 0 0 "load K 块 0" "" k=todo
n l1 1 0 "load K 块 1" "" k=todo
n l2 2 0 "load K 块 2" "" k=todo
n l3 3 0 "load K 块 3" "" k=todo
n c0 2 1 "dot K 块 0" "" k=vio
n c1 3 1 "dot K 块 1" "" k=vio
n c2 4 1 "dot K 块 2" "" k=vio
e l0 -> l1
e l1 -> l2
e l2 -> l3
e l0 -> c0 "已就位" dash
e l1 -> c1 dash
e l2 -> c2 dash
e c0 -> c1
e c1 -> c2
cap: 计算第 k 块时，第 k+1、k+2 块已经在搬运（cp.async / TMA → shared memory）。每多一个 stage 就多占一份 shared memory。
""") + analogy("洗衣、烘干、叠衣服", """
只有一台洗衣机和一台烘干机：第一桶在烘干时，第二桶已经在洗了。软件流水线就是让“搬数据”（洗）和“算”（烘）同时进行，总时间接近两者中较慢的那个，而不是两者之和。
""")

F["t5"] = flow("causal：query 块只需要看左下三角", """
n q 0 1 "query 块 m" "位置 m·BM … (m+1)·BM − 1" k=acc w=1.15
n full 1.5 0 "key 块 0 … m−1" "全部可见：不用 mask" k=pass w=1.1
n diag 1.5 1 "key 块 m（对角）" "需要 query ≥ key 的 mask" k=warn w=1.15
n skip 1.5 2 "key 块 m+1 …" "全部被 mask：直接跳过" k=todo w=1.1
n save 3 2 "省掉约一半计算" "循环上界截到 (m+1)·BM" k=todo w=1.15
n split 3 0 "进阶：拆成两段循环" "只有对角块做 tl.where" k=pass w=1.15
e q -> full
e q -> diag
e q -> skip
e skip -> save
e full -> split dash
""") + analogy("只看已经发生的事", """
causal attention 里每个词只能看见它前面的词，就像写日记时只能回忆过去、不能偷看未来。query 块 m 之后的 key 块全是“未来”，整块跳过；只有对角线那一块是“今天”，一部分已经发生、一部分还没有，需要逐个判断（mask）。
""")

F["l1"] = flow("padding token 怎么被跳过", """
n tok 0 0.5 "batch 被补齐到 8 个 token" "CUDA graph 要求固定形状" k=acc w=1.3
n real 1.6 0 "真实 token" "slot ≥ 0" k=pass
n pad 1.6 1 "padding token" "slot = −1" k=todo
n wr 2.8 0 "写入 cache" "" k=pass
n no 2.8 1 "n_valid = 0" "整个 program 不读不写" k=todo w=1.1
e tok -> real
e tok -> pad
e real -> wr
e pad -> no
""") + analogy("空房卡", """
旅行团人数不固定，但大巴（CUDA graph）的座位数是固定的，空座位上放一张写着“−1”的空房卡。前台（kernel）看到空房卡就什么也不做，不会把任何东西放进 −1 号房间。
""")

F["l2"] = flow("decode 为什么是 memory-bound", """
n tok 0 0.5 "每个 KV token" "" k=acc
n rd 1.2 0 "要读 2·D 个数" "K 和 V（fp16：4D 字节）" k=todo w=1.1
n fl 1.2 1 "只做约 4·D 次运算" "点积 + 加权求和" k=vio w=1.1
n ai 2.6 0.5 "AI ≈ 1 FLOP / 字节" "拐点约 100–300" k=warn w=1.05
n t 3.85 0.5 "时间 ≈ KV 字节数 ÷ 带宽" "" k=pass w=1.3
e tok -> rd
e tok -> fl
e rd -> ai
e fl -> ai
e ai -> t
""") + analogy("每页只读一个字", """
decode 时每读一个 KV token，只做几次乘加就扔掉了，就像为了查一个字把整本字典从书架上搬下来又放回去：时间都花在搬书上。所以一切优化都在减少搬书：GQA（几个人一起查）、KV 量化（字典印得更小）、paged（字典不必连号放）。
""")

F["l3"] = flow("分几片合适？", """
n in 0 0.5 "B · Hkv 个 program" "SM 数 = 170（5090）" k=acc w=1.1
n few 1.35 0 "远少于 SM 数" "例：B = 1 → 8 个" k=todo
n many 1.35 1 "已经是 SM 数的几倍" "例：B = 64 → 512 个" k=pass w=1.1
n sp 2.65 0 "多分几片" "直到 B·Hkv·splits ≈ 几倍 SM" k=warn w=1.25
n no 2.65 1 "不分片" "分片只会多一次写回和合并" k=pass w=1.25
e in -> few
e in -> many
e few -> sp
e many -> no
""") + analogy("搬家公司派几个人", """
只搬一个书柜（batch 小），派一个人慢慢搬，其他 169 个工人在旁边看着；把书柜拆成 64 格分给 64 个人同时搬，再在新家组装（合并），快得多。可如果本来就要搬 512 个书柜（batch 大），每人已经有活干，再拆只会多出组装的功夫。
""")

F["f1"] = flow("BatchDecodeWithPagedKVCacheWrapper 内部", """
n plan 0 0.5 "plan()" "CPU：看序列长度" k=warn
n dec 1.2 0.5 "要不要 split-KV？" "scheduler.cuh" k=vio w=1.05
n k1 2.5 0 "decode kernel" "直接写输出" k=pass
n k2 2.5 1 "decode kernel（分片）" "写部分结果到 workspace" k=pass w=1.15
n mg 3.85 1 "merge kernel" "F2 的 merge_states" k=teal
e plan -> dec
e dec -> k1 "否"
e dec -> k2 "是"
e k2 -> mg
""") + analogy("把座位表交给服务员", """
`plan()` 排好的座位表不是放在前台（CPU），而是复印一份交给每位服务员（拷到 GPU 的 workspace）。这样上菜时（`run()`）服务员不用回前台问，整个上菜过程可以被提前“录像”（CUDA graph），以后每一桌照着录像回放。
""")

F["f2"] = flow("merge_states：N 个状态一次合并", """
n s0 0 0 "(v₀, s₀)" "" k=teal
n s1 0 1 "(v₁, s₁)" "" k=teal
n s2 0 2 "(v_{N−1}, s_{N−1})" "" k=teal
n mx 1.3 1 "m = max sᵢ" "空状态 s = −∞" k=warn
n w 2.4 1 "wᵢ = 2^(sᵢ − m)" "" k=vio
n out 3.5 1 "v = Σwᵢvᵢ / Σwᵢ" "s = m + log₂Σwᵢ" k=pass w=1.1
e s0 -> mx
e s1 -> mx
e s2 -> mx
e mx -> w
e w -> out
""") + analogy("合并是可以交换、结合的", """
两两合并的顺序无所谓：先合并 A 和 B 再合并 C，与先合并 B 和 C 再合并 A 结果相同（除了浮点舍入）。所以 split-KV、cascade、ring attention 都可以随意切分，最后再按任意顺序合并；merge_states 只是把 N 个状态在一个 kernel 里一次合并完。
""")

F["f3"] = flow("PDL：下一个 kernel 提前开始准备", """
n a 0 0 "kernel A（如 GEMM）" "主体计算" k=acc w=1.1
n at 1.3 0 "A 的收尾" "griddepcontrol.launch_dependents" k=acc w=1.35
n bp 1.3 1 "kernel B 的准备" "加载权重 / 计算地址" k=vio w=1.1
n bw 2.75 1 "griddepcontrol.wait" "等 A 的结果可见" k=warn w=1.15
n bm 4 1 "B 的主体" "读 A 的输出" k=pass
e a -> at
e at -> bp "允许 B 启动" dash
e bp -> bw
e bw -> bm
""") + analogy("接力赛提前起跑", """
接力赛里，下一棒不会站着等上一棒跑到面前才起步，而是在交接区提前助跑（PDL：上一个 kernel 快结束时，下一个 kernel 就开始做不依赖上一棒结果的准备），真正交接棒（`griddepcontrol.wait`）时已经在全速奔跑。decode 阶段每个 kernel 只有几微秒，省下的这段启动时间很可观。
""")

F["f4"] = flow("排序 vs 不排序", """
n p 0 0.5 "15 万个概率" "" k=acc
n sort 1.3 0 "全局排序" "O(V log V) · 多次全局读写" k=todo w=1.15
n cum 2.75 0 "前缀和 → 截断" "" k=todo
n piv 1.3 1 "pivot 归约" "每轮一次片上扫描" k=pass w=1.05
n few 2.75 1 "几轮就收敛（FlashInfer）" "本题：固定 31 轮" k=pass w=1.25
e p -> sort
e p -> piv
e sort -> cum
e piv -> few
""") + analogy("找全班第 20 高的人", """
想找出全班最高的 20 个人，不需要让全班按身高排好队。拿一根标杆问“比这根高的站出来”：站出来的太多就把标杆调高，太少就调低。几轮之后正好站出来 20 个人，整个过程大家都没有挪动位置（不需要排序和额外的显存）。
""")

F["f5"] = flow("JIT 缓存命中的判断", """
n call 0 0.5 "调用 API" "dtype · head_dim · 架构 …" k=acc w=1.1
n spec 1.35 0.5 "生成 JitSpec" "名字编码了全部参数" k=vio w=1.05
n hit 2.6 0.5 "缓存里有吗？" "~/.cache/flashinfer" k=warn
n load 3.8 0 "直接加载 .so" "毫秒级" k=pass
n build 3.8 1 "生成源码 + ninja 编译" "首次：几十秒" k=todo w=1.2
e call -> spec
e spec -> hit
e hit -> load "有"
e hit -> build "没有 / 源码改过"
e build:b -> load:r "之后命中" dash
""") + analogy("读源码像拆快递", """
FlashInfer 的一个 API 是层层包装的快递：Python 函数（外箱）→ JIT 生成器（填充物，列出要编译什么）→ csrc 绑定（内盒，把张量换成指针）→ include 里的 kernel（真正的货物）。读源码时不要从中间撕开，按这个顺序一层层拆，每层只问一个问题：这一层负责什么？
""")

F["g1"] = flow("tl.sum(x, axis=1) 之后：SliceLayout", """
n x 0 0.5 "x：BlockedLayout [16, 32]" "lane 排成 4×8" k=acc w=1.3
n red 1.6 0.5 "沿 axis 1 归约" "线程内 4 个 + 3 轮 shuffle" k=vio w=1.25
n sl 3.1 0.5 "SliceLayout(1, parent)" "长度 16" k=teal w=1.15
n dup 3.1 1.5 "同一行的 8 个 lane" "都持有这一行的和" k=warn w=1.15
e x -> red
e red -> sl
e sl -> dup dash
""") + analogy("开会之后每人都有一份纪要", """
一排 8 个人（同一行的 8 个 lane）各自算了一部分，开完会（butterfly 归约）之后，每个人手里都有一份完全相同的会议纪要（这一行的和）。这就是 SliceLayout 里的“复制”：同一个值存在多个线程里，后续每个线程都能直接用，不必再广播。
""")

F["g2"] = flow("swizzle：用异或打散 bank", """
n addr 0 0.5 "shared memory 地址" "" k=acc
n bits 1.25 0.5 "取出高位 [M+S, M+S+B)" "" k=vio w=1.15
n xor 2.65 0.5 "异或到低位 [M, M+B)" "Swizzle<B, M, S>" k=warn w=1.1
n bank 3.95 0.5 "同一列落到不同 bank" "无需 padding" k=pass w=1.1
e addr -> bits
e bits -> xor
e xor -> bank
""") + analogy("按楼层错开房号", """
每层楼都有 32 个房间，大家都住 5 号房（同一列）就会挤在同一个电梯口（同一个 bank）。swizzle 规定“第 i 层的房号要和楼层号做一次异或”，于是每层的“5 号房”实际落在不同的位置，电梯口就不拥挤了；而且和 padding 不同，它一个房间都不浪费。
""")

F["m1"] = flow("decode 时补齐的浪费", """
n t 0 0.5 "T = 16 个 token · top-8" "128 个 (token, 专家) 对" k=acc w=1.25
n e 1.6 0.5 "分到 64 个专家" "平均每个专家 2 行" k=vio w=1.05
n pad 2.9 0 "BLOCK_M = 64" "每块 62 行是补齐" k=todo
n small 2.9 1 "BLOCK_M = 16" "浪费小得多" k=pass
n mask 4.1 0.5 "或者 masked 布局" "DeepGEMM · CUDA graph" k=teal w=1.1
e t -> e
e e -> pad
e e -> small
e small -> mask dash
""") + analogy("一辆大巴只坐两个人", """
每个诊室一次叫 64 个号（BLOCK_M = 64），可 decode 时每个科室平均只有 2 位病人，诊室大部分时间都在给空号看病。要么把每批改小（BLOCK_M = 16），要么换一种调度：每个诊室预留固定的号段，按实际人数开诊（masked 布局）。
""")

F["m2"] = flow("把 silu_and_mul 融合进 GEMM1 的 epilogue", """
n a 0 0.5 "x 的 gather 行" "" k=acc
n gate 1.2 0 "acc_gate" "x @ W_gate[e]ᵀ" k=vio
n up 1.2 1 "acc_up" "x @ W_up[e]ᵀ" k=vio
n act 2.4 0.5 "silu(gate) · up" "还在寄存器里" k=teal
n st 3.5 0.5 "只写 h [T·K, I]" "省掉 [T·K, 2I] 的写回和读取" k=pass w=1.3
e a -> gate
e a -> up
e gate -> act
e up -> act
e act -> st
""") + analogy("顺路办两件事", """
不融合时，GEMM1 把 gate 和 up 写回仓库（HBM），激活函数再把它们取出来、算完再写回去。融合就是在同一个 program 里同时算出 gate 和 up，趁热（还在寄存器里）把激活做了，只把结果送出门：少跑一趟仓库，还少占一半的仓库空间。
""")

F["m3"] = flow("一个离群值对 per-tensor scale 的影响", """
n x 0 0.5 "激活里有一个 1000" "其余大多 < 1" k=acc
n pt 1.3 0 "per-tensor scale" "= 1000 / 448 ≈ 2.2" k=todo
n bs 1.3 1 "1×128 block scale" "离群值只影响它所在的组" k=pass w=1.15
n z 2.6 0 "0.5 / 2.2 ≈ 0.22" "FP8 只剩一两个量级的精度" k=todo w=1.15
n ok 2.6 1 "其他组 scale ≈ 1 / 448" "小数量级也量得准" k=pass w=1.15
e x -> pt
e x -> bs
e pt -> z
e bs -> ok
""") + analogy("一个巨人站进合影", """
集体照里有一个 3 米高的巨人，为了把他拍进去，摄影师只能把镜头拉远（大 scale），结果其他人都只剩几个像素（精度丢失）。分组拍照（block scale）：巨人那一组单独拉远，其他组照常拍特写。
""")

F["x1"] = flow("统一内存：不需要 host ↔ device 拷贝", """
n cpu 0 0 "CPU" "" k=acc
n gpu 0 1 "GPU" "" k=vio
n um 1.3 0.5 "同一块统一内存" "100–800 GB/s" k=pass w=1.05
n nv 2.7 0 "独立显卡：主机内存" "" k=todo w=1.05
n hbm 2.7 1 "独立显卡：显存" "" k=todo
e cpu -> um
e gpu -> um
e nv -> hbm "PCIe 拷贝" dash
""") + analogy("同一张桌子 vs 两个房间", """
独立显卡像两个房间：CPU 在客厅，GPU 在书房，资料要通过走廊（PCIe）来回搬。Apple 的统一内存是两人共用一张桌子，不用搬资料；代价是这张桌子比 NVIDIA 书房里的桌子（HBM）小一些、取东西也慢一些（带宽低一个数量级）。
""")

F["x2"] = flow("改进：online softmax 把三遍读变成两遍", """
n p1 0 0.5 "第 1 遍：同时维护 (m, s)" "s = s·exp(m − m′) + exp(x − m′)" k=pass w=1.4
n red 1.6 0.5 "tg_reduce 合并 (m, s)" "需要自定义的归约" k=warn w=1.15
n p2 3.0 0.5 "第 2 遍：写出" "exp(x − m) / s" k=pass
e p1 -> red
e red -> p2
cap: 跨线程合并 (m, s) 对时，要先取 m 的最大值，再把每个线程的 s 按 exp(mᵢ − m) 折算后相加，和 F2 的 merge_state 是同一个公式。
""") + analogy("simdgroup 是一桌人", """
一个 simdgroup 的 32 个线程像一桌吃饭的人，`simd_sum` 是这一桌直接凑钱，不用离开座位；threadgroup 内存是整个包间的公共托盘，几桌要先把各自的钱放到托盘上（写 scratch），等大家都放好（barrier），再由每桌派人来数托盘。
""")

F["x3"] = flow("Mac 上 decode 的速度上限", """
n m 0 0.5 "7B 模型 · 4-bit 量化" "约 3.5–4 GB 权重" k=acc w=1.15
n bw 1.4 0 "M2：约 100 GB/s" "" k=warn
n bw2 1.4 1 "M2 Ultra：约 800 GB/s" "" k=warn w=1.15
n t1 2.8 0 "约 25 tok/s 上限" "带宽 ÷ 模型字节数" k=pass w=1.1
n t2 2.8 1 "约 200 tok/s 上限" "" k=pass
e m -> bw
e m -> bw2
e bw -> t1
e bw2 -> t2
""") + analogy("每生成一个字都要把书读一遍", """
decode 时每生成一个 token，都要把整个模型的权重从内存里读一遍，就像每写一个字都要从头翻一遍参考书。翻书的速度（带宽）除以书的厚度（模型字节数）就是写字速度的上限：所以把书印得更薄（4-bit 量化）比换一支更快的笔（更快的 GPU 计算）更有效。
""")

for name, body in F.items():
    p = C / f"{name}.md"
    s = p.read_text()
    if MARK in s:
        continue
    block = MARK + "\n" + body + "\n"
    if name == "intro":
        s = s.replace("## 每道题的结构", "## 路线图\n\n" + block + "## 每道题的结构", 1)
    else:
        i = s.index("## 图解与类比")
        j = s.index("\n## ", i + 1) + 1
        s = s[:j] + block + s[j:]
    p.write_text(s)
    print("added", name)
