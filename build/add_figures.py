"""One-off: insert a '## 图解与类比' section (React-Flow-style diagrams + analogies) into each content page.
Drills: before '## 题目要求'. Chapters: before '## 源码与资料'."""
from pathlib import Path

C = Path(__file__).resolve().parent.parent / "content"


def flow(title, body):
    return f"::: flow {title}\n{body.strip()}\n:::\n"


def analogy(title, body):
    return f"::: callout analogy {title}\n{body.strip()}\n:::\n"


F = {}

F["basics_hw"] = flow("软件层级 ↔ 硬件层级", """
n grid 0 0 "grid" "一次 kernel launch" k=acc
n block 0 1 "block / program" "≤ 1024 线程" k=acc
n warp 0 2 "warp" "32 线程同一条指令" k=acc
n thread 0 3 "thread" "私有寄存器" k=acc
n gpu 2 0 "GPU" "108–170 个 SM" k=vio
n sm 2 1 "SM" "同时驻留多个 block" k=vio
n sched 2 2 "warp 调度器" "每周期挑就绪的 warp" k=vio
n core 2 3 "CUDA core / Tensor Core" "执行单元" k=vio w=1.25
e grid -> block "拆成"
e block -> warp "拆成"
e warp -> thread "拆成"
e grid -> gpu "分发到" dash
e block -> sm "整块住在一个 SM" dash
e warp -> sched "被调度" dash
e thread -> core "执行" dash
cap: 左边是你写代码时的抽象，右边是它们在硬件上落在哪里。block 内的线程能共享 shared memory、能同步，正是因为整个 block 住在同一个 SM 上。
""") + flow("内存层次：数据从远到近", """
n hbm 0 0 "HBM（global）" "1.8–3.4 TB/s · 几十 GB" k=todo w=1.1
n l2 1.2 0 "L2 cache" "40–50 MB · 全 GPU 共享" k=warn w=1.1
n smem 2.4 0 "shared / L1" "≤ 228 KB / SM" k=teal w=1.1
n reg 3.6 0 "寄存器" "256 KB / SM · 最快" k=pass w=1.1
e hbm -> l2 "合并访存" k=todo anim
e l2 -> smem "tile 搬运" k=warn anim
e smem -> reg "复用多次" k=pass anim
cap: 每往右一级，容量小一两个数量级、速度快一个数量级。几乎所有优化都在做一件事：让一个从 HBM 读来的数，在更近的地方被用尽可能多次。
""") + analogy("厨房的三层储物", """
HBM 是街对面的仓库（大、远），L2 是楼下储藏室，shared memory 是灶台旁的料理台，寄存器是你手里正拿着的东西。
好厨师不会每切一刀菜就跑一趟仓库：一次把一筐菜搬到料理台（tiling），在台上反复取用（复用）。kernel 优化说到底就是“少跑仓库”。
""") + analogy("warp 是一支龙舟队", """
32 名桨手听同一个鼓点划桨（SIMT：同一条指令）。如果一半人要左转、一半人要右转（分支发散），只能先让左边那半划、右边那半等，再反过来，速度减半。
某支队在等补给（访存）时，调度器马上换另一支队上场：这就是为什么 SM 上要同时驻留足够多的 warp（occupancy）。
""") + analogy("roofline 是工厂的两个上限", """
工厂产量受两件事限制：原料运进来的速度（带宽）和机器加工的速度（算力）。每吨原料需要的加工量就是算术强度：
加工量少（elementwise、decode）时，机器总在等原料：memory-bound；加工量大（大 GEMM、prefill）时，原料堆在门口等机器：compute-bound。先判断卡在哪边，再决定是修路还是买机器。
""")

F["basics_cuda"] = flow("load_inline：一个 .cu 文件怎么变成 torch 算子", """
n cu 0 0 "你的 .cu" "kernel + host 函数" k=acc
n decl 0 1 "C++ 声明" "cpp_sources" k=acc
n nvcc 1 0 "nvcc" "→ cuda.o" k=vio
n cxx 1 1 "c++" "→ main.o（pybind）" k=vio
n so 2 0.5 "ninja 链接" "→ 扩展 .so" k=warn
n py 3 0.5 "Python 模块" "mod.my_op(x)" k=pass
n cache 2 1.65 "~/.cache/torch_extensions" "按名字 + 源码缓存" k=ghost w=1.35
e cu -> nvcc
e decl -> cxx "生成绑定"
e nvcc -> so
e cxx -> so
e so -> py "import"
e so -> cache dash
cap: 第一次调用要编译（30–60 秒），之后直接加载缓存。本仓库的 `common/cuda_ext.py` 把源码哈希放进模块名，所以改完 .cu 会自动重新编译。
""") + flow("kernel launch 是异步的", """
n py 0 0 "Python 调用" "立即返回" k=acc
n q 1 0 "CUDA stream" "kernel 排队" k=warn
n gpu 2 0 "GPU 执行" "grid 分发到 SM" k=pass
n sync 3 0 "同步点" ".item() / synchronize()" k=todo w=1.1
e py -> q "enqueue" anim
e q -> gpu "按顺序执行"
e gpu -> sync "完成后才拿到结果"
cap: 所以用 `time.time()` 包住一次调用，测到的只是“下单”的时间；越界访问的报错也常常在后面某个同步点才出现。
""") + analogy("launch 就是给工厂下订单", """
`kernel<<<...>>>` 像在网上下单：点完“提交”就返回了，货（结果）还在路上。要拿到货（读结果、打印、`.item()`），就得等快递到门口（同步）。
所以计时要在“收货”时按秒表（CUDA event + synchronize），而不是在“下单”时。
""") + analogy("__syncthreads 是集体合影", """
摄影师要等所有人都站好才按快门。如果有人被安排在 `if` 分支里永远不会来，摄影师就一直等下去（死锁）。所以 `__syncthreads()` 必须放在所有线程都会执行到的地方。
""") + analogy("grid-stride loop 是固定人数的流水线", """
不是“来多少箱货就雇多少工人”，而是雇固定数量的工人（SM 数 × k 个 block），每人做完一箱就去传送带上拿下一箱（`i += 总线程数`）。货多货少都用同一套班底，工人的私人账本（寄存器累加）还能一直记下去。
""")

F["basics_triton"] = flow("Triton 的编译管线", """
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
""") + flow("一个 program 怎么变成线程", """
n prog 0 0.5 "program pid" "offs = pid·B + arange(B)" k=acc w=1.25
n w0 1.6 0 "warp 0" "lane 0–31" k=vio
n w1 1.6 1 "warp 1 …" "num_warps 个" k=vio
n regs 2.8 0.5 "每线程若干寄存器" "sizePerThread" k=pass w=1.1
e prog -> w0 "按 layout 切分"
e prog -> w1
e w0 -> regs
e w1 -> regs
""") + analogy("CUDA 写给工人，Triton 写给工头", """
CUDA 里你给每个工人（线程）写说明书：“你是第几号，你负责第几个格子”。Triton 里你只给工头（program）写：“把这一箱（block）处理完”，工头自己决定怎么分给手下（num_warps × 32 个线程）、谁拿几个、怎么搬得最快。
`mask` 就是最后一箱不满，空格子贴上“跳过”标签；`tl.constexpr` 是模具尺寸：换尺寸就要重新开模（重新编译）。
""")

F["basics_workflow"] = flow("每个 kernel 都走的开发循环", """
n ref 0 0 "PyTorch reference" "高精度" k=pass
n naive 1 0 "最简实现" "先求正确" k=acc
n test 2 0 "测试" "边界 shape / dtype" k=acc
n bench 3 0 "benchmark" "GB/s · TFLOP/s" k=warn
n prof 3 1 "profile" "ncu / nsys" k=warn
n opt 2 1 "只改一个因素" "tile / warps / 融合" k=vio
n doc 1 1 "记录" "数字 + 原因" k=teal
e ref -> naive
e naive -> test
e test -> bench
e bench -> prof
e prof -> opt
e opt -> test "回归测试" anim
e opt -> doc dash
""") + analogy("调音台一次只推一个推子", """
同时动三个推子，声音变好了也说不清是哪个起了作用。kernel 调优也一样：每次只改一件事（block 大小、num_warps、padding……），测一次，记下来。
""")

F["landscape"] = flow("一个 decoder layer 里有哪些 kernel", """
cfg cw=182 nw=150
n emb 0 0 "embedding" "gather"
n n1 1 0 "RMSNorm" "融合小算子" k=teal
n qkv 2 0 "QKV GEMM" "GEMM" k=acc
n rope 3 0 "RoPE + 写 KV" "融合 · paged" k=teal
n attn 4 0 "attention" "prefill / decode" k=vio
n o 4 1 "O GEMM" "GEMM" k=acc
n add1 3 1 "残差 + RMSNorm" "fused_add_rmsnorm" k=teal
n up 2 1 "gate_up GEMM" "GEMM / MoE" k=acc
n act 1 1 "SwiGLU" "silu_and_mul" k=teal
n down 0 1 "down GEMM" "+ 残差" k=acc
n head 0 2.2 "LM head" "GEMM" k=acc
n samp 1 2.2 "采样" "top-k / top-p" k=warn
g 0 0 4 1 "× L 层"
e emb -> n1
e n1 -> qkv
e qkv -> rope
e rope -> attn
e attn -> o
e o -> add1
e add1 -> up
e up -> act
e act -> down
e down -> head "最后一层之后" dash
e head -> samp
cap: 颜色对应 kernel 类别：蓝色 GEMM、紫色 attention、青色融合小算子、橙色采样。prefill 时蓝紫两类占大头；decode 时小算子的固定开销和 KV 读取变得显著。
""") + flow("这个 kernel 该往哪个方向优化？", """
n q 0 1 "算术强度 AI" "FLOPs ÷ 字节" k=dark
n mb 1.1 0 "AI < 拐点" "memory-bound" k=warn
n tiny 1.1 1 "kernel < 10 us？" "batch 很小" k=todo
n cb 1.1 2 "AI > 拐点" "compute-bound" k=acc
n mb1 2.3 0 "少读字节" "融合 · 量化 · GQA / MLA" k=warn w=1.25
n tiny1 2.3 1 "减少 launch" "CUDA graph · 融合 · PDL" k=todo w=1.25
n cb1 2.3 2 "用满 Tensor Core" "tile · 流水线 · 新指令" k=acc w=1.25
e q -> mb
e q -> tiny
e q -> cb
e mb -> mb1
e tiny -> tiny1
e cb -> cb1
""") + analogy("生产线上最慢的工位决定产量", """
把一次推理看成一条生产线：embedding → 十几道工序 × L 层 → 采样。哪个工位最慢，就去优化哪个；把一个只占 2% 时间的工位提速 4 倍，整条线只快 1.5%（扩散模型里 attention 只占小部分时就是这样）。
""") + analogy("换硬件就是换工厂", """
B200 工厂里有专门的机械臂（tcgen05 + TMEM）和自动叉车（TMA），最好的工艺（FA4、DeepGEMM）是围绕它们设计的；RTX 5090 的工厂叫同一个品牌，却没有这台机械臂，所以那套工艺搬过去根本开不了工。选 kernel 时看 sm 版本，不看产品名。
""")

F["c1"] = flow("不融合 vs 融合：数一数 HBM 读写", """
n x 0 0.6 "x（HBM）" "" k=todo
n k1 1 0 "kernel 1" "t = x + bias" k=acc
n t 2 0 "t（HBM）" "临时张量" k=todo
n k2 3 0 "kernel 2" "y = gelu(t)" k=acc
n y 4 0.6 "y（HBM）" "" k=todo
n kf 2 1.35 "融合 kernel" "y = gelu(x + bias)" k=pass
e x -> k1 "读"
e k1 -> t "写"
e t -> k2 "读"
e k2 -> y "写"
e x -> kf "读 1 次" k=pass
e kf -> y "写 1 次" k=pass
cap: 上面一条路径 4 趟 HBM 读写、2 次 launch；下面 2 趟、1 次。memory-bound 的 kernel 时间正比于搬运的字节数，所以融合后约快 2 倍（RTX 5090 实测 706 → 379 us）。
""") + flow("一个线程怎么知道自己负责哪个元素", """
n b 0 0 "blockIdx.x = 2" "" k=acc
n t 0 1 "threadIdx.x = 5" "" k=acc
n i 1 0.5 "i = 2·256 + 5" "= 517" k=vio
n chk 2 0.5 "i < n ?" "最后一个 block 不满" k=warn
n w 3 0.5 "c[517] = a + b" "相邻线程 → 相邻地址" k=pass
e b -> i
e t -> i
e i -> chk
e chk -> w "是"
""") + analogy("演唱会的座位号", """
blockIdx 是区号，threadIdx 是区内座位号，全局编号 = 区号 × 每区座位数 + 座位号。最后一个区通常坐不满，多出来的座位不能发票（`if (i < n)`）。
相邻座位的观众去相邻的检票口（合并访存），入场才最快。
""") + analogy("融合 = 洗切炒一气呵成", """
不融合就像洗完菜放回冰箱，切菜时再拿出来，切完又放回去，炒的时候再拿：每次进出冰箱（HBM）都要时间。融合是在料理台上一次做完，只在开始拿一次、最后装盘一次。
""")

F["c2"] = flow("reduction 的四层结构", """
n t 0 0.5 "每个线程" "grid-stride 累加" k=pass
n w 1 0.5 "warp 内" "5 轮 __shfl_down_sync" k=teal w=1.05
n s 2.1 0.5 "block 内" "lane 0 → smem[warp]" k=warn
n w0 3.1 0.5 "warp 0" "归约 8 个部分和" k=teal
n a 4.1 0.5 "grid" "每 block 1 次 atomicAdd" k=todo w=1.1
e t -> w "寄存器"
e w -> s "shared memory"
e s -> w0 "__syncthreads"
e w0 -> a "global memory"
cap: 越往右通信越贵，所以越往右参与的数据越少：几百万个数在寄存器里就被压成几十万个，到 atomic 这一层只剩几百个。
""") + analogy("选举计票", """
每个投票站（线程）先自己数完；同一个镇的投票站用对讲机报数汇总（warp shuffle，不用开会）；县里把各镇代表叫到会议室开一次会（shared memory + `__syncthreads`）；最后每个县只给省里打一个电话（atomicAdd）。
电话打得越少越好，所以要让每个投票站尽量多数一些票（grid-stride loop）。atomic 的先后顺序不确定，所以最后几位小数可能每次不同。
""")

F["c3"] = flow("用 shared memory 中转的转置", """
n x 0 0.5 "x 的 32×32 tile" "行主序" k=acc
n sm 1.5 0.5 "smem tile[32][33]" "+1 列 padding" k=teal w=1.15
n y 3.1 0.5 "y 的 32×32 tile" "行主序" k=pass
e x -> sm "按行读：合并" k=acc anim
e sm -> y "按列取、按行写：合并" k=pass anim
cap: naive 版本里“读”和“写”总有一个是跨步的；中转之后，跨步访问只发生在 shared memory 里，而 shared memory 不怕跨步，只怕 bank conflict。
""") + flow("为什么要多加一列", """
n col 0 0.5 "warp 读一列" "tile[0..31][j]" k=acc
n nop 1.3 0 "不加 padding" "bank = j（全相同）" k=todo
n pad 1.3 1 "padding 到 33" "bank = (i + j) % 32" k=pass w=1.05
n s1 2.6 0 "32 路冲突" "串行 32 次" k=todo
n s2 2.6 1 "无冲突" "一次完成" k=pass
e col -> nop
e col -> pad
e nop -> s1
e pad -> s2
""") + analogy("32 个收银台", """
shared memory 有 32 个 bank，就像 32 个收银台。一列数据在不加 padding 时全部落在同一个台（每行正好 32 个数，绕一圈回到原位），32 个人排一条队；
每行多放一个空位后，第 i 行的第 j 个数落在第 (i + j) 个台，32 个人正好分散到 32 个台。
""") + analogy("搬书", """
naive 转置像从书架 A 按行取书、再按列塞进书架 B：每塞一本都要走到另一排。shared memory 版本是先按行把一摞书搬到桌上，再按书架 B 顺手的方向一排一排放进去：两边都是顺着走。
""")

F["c4"] = flow("GEMM 的数据流：一次读，多次用", """
n A 0 0 "A（HBM）" "[M, K]" k=todo
n B 0 1 "B（HBM）" "[K, N]" k=todo
n As 1 0 "As tile" "shared memory" k=teal
n Bs 1 1 "Bs tile" "shared memory" k=teal
n fr 2 0.5 "a_frag · b_frag" "寄存器" k=pass
n acc 3 0.5 "acc[TM][TN]" "外积累加" k=pass
n C 4 0.5 "C（HBM）" "最后写一次" k=todo
e A -> As "每个 K 块读一次"
e B -> Bs
e As -> fr "复用 TN 次"
e Bs -> fr "复用 TM 次"
e fr -> acc
e acc -> C
e acc:t -> As:t "下一个 K 块" anim
cap: naive 每次乘加都要从 HBM 读 2 个数；shared memory tiling 让每个数在 block 内被用 32 次；register tiling 再让每个从 shared memory 读来的数被用 4 次。
""") + analogy("一次做 100 份菜", """
naive：每做一道菜就去仓库拿两样食材。shared-memory tiling：一次拿一整筐放到料理台，32 道菜共用这一筐。register tiling：每个厨师同时端着 4×4 = 16 个盘子，一次拿起来的配料直接分到 4 个盘子里。
筐越大越省路，但料理台和手的容量有限（shared memory、寄存器）：tile 太大，能同时干活的厨师就少了。
""")

F["t1"] = flow("一个 program 做了什么", """
cfg cw=190
n pid 0 0.5 "program_id = 3" "" k=acc
n offs 1 0.5 "offs = 3·1024 + arange" "[3072 … 4095]" k=vio w=1.3
n mask 2.3 0.5 "mask = offs < n" "最后一块不满" k=warn
n ld 3.5 0 "tl.load(x + offs)" "→ fp32 计算" k=pass
n st 3.5 1 "tl.store(out + offs)" "→ 转回 fp16" k=pass
e pid -> offs
e offs -> mask
e mask -> ld
e ld -> st "gelu(x + y)"
""") + analogy("工头领箱子", """
每个工头（program）领到一个箱号（program_id），箱子里有 1024 个格子（arange）。最后一箱可能只有半箱货：没货的格子贴“跳过”（mask），load 时给它填一个无害的默认值（`other`），store 时直接不写。
""")

F["t2"] = flow("一行一个 program：全部在片上完成", """
n hbm 0 0.5 "x[row]（HBM）" "读 1 次" k=todo
n reg 1 0.5 "整行放进寄存器" "BLOCK = 2 的幂" k=pass w=1.05
n mx 2.1 0 "max" "tl.max" k=teal
n ex 2.1 1 "exp(x − max)" "" k=teal
n sm 3.1 0.5 "÷ sum" "tl.sum" k=teal
n out 4.1 0.5 "y[row]（HBM）" "写 1 次" k=todo
e hbm -> reg
e reg -> mx
e mx -> ex "减去"
e reg -> ex
e ex -> sm
e sm -> out
""") + analogy("以最高的人为基准量身高", """
直接算 eˣ 就像用海拔量身高：数字大得离谱（溢出）。先减去最大值，相当于以全班最高的人为 0 点，大家的“相对身高”都 ≤ 0，e 的幂都在 (0, 1] 之间，怎么都不会溢出；最后除以总和，就是按比例分蛋糕。
padding 位置填 −∞，等于给不存在的同学记一个“无限矮”，他既不会成为最高的人，分到的蛋糕也是 0。
""")

F["t3"] = flow("LayerNorm 的一行", """
n x 0 0.5 "x 行（fp32）" "" k=acc
n mean 1 0 "mean" "Σx / N" k=teal
n xc 2 0 "xc = where(mask, x − mean, 0)" "padding 必须清零" k=warn w=1.5
n var 3.6 0 "var · rstd" "Σxc² / N" k=teal
n y 3.6 1 "y = xc·rstd·w + b" "" k=pass w=1.15
n save 1 1.3 "存 mean / rstd" "backward 要用" k=ghost
e x -> mean
e mean -> xc
e x -> xc
e xc -> var
e var -> y
e var -> save dash
""") + analogy("成绩标准化", """
先算全班平均分，每个人减去平均分，再除以标准差。缺考的同学（padding）在求和时按 0 分算没问题，但“减去平均分”之后他会变成 −平均分，平方后就把方差拉大了：必须在平方之前把他重新清零。
""")

F["t4"] = flow("K 循环：两个指针块 + 一次 tl.dot", """
n ap 0 0 "a_ptrs [BM, BK]" "offs_m × offs_k" k=acc
n bp 0 1 "b_ptrs [BK, BN]" "offs_k × offs_n" k=acc
n ld 1 0.5 "masked load" "k_remaining" k=warn
n dot 2 0.5 "tl.dot(a, b, acc)" "Tensor Core" k=vio
n acc 3 0.5 "acc（fp32）" "[BM, BN]" k=pass
n st 4 0.5 "store C" "cast + mask" k=pass
e ap -> ld
e bp -> ld
e ld -> dot
e dot -> acc
e acc -> st
e acc:b -> ld:b "指针前进 BK" anim
""") + flow("grouped ordering 为什么能提高 L2 命中", """
n p 0 0.5 "pid（一维）" "" k=dark
n row 1.2 0 "行主序编号" "同时运行：一整行 tile" k=todo w=1.15
n grp 1.2 1 "GROUP_M 分组" "同时运行：一块方形区域" k=pass w=1.2
n l2a 2.6 0 "B 面板各不相同" "L2 装不下" k=todo w=1.1
n l2b 2.6 1 "A、B 面板都被复用" "L2 命中高" k=pass w=1.15
e p -> row
e p -> grp
e row -> l2a
e grp -> l2b
""") + analogy("图书馆的服务台", """
同一时刻在读书的人越是集中在少数几排书架，管理员（L2）越能把这些书都摆在服务台上；如果每个人要的都是不同书架上的书，服务台堆不下，只能一趟趟回库房（HBM）。grouped ordering 就是安排大家“先把这几排书架读完再换”。
""")

F["t5"] = flow("FlashAttention：Q 常驻，K/V 流过", """
n q 0 0.5 "Q tile [BM, D]" "加载一次，常驻" k=acc
n kv 1 0 "K / V 块 j" "从 HBM 流入" k=todo
n s 1 1 "S = Q·Kᵀ·scale" "[BM, BN] 只在片上" k=vio w=1.05
n st 2.2 0.5 "online softmax" "更新 m · l · α" k=teal
n acc 3.2 0.5 "acc = α·acc + P·V" "" k=pass w=1.05
n o 4.3 0.5 "O = acc / l" "只写一次" k=pass
e q -> s
e kv -> s
e s -> st
e st -> acc
e acc -> o
e acc:t -> kv:t "下一块 j + 1" anim
cap: 从头到尾不存在 S×S 的矩阵：每一时刻片上只有 BM×BN 的一小块分数。HBM 读写从 O(S²) 降到 O(S·D)。
""") + analogy("比赛记分：只记最高分和折算后的总分", """
你不需要保存每位选手的成绩，只要记住“目前的最高分 m”和“按这个最高分折算后的总分 l”。来了一位更高分的选手，最高分从 m_old 变成 m_new，之前的总分要整体打个折：乘以 α = exp(m_old − m_new)。
加权平均（acc）也同样打折。全部选手看完，用 acc / l 就得到和一次性看完全部成绩完全相同的结果。
""")

F["l1"] = flow("一个 token 的 KV 写到哪里", """
n tok 0 0.5 "token（seq 0, pos 37）" "" k=acc w=1.2
n lb 1.35 0 "逻辑块 37 // 16 = 2" "offset = 37 % 16 = 5" k=vio w=1.25
n bt 1.35 1 "block_table[0][2]" "= 物理块 9" k=warn w=1.1
n slot 2.8 0.5 "slot = 9·16 + 5 = 149" "" k=pass w=1.2
n cache 4.1 0.5 "k_cache 扁平偏移" "149 · H · D" k=todo
e tok -> lb
e lb -> bt "查表"
e bt -> slot
e lb -> slot "offset"
e slot -> cache
""") + analogy("酒店分房", """
每个旅行团（序列）有一张房间清单（block table）：“团里第 2 间房 = 905 号房”。客人（token）只知道自己是团里第 37 位：第 37 // 16 = 2 间房、第 37 % 16 = 5 张床，于是去 905 号房 5 号床（slot）。
房间按需分配、不必连号，团员增加时再开新房间：这就是 paged KV cache 没有碎片的原因。
""")

F["l2"] = flow("GQA decode：一个 KV head 服务一组 query head", """
n q 0 0.5 "8 个 query head" "同一个 KV head 下" k=acc
n pack 1 0.5 "打包成 [16, D]" "补齐到 16 行" k=vio
n bt 1 1.65 "block_table[b][p]" "→ 物理 page" k=warn w=1.05
n kv 2.2 1.65 "K / V page [16, D]" "每页只读一次" k=todo w=1.05
n dot 2.2 0.5 "S = Q·Kᵀ" "tl.dot" k=vio
n sm 3.3 0.5 "online softmax" "" k=teal
n o 4.3 0.5 "8 个 head 的输出" "" k=pass
e q -> pack
e pack -> dot
e bt -> kv
e kv -> dot
e dot -> sm
e sm -> o
e sm:b -> kv:r "下一页" anim
""") + analogy("组团借书", """
8 个人（query head）要看同一本参考书（KV head）。各自去图书馆借一次，就要搬 8 趟同一本书；组团借一次、大家围着一起看（GQA 打包），只搬 1 趟。decode 的瓶颈正是“搬书”（读 KV），所以这一招直接省下 7/8 的时间。
""")

F["l3"] = flow("split-KV：沿上下文切开，再精确合并", """
n ctx 0 1 "一个序列 32K token" "" k=acc w=1.1
n s0 1.25 0 "split 0" "page 0 – 511" k=vio
n s1 1.25 1 "split 1" "page 512 – 1023" k=vio
n s2 1.25 2 "split …" "" k=vio
n p 2.4 1 "(o_s, lse_s)" "fp32 部分结果" k=teal
n m 3.4 1 "merge" "w = exp(lse − max)" k=warn
n o 4.4 1 "out" "" k=pass
e ctx -> s0
e ctx -> s1
e ctx -> s2
e s0 -> p
e s1 -> p
e s2 -> p
e p -> m
e m -> o
""") + analogy("64 个人分头读一本书", """
一个人读 1000 页太慢，就分给 64 个人各读一段。每人交两样东西：这一段的摘要（o_s），以及“这一段占全书多少分量”（lse_s）。主编按分量加权合并摘要。
只交摘要不交分量，主编无法知道哪段更重要，也就没法正确合并；空的分段（短序列分不到页）分量是 0（lse = −∞），直接忽略。
""")

F["f1"] = flow("plan 一次，run 很多次", """
n lens 0 0.5 "这批序列的长度" "CPU" k=acc
n plan 1 0.5 "plan()" "分几片 · 每 CTA 的 page" k=warn w=1.05
n ws 2.1 0.5 "workspace" "GPU 上的调度元数据" k=teal
n r1 3.2 0 "run() · 第 1 层" "" k=pass
n r2 3.2 1 "run() · 第 L 层" "" k=pass
n graph 4.3 0.5 "CUDA graph" "可捕获：无 CPU 同步" k=vio
e lens -> plan
e plan -> ws "一次"
e ws -> r1
e ws -> r2
e r1 -> graph dash
e r2 -> graph dash
""") + flow("block table → CSR", """
n bt 0 1 "block_tables [B, max]" "按最长序列补齐" k=todo w=1.2
n ip 1.6 0 "indptr [0, 3, 4, 8]" "每个序列从哪开始" k=pass w=1.2
n ix 1.6 1 "indices [7,2,5, 3, 1,4,6,8]" "紧凑拼接" k=pass w=1.4
n lp 1.6 2 "last_page_len [1, 16, 1]" "最后一页用了几格" k=pass w=1.25
e bt -> ip
e bt -> ix
e bt -> lp
""") + analogy("婚宴排座", """
开席前把座位表排好一次（`plan()`），之后每上一道菜（每一层的 `run()`）服务员都照同一张表走，不用每道菜重新排座。
CSR 是不给每桌都摆 20 把椅子（按最长序列补齐），而是按实际人数紧凑排成一长排，再附一张“第几桌从第几号椅子开始”的索引（indptr）。
""")

F["f2"] = flow("cascade：共享前缀只读一次", """
n q 0 1 "64 个请求的 query" "" k=acc
n pre 1.1 0 "共享前缀 8K" "KV 只读一次" k=vio
n suf 1.1 2 "各自的后缀" "每个请求单独" k=vio
n s1 2.2 0 "(v_pre, s_pre)" "base-2 LSE" k=teal
n s2 2.2 2 "(v_suf, s_suf)" "" k=teal
n m 3.2 1 "merge_state" "2^(s − max) 加权" k=warn
n o 4.2 1 "最终输出" "" k=pass
e q -> pre
e q -> suf
e pre -> s1
e suf -> s2
e s1 -> m
e s2 -> m
e m -> o
""") + analogy("两个班合并算平均分", """
两个班各自算出平均分（v），还要知道各班的“总权重”（lse 编码的就是 softmax 的分母），按权重合并才是全年级的平均分。
底数就像单位：FlashInfer 用“以 2 为底”，你用“以 e 为底”，就像一边写斤一边写公斤，直接相加就错了。
""")

F["f3"] = flow("fused_add_rmsnorm：读两个张量，原地写回两个张量", """
n x 0 0 "x" "子层输出" k=acc
n r 0 1 "residual" "残差流" k=acc
n h 1 0.5 "h = x + residual" "fp32" k=vio
n rnew 2.1 1 "residual ← h" "原地写回" k=pass
n norm 2.1 0 "rstd = rsqrt(mean(h²) + ε)" "" k=teal w=1.4
n xnew 3.7 0 "x ← h · rstd · w" "原地写回" k=pass
e x -> h "读"
e r -> h "读"
e h -> rnew
e h -> norm
e norm -> xnew
""") + analogy("记账：一次翻开账本做完两件事", """
把今天的收入（x）记进账户余额（residual），这是给后面所有人看的总账；再把余额“标准化”一下交给下一个部门（归一化后的 x）。
不融合就是翻开账本记一笔、合上，再翻开做标准化：每翻一次账本都是一趟 HBM 读写。
""")

F["f4"] = flow("对浮点位模式二分：31 轮找到阈值", """
n row 0 0.5 "一行概率" "V = 151936" k=acc
n init 1 0.5 "lo = 0, hi = bits(max) + 1" "int32 位模式" k=vio w=1.35
n pass 2.5 0.5 "扫一遍整行" "count / mass(bits ≥ mid)" k=teal w=1.2
n upd 3.8 0.5 "更新 lo 或 hi" "保持不变式" k=warn
n out 3.8 1.6 "保留 x ≥ value(lo)" "再归一化" k=pass w=1.2
e row -> init
e init -> pass
e pass -> upd
e upd:t -> pass:t "31 轮" anim
e upd -> out "收敛"
""") + analogy("猜门槛游戏", """
每轮问一个问题：“概率 ≥ 这个门槛的有多少个（或者加起来多少）？”太多就抬高门槛，太少就降低。
在浮点值上二分像在一条刻度不均匀的尺子上找点，越找越慢；在位模式上二分像在字典里按页码翻：fp32 只有 2³¹ 个非负的值，31 次必然翻到那一页。
""")

F["f5"] = flow("rmsnorm 在 FlashInfer 里的调用链", """
n py 0 0 "flashinfer.rmsnorm()" "Python API" k=acc w=1.2
n mod 1.35 0 "get_norm_module()" "@functools.cache" k=acc w=1.1
n jit 2.65 0 "gen_norm_module()" "flashinfer/jit/norm.py" k=vio w=1.1
n csrc 2.65 1 "csrc/norm.cu" "+ TVM-FFI binding" k=warn w=1.1
n inc 1.35 1 "include/…/norm.cuh" "RMSNorm → RMSNormKernel" k=teal w=1.2
n so 0 1 "~/.cache/flashinfer" "编译好的 .so" k=ghost w=1.2
e py -> mod
e mod -> jit "首次调用"
e jit -> csrc "ninja 编译"
e csrc -> inc "模板实例化"
e inc -> so dash
e so -> py "之后直接加载" dash
""") + flow("一个贡献的完整流程", """
n is 0 0 "找 issue / 缺口" "" k=acc
n k 1 0 "kernel + 测试" "include · csrc · tests" k=vio
n b 2 0 "benchmark" "前后绝对数字 + GPU" k=warn
n pc 3 0 "pre-commit" "run -a" k=teal
n pr 3 1 "PR（默认模板）" "" k=pass
n ci 1.6 1 "@flashinfer-bot run tests/…" "只跑相关测试" k=pass w=1.45
n rev 0 1 "review / 合并" "" k=pass
e is -> k
e k -> b
e b -> pc
e pc -> pr
e pr -> ci
e ci -> rev
""") + analogy("JIT 是现点现做的餐厅", """
FlashInfer 的 kernel 组合（dtype × head_dim × 架构 × 各种开关）多到没法全部预先做好。JIT 是有人点哪道菜才做哪道，做过的菜谱冻在冰箱里（`~/.cache/flashinfer`），下次直接热；AOT 和 jit-cache 包则是开业前把最常点的菜提前做好，避免第一位客人等半分钟。
""")

F["g1"] = flow("BlockedLayout：元素 → warp → lane → 寄存器", """
n t 0 0.5 "张量 [16, 32]" "" k=acc
n wp 1 0.5 "warpsPerCTA [4, 1]" "每个 warp 4 行" k=vio w=1.05
n ln 2.1 0.5 "threadsPerWarp [4, 8]" "lane 排成 4×8" k=teal w=1.1
n rg 3.25 0.5 "sizePerThread [1, 4]" "每 lane 4 个寄存器" k=pass w=1.05
n e 1.6 1.6 "(5, 13) → warp 1 · lane 11 · reg 1" "" k=warn w=1.85
e t -> wp
e wp -> ln
e ln -> rg
e rg -> e "推导" dash
""") + analogy("发扑克牌", """
先把牌桌分成几个区（warpsPerCTA），每个区里按座位（threadsPerWarp）发牌，每人一次拿几张连号的牌（sizePerThread）；`order` 决定是先横着发还是先竖着发。
连号的牌正好是内存里相邻的地址时，一个人一次就能拿起一整摞（向量化），一桌人拿的牌也挨在一起（合并访存）。
""")

F["g2"] = flow("logical_divide：用布局代数切 tile", """
n A 0 1 "A = 24:1" "一维 24 个元素" k=acc
n B 1.2 0 "tiler B = 4:2" "每块取 4 个、隔 1 个" k=vio w=1.1
n cB 1.2 2 "complement(B, 24)" "= (2, 3):(1, 8)" k=vio w=1.1
n BB 2.5 1 "(B, complement)" "拼成两个模式" k=teal
n R 3.6 1 "A ∘ (B, comp)" "模式 0：块内 · 模式 1：第几块" k=pass w=1.35
e B -> cB
e B -> BB
e cB -> BB
e A -> R
e BB -> R
""") + analogy("layout 是一张地址公式", """
Layout 就是“坐标 → 地址”的公式；composition 是先查楼层表、再查房间表；complement 是这栋楼里你还没占用的房间；logical_divide 把一栋楼按“每 4 间一组”重新编号，编号的方式由公式推出来，而不是手写下标。
""")

F["m1"] = flow("从 router logits 到“每个块属于哪个专家”", """
n log 0 0.5 "router logits [T, E]" "" k=acc w=1.15
n tk 1.3 0.5 "softmax + top-k" "(weights, ids)" k=vio
n cnt 2.4 0 "bincount" "每个专家多少 token" k=teal w=1.05
n srt 2.4 1 "稳定排序" "按专家分组" k=teal
n pad 3.6 0.5 "补齐到 BLOCK_M" "padding = T·K" k=warn
n blk 4.7 0.5 "块 → 专家" "expert_ids" k=pass
e log -> tk
e tk -> cnt
e tk -> srt
e cnt -> pad
e srt -> pad
e pad -> blk
""") + analogy("医院分诊", """
每位病人（token）挂号时选 k 个科室（top-k 专家）。按科室把挂号单排好队（稳定排序），每个诊室一次叫 BLOCK_M 个号；人数凑不满一批时，用空号（padding = T·K）补齐。这样每个诊室（M2 里的一个 program）只看自己科室的病人，一次叫号就能做完一整批。
""")

F["m2"] = flow("一个 MoE 层的完整流水", """
cfg cw=182 nw=150
n x 0 0.5 "x [T, H]" "" k=acc
n al 1 0.5 "route + align" "M1" k=vio
n g1 2 0.5 "grouped GEMM 1" "x[f//K] @ w13[e]ᵀ" k=acc w=1.1
n act 3.1 0.5 "silu_and_mul" "F3" k=teal
n g2 4.1 0.5 "grouped GEMM 2" "h[f] @ w2[e]ᵀ · w" k=acc w=1.1
n sum 5.2 0.5 "Σ_k" "合并 k 个专家" k=pass
e x -> al
e al -> g1 "sorted_ids"
e g1 -> act
e act -> g2
e g2 -> sum
""") + analogy("快递分拣中心", """
先按目的地（专家）把包裹分拣成整车（BLOCK_M），每辆车（program）只跑一个目的地，路线（权重）只需要装载一次。每件包裹的面单（sorted_token_ids）告诉司机去哪个货架取货（gather 读 A 的行），送到后按原来的订单号放回（写 out[flat_id]）。
""")

F["m3"] = flow("FP8 block-scaled GEMM 的数据流", """
n a 0 0 "激活 bf16 [M, K]" "" k=acc w=1.05
n q 1.2 0 "per-token-group 量化" "1×128 → 一个 scale" k=vio w=1.15
n w 0 1 "权重 bf16 [N, K]" "" k=acc w=1.05
n wq 1.2 1 "block 量化（离线）" "128×128 → 一个 scale" k=vio w=1.15
n dot 2.5 0.5 "fp8 tl.dot" "每个 128 宽的 K 块" k=warn w=1.05
n sc 3.6 0.5 "× sa · sb" "fp32 累加" k=teal
n c 4.6 0.5 "C（bf16）" "" k=pass
e a -> q
e w -> wq
e q -> dot
e wq -> dot
e dot -> sc
e sc -> c
e sc:t -> dot:t "下一个 K 块" anim
""") + analogy("每 128 个数一把尺子", """
FP8 只有 3 位尾数，像一把刻度很粗的尺子。用一把尺子（per-tensor scale）同时量蚂蚁和大象，蚂蚁全变成 0；给每 128 个数各配一把合适量程的尺子（block scale），大数小数都量得准。
尺子的切换正好落在每个 K 块的边界上，所以 kernel 每算完一块乘一次 scale 就行，几乎没有额外开销。
""")

F["x1"] = flow("mx.fast.metal_kernel 做了什么", """
n py 0 0.5 "mx.fast.metal_kernel" "你只写函数体" k=acc w=1.2
n sig 1.4 0.5 "MLX 生成签名" "inputs · outputs · 属性" k=vio w=1.15
n mtl 2.75 0.5 "Metal 编译" "按 name + 源码缓存" k=warn w=1.05
n disp 3.95 0.5 "dispatchThreads" "grid = 线程总数" k=pass w=1.05
e py -> sig
e sig -> mtl
e mtl -> disp
""") + analogy("报名人数，而不是班级数", """
CUDA 的 `<<<grid, block>>>` 说的是“开几个班、每班几个人”；Metal / MLX 的 `grid=` 说的是“总共多少个学生”，`threadgroup=` 才是每班人数，最后一个班可以不满。用 CUDA 的习惯把班级数填进 grid，就会多开出 256 倍的学生。
""")

F["x2"] = flow("threadgroup 级归约", """
n lanes 0 0.5 "每个线程的部分值" "跨步扫这一行" k=acc w=1.1
n simd 1.35 0.5 "simd_sum / simd_max" "32 个 lane 都拿到" k=teal w=1.1
n scr 2.7 0.5 "scratch[sg]" "threadgroup 内存" k=warn
n bar 3.75 0.5 "barrier" "" k=todo
n fin 4.8 0.5 "再 simd_sum 一次" "每个 simdgroup" k=teal
e lanes -> simd
e simd -> scr "lane 0 写"
e scr -> bar
e bar -> fin
""") + analogy("快递送到前台还是仓库", """
MLX 可能把很小的输入放进 `constant` 地址空间（前台），大的放进 `device`（仓库）。写死 `const device T*` 等于坚持“包裹一定在仓库”，前台的包裹就取不到（编译错误）；写 `auto` 就是“送到哪我去哪取”。
""")

F["x3"] = flow("simdgroup_matrix GEMM 的分工", """
n tg 0 0.5 "threadgroup：C 的 32×32 块" "128 线程" k=acc w=1.35
n s0 1.55 0 "simdgroup 0 · 1" "上半部分 16 行" k=vio
n s1 1.55 1 "simdgroup 2 · 3" "下半部分 16 行" k=vio
n fr 2.65 0.5 "2×2 个 8×8 累加器" "simdgroup_float8x8" k=teal w=1.15
n mac 3.95 0.5 "simdgroup_multiply_accumulate" "8×8×8" k=pass w=1.55
e tg -> s0
e tg -> s1
e s0 -> fr
e s1 -> fr
e fr -> mac
""") + analogy("32 个人合抬一块拼图板", """
`simdgroup_matrix` 是一个 simdgroup（32 个线程）合力持有一块 8×8 的拼图板，一条指令把两块板相乘再加到第三块上。一个 threadgroup 的 4 组人各拼 4 块（2×2），合起来就是 C 的一个 32×32 区域。MLX 的 steel GEMM 让每组人同时拼更多块，复用更多，所以更快。
""")

CHAPTERS = {"basics_hw", "basics_cuda", "basics_triton", "basics_workflow", "landscape"}

for name, body in F.items():
    p = C / f"{name}.md"
    s = p.read_text()
    if "## 图解与类比" in s:
        continue
    sec = "## 图解与类比\n\n" + body + "\n"
    anchor = "## 源码与资料" if name in CHAPTERS else ("## Lab 1：开发环境" if name == "f5" else "## 题目要求")
    assert anchor in s, (name, anchor)
    s = s.replace(anchor, sec + anchor, 1)
    p.write_text(s)
    print("added", name)
