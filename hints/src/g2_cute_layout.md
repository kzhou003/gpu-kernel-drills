title: G2 讲解：CuTe 布局代数
lede: CuTe 的 Layout 就是一个从坐标到偏移的函数。这一页把参考解里的六个函数逐个讲透，每个都配一个能手算、并且用参考解验证过的例子：列字典序怎么把下标拆成坐标、coalesce 什么时候能合并、composition 为什么是“用 B 去选 A 的元素”、complement 怎么找空隙，以及 logical_divide 怎么把一维切成 tile × 重复。

::: callout info 怎么读这一页
对照 [[here:solutions/g2_cute_layout.py|g2_cute_layout.py]] 读。这题是纯 Python：`from solutions.g2_cute_layout import *`，本页每一个结果都可以在终端里复现。CuTe 官方文档 [[cutlass:media/docs/cpp/cute/02_layout_algebra.md|02_layout_algebra.md]] 是这一页的权威参考。
:::

## ① 先看全貌：Layout = 函数

`Layout(shape, stride)`，记作 `shape:stride`。它把坐标映射成偏移：`offset = Σ coord_i × stride_i`。

| 记号 | 意思 | 下标 0..7 映射到的偏移 |
|---|---|---|
| `(4,2):(1,4)` | 4×2 列主序 | 0, 1, 2, 3, 4, 5, 6, 7 |
| `(4,2):(2,1)` | 4×2 行主序 | 0, 2, 4, 6, 1, 3, 5, 7 |

两个 layout 的 shape 一样，只是 stride 不同，于是“第 5 个元素”在内存里的位置不同：列主序是 5，行主序是 3。

::: flow 六个运算之间的关系
cfg cw=225 nw=175
n L 0 0.65 "Layout = shape:stride" "坐标 → 偏移" k=acc w=1.05
n co 1 0 "coalesce" "同一个函数，模式最少" k=teal
n cp 1 1.3 "composition A∘B" "用 B 选 A 的元素" k=vio
n cm 2 1.3 "complement(A, M)" "A 没覆盖到的空隙" k=warn
n dv 3.1 0.65 "logical_divide" "A ∘ (B, complement(B))" k=pass
e L -> co
e L -> cp
e cp -> dv
e cm -> dv
e co -> cp "内部先化简" dash
cap: logical_divide 是 composition 和 complement 的组合，CuTe 里的 local_tile、zipped_divide、TV layout 的切分全都建立在它上面。
:::

::: callout analogy 地址簿
Layout 像一本地址簿：你说“第 3 排第 1 个”（坐标），它告诉你门牌号（偏移）。shape 是地址簿的格式（几排、每排几个），stride 是门牌号的编排规则（往右走一格门牌号加多少、往下走一排加多少）。同一栋楼可以有好几本格式不同的地址簿（行主序、列主序、分块），它们都指向同一批房间。CuTe 的所有运算都是在“改写地址簿”，而不是搬动房间里的人。
:::

## ② 下标 → 坐标：列字典序

[[here:solutions/g2_cute_layout.py@L26-L35|_idx2crd（26–35 行）]] · [[here:solutions/g2_cute_layout.py@L38-L52|_inner / __call__（38–52 行）]]

整数下标先按**列字典序**（第一个模式变化最快）拆成坐标，再和 stride 做点积。

下标 5，shape (4, 2)：5 % 4 = **1**，5 // 4 = 1，1 % 2 = **1**，坐标 (1, 1)。

- 列主序 `(4,2):(1,4)`：1·1 + 1·4 = **5**
- 行主序 `(4,2):(2,1)`：1·2 + 1·1 = **3**

**嵌套的 shape** 按嵌套结构递归拆：`((2,2),3):((1,2),4)` 的下标 7 → 7 % 4 = 3 交给内层 (2,2) 拆成 (1,1)，7 // 4 = 1 给外层，坐标是 ((1,1),1)，偏移 1·1 + 1·2 + 1·4 = 7。嵌套让一个“逻辑上的维度”由几个物理维度组成，比如“一行 4 个元素，其实是 2 个线程各拿 2 个”。

**和 G1 的区别**：G1 用 `order` 指定哪一维最快；CuTe 固定第一个模式最快，想要别的顺序就改 stride。

### size 和 cosize

[[here:solutions/g2_cute_layout.py@L54-L58|54–58 行]]：size 是坐标的个数（shape 的乘积），cosize 是最大偏移 + 1。`(4,2):(1,4)` 的 size 和 cosize 都是 8（紧凑）；`(4,2):(2,8)` 的 size 是 8，cosize 是 3·2 + 1·8 + 1 = **15**，中间有空隙。

### 两种紧凑布局

[[here:solutions/g2_cute_layout.py@L64-L86|make_layout_right / make_layout_left（64–86 行）]]：`make_layout_right((4,2))` = `(4,2):(2,1)`（行主序，最后一个模式 stride 为 1）；`make_layout_left((4,2))` = `(4,2):(1,4)`（列主序，stride 是 shape 的前缀积）。

## ③ coalesce：同一个函数，最少的模式

[[here:solutions/g2_cute_layout.py@L89-L104|coalesce（89–104 行）]]

规则只有两条：去掉大小为 1 的模式；相邻两个模式 `s0:d0` 和 `s1:d1`，如果 `d1 == s0 × d0`（第二个正好接着第一个走完的地方），合并成 `(s0·s1):d0`。

| 输入 | 过程 | 结果 |
|---|---|---|
| `(2,1,4):(1,7,2)` | 去掉 1:7；2:1 走完到 2，下一个 stride 正好是 2，合并 | **8:1** |
| `((2,2),2):((1,2),4)` | 展平成 (2,2,2):(1,2,4)，三个依次接上 | **8:1** |
| `(2,4):(4,1)` | 2:4 走完到 8，下一个 stride 是 1 ≠ 8，不能合并 | **(2,4):(4,1)** |

coalesce 不改变函数（每个下标映射到的偏移不变），只是把描述化简。在 composition 里先对 A 做 coalesce，可以减少要处理的模式数。

## ④ composition：用 B 去选 A 的元素

[[here:solutions/g2_cute_layout.py@L107-L135|composition（107–135 行）]]

`R = A ∘ B`，`R(i) = A(B(i))`。B 先把 i 变成一个“A 的下标”，A 再把它变成偏移。

**例子**：A = `(4,2):(2,1)`（行主序 4×2），B = `4:2`（取第 0、2、4、6 个）。

| i | B(i) | A(B(i)) |
|---|---|---|
| 0 | 0 | A(0) = A((0,0)) = 0 |
| 1 | 2 | A(2) = A((2,0)) = 4 |
| 2 | 4 | A(4) = A((0,1)) = 1 |
| 3 | 6 | A(6) = A((2,1)) = 5 |

参考解给出的 R = **`(2,2):(4,1)`**：下标 0..3 → 0, 4, 1, 5 ✓。

注意结果不是简单的“stride 乘 2”：B 跨的步子跨过了 A 的第一个模式的边界（A 的第一维只有 4 个），所以结果被拆成了两个模式。参考解的算法就是在做这件事：对 B 的一个模式 `s:d`，沿 A 的模式往下走，先“跳过前 d 个”（除），再“取 s 个”（取模），每跨过 A 的一个模式边界，就在结果里多出一个模式。

另一个例子：A = `(4,4):(4,1)`（行主序 4×4），B = `(4,4):(4,1)`。A∘B = **`(4,4):(1,4)`**：用一个行主序的选取方式去读一个行主序的矩阵，结果相当于转置。composition 就是“重新排列数据”的统一表达。

::: callout analogy 先查目录，再查门牌
composition 像两本地址簿接力：B 是“我要哪几个人”的名单，名单上写的是 A 地址簿里的编号；A 再把编号翻译成门牌号。A∘B 是把两本合成一本：直接从“我要的第几个人”查到门牌号。CuTe 里切 tile、给线程分数据，都是先写一本“要哪些”的名单（B），再和数据的地址簿（A）合成。
:::

## ⑤ complement：A 的空隙

[[here:solutions/g2_cute_layout.py@L138-L152|complement（138–152 行）]]

`complement(A, M)` 是 A 在 `[0, M)` 里“没覆盖到的部分”的布局。A 和它的 complement 拼起来，恰好不重不漏地铺满 `[0, M)`。

| A | M | A 覆盖的偏移 | complement | complement 覆盖的偏移 |
|---|---|---|---|---|
| `4:1` | 16 | 0, 1, 2, 3 | **`4:4`** | 0, 4, 8, 12（每块的起点） |
| `4:2` | 16 | 0, 2, 4, 6 | **`(2,2):(1,8)`** | 0, 1, 8, 9 |
| `(2,2):(1,8)` | 32 | 0, 1, 8, 9 | **`(4,2):(2,16)`** | 0, 2, 4, 6, 16, 18, 20, 22 |

第二行：A 每隔一个取一个，取 4 个，覆盖了 0–7 里的偶数位置。空隙有两种：“奇数位”（偏移 +1）和“后半段”（偏移 +8）。A(i) + complement(j) 遍历所有组合，正好是 0..15 每个恰好一次。

算法（注释 141–142 行）：把 A 的模式按 stride 排序，每两个模式之间的“空档”变成 complement 的一个模式，最后再补一个模式一直延伸到 M。

## ⑥ logical_divide：切成 tile × 重复

[[here:solutions/g2_cute_layout.py@L155-L159|logical_divide（155–159 行）]]

```text
logical_divide(A, B) = A ∘ (B, complement(B, size(A)))
```

结果有两个模式：模式 0 是“一个 tile 里的元素”（B 选出来的），模式 1 是“tile 的重复”（complement 选出来的）。

| A | tiler B | 结果 | 含义 |
|---|---|---|---|
| `16:1` | `4:1` | **`(4,4):(1,4)`** | 连续切成 4 块，每块 4 个 |
| `16:1` | `4:2` | **`(4,(2,2)):(2,(1,8))`** | 每块是“隔一个取一个”的 4 个 |
| `16:2` | `4:1` | **`(4,4):(2,8)`** | A 本身 stride 是 2，块内偏移跟着放大 |

第二行展开（模式 0 是 tile 内的第几个，模式 1 是第几个 tile），每个 tile 拿到的偏移：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 90px repeat(4, minmax(56px, 1fr)); max-width: 480px">
<div class="lbl"></div><div class="lbl" style="text-align:center">元素 0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div><div class="lbl" style="text-align:center">3</div>
<div class="lbl">tile 0</div><div class="c0">0</div><div class="c0">2</div><div class="c0">4</div><div class="c0">6</div>
<div class="lbl">tile 1</div><div class="c1">1</div><div class="c1">3</div><div class="c1">5</div><div class="c1">7</div>
<div class="lbl">tile 2</div><div class="c2">8</div><div class="c2">10</div><div class="c2">12</div><div class="c2">14</div>
<div class="lbl">tile 3</div><div class="c3">9</div><div class="c3">11</div><div class="c3">13</div><div class="c3">15</div>
</div></div>
<p class="legend">tiler 4:2 决定了每个 tile 的形状（隔一个取一个），complement (2,2):(1,8) 决定了 tile 之间怎么排（先 +1，再 +8）。16 个偏移每个恰好出现一次。</p>
:::

::: flow logical_divide(16:1, 4:2) 是怎么算出来的
cfg cw=230 nw=180
n b 0 0 "tiler B = 4:2" "tile 的形状" k=acc
n c 0 1.3 "complement(B, 16)" "= (2,2):(1,8)，tile 的排列" k=warn w=1.05
n bc 1 0.65 "(B, complement)" "(4,(2,2)):(2,(1,8))" k=vio
n r 2.1 0.65 "A ∘ (…)" "A = 16:1，结果不变" k=pass
e b -> bc
e c -> bc
e bc -> r
cap: A 是 16:1（恒等映射）时，A∘X = X，所以结果就是拼起来的 (B, complement)。A 不是恒等映射时（比如第三行的 16:2），composition 会把 A 的 stride 带进来。
:::

**这为什么有用**：GEMM 里“把 C 矩阵切成 128×128 的 tile，第 (m, n) 个 block 拿第 (m, n) 个 tile”，在 CuTe 里就是 `local_tile` = 对 C 的 layout 做 divide 再取模式 1 的某个坐标。“把一个 tile 分给 128 个线程，每个线程拿 4 个”也是 divide。C4 里你手算的 `tid % 16`、`tid / 16` 和 4×4 小块，在 CuTe 里都可以写成 layout 运算。

## ⑦ 对照 CuTe 源码

- **C++（CuTe 本体）**：[[cutlass:include/cute/layout.hpp@L867|coalesce]] · [[cutlass:include/cute/layout.hpp@L1136|composition]] · [[cutlass:include/cute/layout.hpp@L1274|complement]] · [[cutlass:include/cute/layout.hpp@L1599|logical_divide]]
- **pycute（官方的纯 Python 参考实现，和这题最接近）**：[[cutlass:python/pycute/layout.py@L137|coalesce]] · [[cutlass:python/pycute/layout.py@L190|composition]] · [[cutlass:python/pycute/layout.py@L232|complement]] · [[cutlass:python/pycute/layout.py@L297|logical_divide]] · [[cutlass:python/pycute/layout.py@L343|zipped_divide]]
- **官方文档**：[[cutlass:media/docs/cpp/cute/01_layout.md|01_layout.md]] · [[cutlass:media/docs/cpp/cute/02_layout_algebra.md@L13|Coalesce]] · [[cutlass:media/docs/cpp/cute/02_layout_algebra.md@L77|Composition]] · [[cutlass:media/docs/cpp/cute/02_layout_algebra.md@L338|Complement]] · [[cutlass:media/docs/cpp/cute/02_layout_algebra.md@L381|Division]]
- **CuTe DSL notebook**：[[cutlass:examples/python/CuTeDSL/cute/notebooks/cute_layout_algebra.ipynb|cute_layout_algebra.ipynb]]
- **实际 kernel**：[[cutlass:examples/cute/tutorial/sgemm_1.cu@L52|CuTe sgemm_1]]（C4 的同一个算法，用 layout 运算写成）· swizzle（C3 讲过的异或）[[cutlass:include/cute/swizzle.hpp@L55|Swizzle]]

## ⑧ 常见坑与自测

- **按行字典序拆下标**：CuTe 是第一个模式最快。`(4,2)` 的下标 1 是 (1, 0)，不是 (0, 1)。
- **coalesce 时比较顺序弄反**：条件是 `d1 == s0 × d0`，用的是**前一个**模式的 shape 和 stride。
- **以为 composition 的结果模式数和 B 一样**：B 跨过 A 的模式边界时，结果会多出模式（第 ④ 节的例子）。
- **complement 用了未排序的模式**：要先按 stride 排序。

::: answer 自测 1：Layout((2,3),(3,1)) 是行主序还是列主序？下标 0..5 映射到哪些偏移？
shape (2,3)，stride (3,1)：第二个模式 stride 为 1，是 2×3 的行主序。下标按列字典序拆：0→(0,0)=0，1→(1,0)=3，2→(0,1)=1，3→(1,1)=4，4→(0,2)=2，5→(1,2)=5。偏移是 0, 3, 1, 4, 2, 5。
:::

::: answer 自测 2：complement(8:1, 32) 是什么？logical_divide(32:1, 8:1) 呢？
8:1 覆盖 0..7，空隙是“每 8 个一块的起点”：complement = 4:8。logical_divide = (8, 4):(1, 8)：4 个连续的 tile，每个 8 个元素。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/g2_cute_layout.py|solutions/g2_cute_layout.py]] · [[here:mine/g2_cute_layout.py|mine/g2_cute_layout.py]] · [[here:tests/test_g2_cute_layout.py|tests/test_g2_cute_layout.py]] · [[here:content/g2.md|content/g2.md]] · 前置：[G1 讲解](g1_layouts.html)

**CuTe**：[[cutlass:media/docs/cpp/cute/02_layout_algebra.md|layout algebra 文档]] · [[cutlass:python/pycute/layout.py|pycute]] · [[cutlass:include/cute/layout.hpp|layout.hpp]] · [[cutlass:examples/python/CuTeDSL/cute/notebooks/cute_layout_algebra.ipynb|CuTe DSL notebook]]
