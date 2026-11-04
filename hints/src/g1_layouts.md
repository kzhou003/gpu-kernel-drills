title: G1 讲解：BlockedLayout 和 SliceLayout 到线程的映射
lede: Triton 里一个 [16, 16] 的张量看起来是一整块，硬件上它被拆给 2 个 warp × 32 个 lane，每个 lane 持有 4 个寄存器。这一页用参考解本身算出一张完整的“谁持有哪个元素”的图，逐个函数讲清楚坐标公式、order、重复和广播，再用同一个 layout 回答三个实际问题：tl.sum 沿哪个轴便宜、load 是否合并、归约之后的 SliceLayout 长什么样。

::: callout info 怎么读这一页
对照 [[here:solutions/g1_layouts.py|g1_layouts.py]] 读。这题是纯 Python，可以直接 `from solutions.g1_layouts import *` 在终端里跟着算。本页所有数字都是用参考解算出来的。
:::

## ① 先看全貌：四个参数

一个 BlockedLayout 由四个参数描述（[[here:solutions/g1_layouts.py@L26-L31|26–31 行]]）：

| 参数 | 含义 | 本页的例子 |
|---|---|---|
| `size_per_thread` (spt) | 每个线程沿每一维持有几个**连续**元素 | [1, 4]：每个线程拿一行里连续的 4 个 |
| `threads_per_warp` (tpw) | 一个 warp 的 32 个 lane 沿每一维怎么排（乘积必须是 32） | [8, 4]：8 行 × 4 列 |
| `warps_per_cta` (wpc) | warp 沿每一维怎么排（乘积是 num_warps） | [2, 1]：2 个 warp 上下摆 |
| `order` | 编号时哪一维变得最快（order[0] 最快） | [1, 0]：列（第 1 维）最快，也就是行优先 |

一个 CTA tile 沿每一维覆盖 `spt × tpw × wpc` 个元素（[[here:solutions/g1_layouts.py@L47-L49|tile()]]）：[1·8·2, 4·4·1] = **[16, 16]**，正好等于张量的形状。

用参考解算出来的完整映射，每格写的是持有它的 lane 编号，颜色是 warp：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 30px repeat(16, minmax(30px, 1fr)); min-width: 600px; gap: 2px">
<div class="lbl"></div><div class="lbl" style="text-align:center">0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div><div class="lbl" style="text-align:center">3</div><div class="lbl" style="text-align:center">4</div><div class="lbl" style="text-align:center">5</div><div class="lbl" style="text-align:center">6</div><div class="lbl" style="text-align:center">7</div><div class="lbl" style="text-align:center">8</div><div class="lbl" style="text-align:center">9</div><div class="lbl" style="text-align:center">10</div><div class="lbl" style="text-align:center">11</div><div class="lbl" style="text-align:center">12</div><div class="lbl" style="text-align:center">13</div><div class="lbl" style="text-align:center">14</div><div class="lbl" style="text-align:center">15</div>
<div class="lbl">0</div><div class="c0">0</div><div class="c0">0</div><div class="c0">0</div><div class="c0">0</div><div class="c0">1</div><div class="c0">1</div><div class="c0">1</div><div class="c0">1</div><div class="c0">2</div><div class="c0">2</div><div class="c0">2</div><div class="c0">2</div><div class="c0">3</div><div class="c0">3</div><div class="c0">3</div><div class="c0">3</div>
<div class="lbl">1</div><div class="c0">4</div><div class="c0">4</div><div class="c0">4</div><div class="c0">4</div><div class="c3">5</div><div class="c3">5</div><div class="c3">5</div><div class="c3">5</div><div class="c0">6</div><div class="c0">6</div><div class="c0">6</div><div class="c0">6</div><div class="c0">7</div><div class="c0">7</div><div class="c0">7</div><div class="c0">7</div>
<div class="lbl">2</div><div class="c0">8</div><div class="c0">8</div><div class="c0">8</div><div class="c0">8</div><div class="c0">9</div><div class="c0">9</div><div class="c0">9</div><div class="c0">9</div><div class="c0">10</div><div class="c0">10</div><div class="c0">10</div><div class="c0">10</div><div class="c0">11</div><div class="c0">11</div><div class="c0">11</div><div class="c0">11</div>
<div class="lbl">3</div><div class="c0">12</div><div class="c0">12</div><div class="c0">12</div><div class="c0">12</div><div class="c0">13</div><div class="c0">13</div><div class="c0">13</div><div class="c0">13</div><div class="c0">14</div><div class="c0">14</div><div class="c0">14</div><div class="c0">14</div><div class="c0">15</div><div class="c0">15</div><div class="c0">15</div><div class="c0">15</div>
<div class="lbl">4</div><div class="c0">16</div><div class="c0">16</div><div class="c0">16</div><div class="c0">16</div><div class="c0">17</div><div class="c0">17</div><div class="c0">17</div><div class="c0">17</div><div class="c0">18</div><div class="c0">18</div><div class="c0">18</div><div class="c0">18</div><div class="c0">19</div><div class="c0">19</div><div class="c0">19</div><div class="c0">19</div>
<div class="lbl">5</div><div class="c0">20</div><div class="c0">20</div><div class="c0">20</div><div class="c0">20</div><div class="c0">21</div><div class="c0">21</div><div class="c0">21</div><div class="c0">21</div><div class="c0">22</div><div class="c0">22</div><div class="c0">22</div><div class="c0">22</div><div class="c0">23</div><div class="c0">23</div><div class="c0">23</div><div class="c0">23</div>
<div class="lbl">6</div><div class="c0">24</div><div class="c0">24</div><div class="c0">24</div><div class="c0">24</div><div class="c0">25</div><div class="c0">25</div><div class="c0">25</div><div class="c0">25</div><div class="c0">26</div><div class="c0">26</div><div class="c0">26</div><div class="c0">26</div><div class="c0">27</div><div class="c0">27</div><div class="c0">27</div><div class="c0">27</div>
<div class="lbl">7</div><div class="c0">28</div><div class="c0">28</div><div class="c0">28</div><div class="c0">28</div><div class="c0">29</div><div class="c0">29</div><div class="c0">29</div><div class="c0">29</div><div class="c0">30</div><div class="c0">30</div><div class="c0">30</div><div class="c0">30</div><div class="c0">31</div><div class="c0">31</div><div class="c0">31</div><div class="c0">31</div>
<div class="lbl">8</div><div class="c1">0</div><div class="c1">0</div><div class="c1">0</div><div class="c1">0</div><div class="c1">1</div><div class="c1">1</div><div class="c1">1</div><div class="c1">1</div><div class="c1">2</div><div class="c1">2</div><div class="c1">2</div><div class="c1">2</div><div class="c1">3</div><div class="c1">3</div><div class="c1">3</div><div class="c1">3</div>
<div class="lbl">9</div><div class="c1">4</div><div class="c1">4</div><div class="c1">4</div><div class="c1">4</div><div class="c1">5</div><div class="c1">5</div><div class="c1">5</div><div class="c1">5</div><div class="c1">6</div><div class="c1">6</div><div class="c1">6</div><div class="c1">6</div><div class="c1">7</div><div class="c1">7</div><div class="c1">7</div><div class="c1">7</div>
<div class="lbl">10</div><div class="c1">8</div><div class="c1">8</div><div class="c1">8</div><div class="c1">8</div><div class="c1">9</div><div class="c1">9</div><div class="c1">9</div><div class="c1">9</div><div class="c1">10</div><div class="c1">10</div><div class="c1">10</div><div class="c1">10</div><div class="c1">11</div><div class="c1">11</div><div class="c1">11</div><div class="c1">11</div>
<div class="lbl">11</div><div class="c1">12</div><div class="c1">12</div><div class="c1">12</div><div class="c1">12</div><div class="c1">13</div><div class="c1">13</div><div class="c1">13</div><div class="c1">13</div><div class="c1">14</div><div class="c1">14</div><div class="c1">14</div><div class="c1">14</div><div class="c1">15</div><div class="c1">15</div><div class="c1">15</div><div class="c1">15</div>
<div class="lbl">12</div><div class="c1">16</div><div class="c1">16</div><div class="c1">16</div><div class="c1">16</div><div class="c1">17</div><div class="c1">17</div><div class="c1">17</div><div class="c1">17</div><div class="c1">18</div><div class="c1">18</div><div class="c1">18</div><div class="c1">18</div><div class="c1">19</div><div class="c1">19</div><div class="c1">19</div><div class="c1">19</div>
<div class="lbl">13</div><div class="c1">20</div><div class="c1">20</div><div class="c1">20</div><div class="c1">20</div><div class="c1">21</div><div class="c1">21</div><div class="c1">21</div><div class="c1">21</div><div class="c1">22</div><div class="c1">22</div><div class="c1">22</div><div class="c1">22</div><div class="c1">23</div><div class="c1">23</div><div class="c1">23</div><div class="c1">23</div>
<div class="lbl">14</div><div class="c1">24</div><div class="c1">24</div><div class="c1">24</div><div class="c1">24</div><div class="c1">25</div><div class="c1">25</div><div class="c1">25</div><div class="c1">25</div><div class="c1">26</div><div class="c1">26</div><div class="c1">26</div><div class="c1">26</div><div class="c1">27</div><div class="c1">27</div><div class="c1">27</div><div class="c1">27</div>
<div class="lbl">15</div><div class="c1">28</div><div class="c1">28</div><div class="c1">28</div><div class="c1">28</div><div class="c1">29</div><div class="c1">29</div><div class="c1">29</div><div class="c1">29</div><div class="c1">30</div><div class="c1">30</div><div class="c1">30</div><div class="c1">30</div><div class="c1">31</div><div class="c1">31</div><div class="c1">31</div><div class="c1">31</div>
</div></div>
<p class="legend"><span class="c0"></span>warp 0（行 0–7）<span class="c1"></span>warp 1（行 8–15）<span class="c3"></span>warp 0 的 lane 5：第 1 行第 4–7 列，4 个寄存器</p>
:::

::: flow 从四个参数到一张图
cfg cw=250 nw=170
n spt 0 0 "size_per_thread [1, 4]" "一个线程：1×4 的小块" k=pass w=1.05
n tpw 1 0 "threads_per_warp [8, 4]" "一个 warp：8×16" k=teal w=1.05
n wpc 2 0 "warps_per_cta [2, 1]" "一个 CTA tile：16×16" k=vio w=1.05
n ord 1 1.3 "order [1, 0]" "lane 先沿列编号，再沿行" k=warn
e spt -> tpw "× 4 × 4"
e tpw -> wpc "× 2"
e ord -> tpw dash
cap: 每一层都是“把上一层的小块按某个形状摆开”：一个线程拿 1×4，32 个 lane 按 8×4 摆成 8×16，2 个 warp 按 2×1 摆成 16×16。order 决定 lane 0、1、2… 是先往右排还是先往下排。
:::

::: callout analogy 剧院售票
一个剧院 16 排 × 16 座。每张票是连着的 4 个座位（size_per_thread = [1, 4]）。每个售票员（warp）手下有 32 个票贩子（lane），按 8 排 × 4 组的方式分座位（threads_per_warp = [8, 4]），order = [1, 0] 意思是票贩子先在同一排里从左到右编号，排满一排再到下一排。两个售票员一个管前 8 排、一个管后 8 排（warps_per_cta = [2, 1]）。想知道 9 排 6 座是谁卖的，就按这个规则倒推：后半场 → 售票员 1；第 1 排（相对）、第 2 组 → 票贩子 5；组内第 3 个座 → 第 2 号寄存器。
:::

## ② 坐标公式和 order

[[here:solutions/g1_layouts.py@L78-L95|thread_elements（78–95 行）]] 的核心是这一行：

```python
coord[d] = (rep_c[d] * tile[d] + warp_c[d] * spt[d] * tpw[d] + lane_c[d] * spt[d] + s_c[d]) % shape[d]
```

四项从粗到细：第几个 tile 重复、warp 在 tile 里的位置、lane 在 warp 里的位置、元素在线程的小块里的位置。

**order 的作用在 `_delinearize`**（[[here:solutions/g1_layouts.py@L52-L58|52–58 行]]）：把一个一维编号（lane 号、warp 号）拆成多维坐标，`order[0]` 那一维变得最快。

**手算 lane 5**（tpw = [8, 4]，order = [1, 0]）：

| 步骤 | d | 计算 | 结果 |
|---|---|---|---|
| 先处理 order[0] = 1（列） | 1 | 5 % 4 = 1，剩下 5 // 4 = 1 | lane_c[1] = 1 |
| 再处理 order[1] = 0（行） | 0 | 1 % 8 = 1 | lane_c[0] = 1 |

lane 5 在 warp 里位于第 1 行第 1 组。warp 0 的 warp_c = [0, 0]。代入公式：行 = 0 + 0 + 1·1 + 0 = 1，列 = 0 + 0 + 1·4 + s = 4..7。所以 warp 0 lane 5 持有 (1, 4)、(1, 5)、(1, 6)、(1, 7)，和图里一致。

**反过来查**：[[here:solutions/g1_layouts.py@L98-L106|owners((9, 6))]] 返回 `[(1, 5, 2)]`：warp 1、lane 5、第 2 号寄存器。

**寄存器编号的约定**（docstring 17–19 行）：同一个小块里的元素先编号，然后才是重复。这和 Triton 对 blocked layout 的寄存器编号方式一致。

## ③ 重复和广播

[[here:solutions/g1_layouts.py@L69-L75|reps()]]：张量比 tile 大时，同一个图案沿那一维**重复**，每个线程拿到更多寄存器；比 tile 小时，坐标**取模回绕**，几个线程持有同一个元素（广播）。

| 张量形状 | reps | warp 0 lane 0 持有的元素 | 现象 |
|---|---|---|---|
| [16, 16] | (1, 1) | (0, 0..3) | 正好一个 tile，4 个寄存器 |
| [32, 16] | (2, 1) | (0, 0..3) 和 (16, 0..3) | 沿行重复两次，8 个寄存器 |
| [8, 16] | (1, 1) | (0, 0..3) | tile 有 16 行，取模后 warp 1 也落在第 0–7 行：`owners((0,0))` 是 warp 0 lane 0 **和** warp 1 lane 0 |

广播意味着两个 warp 在做完全相同的事，算力浪费了一半。layout 和张量形状不匹配时就会出现这种情况。

## ④ 用 layout 回答三个实际问题

### 问题 1：tl.sum 沿哪个轴便宜？

[[here:solutions/g1_layouts.py@L123-L136|reduce_plan（123–136 行）]] 把一次归约拆成 C2 讲过的三层：

| | axis = 1（沿列，行内求和） | axis = 0（沿行，列内求和） |
|---|---|---|
| 线程内（in_thread） | spt[1] × reps[1] = **4** 个数在寄存器里加 | spt[0] × reps[0] = 1 |
| warp 内 shuffle 轮数 | log₂ tpw[1] = log₂ 4 = **2** | log₂ tpw[0] = log₂ 8 = 3 |
| 跨 warp（shared memory + barrier） | wpc[1] = 1：**不需要** | wpc[0] = 2：**需要** |

::: flow 同一个 layout，两个方向的归约代价
cfg cw=230 nw=175
n a1 0 0 "tl.sum(x, axis=1)" "每行 16 个数" k=pass
n a1b 1 0 "线程内 4 个 + 2 轮 shuffle" "" k=pass w=1.1
n a1c 2.15 0 "结束" "没有 barrier" k=pass
n a0 0 1.3 "tl.sum(x, axis=0)" "每列 16 个数" k=todo
n a0b 1 1.3 "1 个 + 3 轮 shuffle" "" k=warn w=1.1
n a0c 2.15 1.3 "跨 warp" "shared memory + barrier" k=todo
e a1 -> a1b
e a1b -> a1c
e a0 -> a0b
e a0b -> a0c
cap: 这就是 T2、T3 里“一行一个 program、行内归约”的底层代价。好的 layout 让归约方向上的数据尽量留在一个线程、一个 warp 里。
:::

### 问题 2：load 是否合并？

[[here:solutions/g1_layouts.py@L139-L153|warp_load_sectors（139–153 行）]]：一个 warp 的一条向量化 load 碰到了多少个 32 字节 sector，和实际请求的字节数比。

| layout | dtype | 碰到的 sector | 请求的字节 | 效率 |
|---|---|---|---|---|
| 本页的例子 [1,4]、order [1,0] | fp32 | 16 | 512 | 16 × 32 = 512，**100%** |
| 同上 | fp16 | 8 | 256 | **100%** |
| 按列切：spt [4,1]、order [0,1]，形状 [64,16] | fp32 | 32 | 512 | 32 × 32 = 1024，**50%** |

第一种：每个 lane 拿一行里连续的 4 个 fp32（16 字节，一次向量化读取），同一行的 4 个 lane 拼成连续的 64 字节。第二种：每个 lane 拿的是一**列**里的 4 个，行优先的张量里它们相隔一整行，一个 sector 只用到一部分。

### 问题 3：归约之后是什么 layout？

`tl.sum(x, axis=1)` 的结果是一维的 `[16]`，它的 layout 是 `SliceLayout(dim=1, parent)`：把父 layout 的第 1 维去掉（[[here:solutions/g1_layouts.py@L109-L120|slice_elements（109–120 行）]]）。

warp 0 的 lane 0–7 在 SliceLayout 下持有的元素：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 90px repeat(8, minmax(44px, 1fr)); min-width: 480px">
<div class="lbl">lane</div><div>0</div><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>7</div>
<div class="lbl">持有</div><div class="c0">(0,)</div><div class="c0">(0,)</div><div class="c0">(0,)</div><div class="c0">(0,)</div><div class="c1">(1,)</div><div class="c1">(1,)</div><div class="c1">(1,)</div><div class="c1">(1,)</div>
</div></div>
<p class="legend">父 layout 里第 0 行的 16 个元素分在 lane 0–3 手里，求和之后这 4 个 lane 都持有同一个结果：第 0 行的和。SliceLayout 天然带着广播。</p>
:::

这就是为什么 T2 里 `x − tl.max(x, axis=0)` 可以直接相减：max 的结果是 SliceLayout，它的每个值已经在需要它的每个线程手里了，广播回父 layout 不需要任何数据搬运。反过来，如果两个张量的 layout 不兼容（比如一个 blocked、一个 slice 但父 layout 不同），Triton 会插入一次 `convert_layout`，通常要经过 shared memory，很贵。

## ⑤ 对照 Triton 源码

- **BlockedLayout 的定义和官方说明**：[[triton:include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td@L752|BlockedEncodingAttr]]，文档里有 [[triton:include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td@L756-L830|sizePerThread = {2, 2} 的图示例子]]；[[triton:include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td@L1363|SliceEncodingAttr]]。
- **Gluon**（Triton 的底层方言，可以显式指定 layout）：[[triton:python/triton/experimental/gluon/language/_layouts.py@L60|gl.BlockedLayout]] · [[triton:python/triton/experimental/gluon/language/_layouts.py@L122|gl.SliceLayout]]
- **Gluon tutorial 02-layouts**：[[triton:python/tutorials/gluon/02-layouts.py@L22-L36|size_per_thread=(2, 4) 的例子和 block shape 的算法]]，[[triton:python/tutorials/gluon/02-layouts.py@L464-L465|用 SliceLayout 生成一维下标]]，以及用 [[triton:python/tutorials/gluon/02-layouts.py@L165|memcpy kernel]] 测不同 layout 的吞吐。
- 编译器内部所有 layout 最终都转换成 linear layout：[[triton:lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp|LinearLayoutConversions.cpp]]。G2 的 CuTe 布局代数是同一类思想。

## ⑥ 常见坑与自测

- **order 理解反了**：order[0] 是**最快**变化的维度。行优先的二维张量通常是 order = [1, 0]。
- **把 tpw 的乘积写成不等于 32**：参考解在 [[here:solutions/g1_layouts.py@L37|__post_init__]] 里断言了。
- **以为张量小于 tile 时线程会闲着**：其实是取模回绕、重复持有（广播）。
- **寄存器顺序**：先小块内、再重复，弄反了 `thread_elements` 的顺序就和测试对不上。

::: answer 自测 1：同样的 layout，形状 [16, 32]。warp 0 lane 5 持有哪些元素？tl.sum(axis=1) 的 in_thread 是多少？
tile 是 [16, 16]，reps = (1, 2)，沿列重复两次。lane 5 持有 (1, 4..7) 和 (1, 20..23)，共 8 个寄存器。in_thread = spt[1] × reps[1] = 4 × 2 = 8。
:::

::: answer 自测 2：想让 tl.sum(axis=0) 也不需要跨 warp，warps_per_cta 应该怎么改？
改成 [1, 2]：两个 warp 左右摆，每个 warp 负责全部 16 行中的一半列。这时 tpw = [8, 4] 只覆盖 8 行，tile 变成 [8, 32]，对 [16, 16] 的张量沿行要重复 2 次、沿列取模回绕（广播）。所以还要同时调整 tpw（比如 [16, 2]），让 tile 重新和形状匹配。layout 的几个参数总是要一起调。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/g1_layouts.py|solutions/g1_layouts.py]] · [[here:mine/g1_layouts.py|mine/g1_layouts.py]] · [[here:tests/test_g1_layouts.py|tests/test_g1_layouts.py]] · [[here:content/g1.md|content/g1.md]]

**Triton**：[[triton:include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td@L752|BlockedEncodingAttr]] · [[triton:python/tutorials/gluon/02-layouts.py|Gluon 02-layouts]] · [[triton:python/tutorials/gluon/01-intro.py|Gluon 01-intro]]

**相关讲解**：[C2 reduction](c2_reduce_sum.html)（归约的三层）· [T1](t1_add_gelu.html)（编译器怎么拆一维向量）· [T2](t2_softmax.html)（tl.max 的广播）
