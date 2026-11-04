title: C3 讲解：转置里的合并访存与 bank conflict
lede: 转置一个数都不算，却能把带宽从 ~800 GB/s 提到 ~1400 GB/s。逐行读 naive 和 tiled 两个 kernel：二维索引、为什么写入不合并、shared memory 中转时 block 坐标怎么对调，以及那一列 padding 怎么消掉 bank conflict。

::: callout info 怎么读这一页
对照 [[here:solutions/c3_transpose.cu|c3_transpose.cu]] 读。最容易绕晕的是第 ③ 节的“坐标对调”，那里用一个具体的线程把所有下标都算了一遍。
:::

## ① 先看全貌：读和写不可能同时连续

`y = x.T`，x 是 `[M, N]`，y 是 `[N, M]`，都是行优先。`y[c][r] = x[r][c]`。

- 如果让相邻线程走 x 的**相邻列**（c 连续）：读 `x[r][c]` 是连续的，写 `y[c][r]` 时相邻线程的地址相差 M 个 float，**跨步写**。
- 如果反过来让相邻线程走 x 的相邻行：写连续了，读又变成跨步。

两头总有一头是跨步的。解决办法是**找一个不怕跨步的中转站**：shared memory。

::: flow naive vs tiled：跨步访问发生在哪里
cfg cw=240 nw=170
n x1 0 0 "x（显存）" "按行读：合并" k=acc
n y1 1 0 "y（显存）" "按列写：跨步" k=todo
n x2 0 1.3 "x（显存）" "按行读：合并" k=acc
n sm 1 1.3 "tile[32][33]" "shared memory" k=teal
n y2 2 1.3 "y（显存）" "按行写：合并" k=pass
g 0 0 1 0 "naive" k=todo
g 0 1.3 2 1.3 "tiled" k=pass
e x1 -> y1 "直接" k=todo
e x2 -> sm "合并"
e sm -> y2 "合并" k=pass
cap: tiled 版本把“跨步”挪进了 shared memory：片上内存没有 sector 的概念，跨步读不浪费带宽，只需要避开 bank conflict。
:::

RTX 5090 上 8192×8192 的实测（题目页的数据）：naive 805 GB/s，`x.t().contiguous()` 804 GB/s，tiled **1409 GB/s**。

::: callout analogy 搬书换书架
旧书架上的书按行摆，新书架要求按列摆。naive 的做法：拿起一本书，走到新书架对应的位置放下，再回来拿下一本。旧书架这边是顺手一排排拿的，新书架那边每放一本都要跑到另一列：腿都跑断了（跨步写）。tiled 的做法：先从旧书架一次抱一摞（32×32）到手边的桌子上（shared memory），在桌上换个方向，再一排排地放上新书架。桌子就在手边，在桌上怎么挪都很快。
:::

## ② naive kernel：二维 launch 和二维下标

[[here:solutions/c3_transpose.cu@L18-L24|transpose_naive_kernel（18–24 行）]] · host 端 [[here:solutions/c3_transpose.cu@L66-L67|dim3 block / grid（66–67 行）]]

```cpp
dim3 block(32, 8);                                         // 每个 block 32 × 8 = 256 个线程
dim3 grid((N + 31) / 32, (M + 7) / 8);                     // x 方向覆盖列，y 方向覆盖行

const int c = blockIdx.x * blockDim.x + threadIdx.x;       // 列：快维度
const int r = blockIdx.y * blockDim.y + threadIdx.y;       // 行
if (r < M && c < N) y[(int64_t)c * M + r] = x[(int64_t)r * N + c];
```

- `dim3` 是三个分量的结构体，没写的分量默认是 1。block 和 grid 都可以是二维的，kernel 里用 `.x`、`.y` 分别取。
- **约定 `threadIdx.x` 对应内存中连续的那一维**。warp 是按 `threadIdx.x` 优先切分的：一个 warp = 同一个 `threadIdx.y` 下的 32 个连续 `threadIdx.x`。所以一个 warp 的 32 个线程有相同的 r、连续的 c。
- 二维下标展开成一维地址：`x[r][c]` 在行优先的 `[M, N]` 里是 `r * N + c`；`y[c][r]` 在 `[N, M]` 里是 `c * M + r`。先转 `int64_t` 再乘，8192 × 8192 已经接近 int 的上限。
- 越界检查要同时检查两个维度。

同一个 warp 里，读写地址长这样（M = N = 8192）：

| lane | c | 读 `x[r*N + c]` | 写 `y[c*M + r]` |
|---|---|---|---|
| 0 | 0 | r·8192 + 0 | 0 + r |
| 1 | 1 | r·8192 + 1 | **8192** + r |
| 2 | 2 | r·8192 + 2 | **16384** + r |
| … | … | 连续，4 个 sector | 每个相隔 32 KB，**32 个 sector** |

写入要 32 个独立的 32 字节 sector，每个只用 4 字节，有效带宽只剩 1/8。

## ③ tiled kernel：最绕的是坐标对调

[[here:solutions/c3_transpose.cu@L26-L53|transpose_tiled_kernel（26–53 行）]]

每个 block 负责 x 的一个 32×32 tile，用 32 × 8 = 256 个线程，每个线程搬 4 个元素。分两个阶段，中间一个 barrier。

#### 先看动画：一个 tile 的两个阶段

::: raw
<style>
:root { --an-l: 86%; --an-s: 65%; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --an-l: 32%; --an-s: 45%; } }
:root[data-theme="dark"] { --an-l: 32%; --an-s: 45%; }
.anim { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); box-shadow: var(--shadow); margin: 16px 0; overflow: hidden; }
.anim-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 7px 12px; background: var(--surface-2); border-bottom: 1px solid var(--border); font-size: 14px; }
.anim-head b { margin-right: auto; }
.anim-head .btn { font-size: 12.5px; padding: 1px 9px; }
.anim-head .btn.on { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.anim-body { padding: 12px; }
.anim-stage { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; align-items: flex-start; }
.anim-panel { text-align: center; }
.anim-panel .cap { font-size: 12.5px; color: var(--muted); margin-bottom: 4px; font-family: var(--mono); }
.anim-grid { display: grid; gap: 2px; justify-content: center; }
.an-cell { width: 26px; height: 26px; border-radius: 4px; border: 1px solid var(--border); background: var(--surface-2);
  font: 10.5px/24px var(--mono); color: var(--muted); text-align: center; transition: background .25s, transform .2s, box-shadow .2s; position: relative; }
.an-cell.fill { background: hsl(var(--h) var(--an-s) var(--an-l)); color: var(--text); }
.an-cell.pad { border-style: dashed; background: none; color: var(--muted); font-size: 9px; }
.an-cell.act { box-shadow: 0 0 0 2px var(--accent); transform: scale(1.08); z-index: 1; }
.an-cell.src { box-shadow: 0 0 0 2px var(--f-warn); }
.an-cell.pop { animation: anpop .45s ease-out; }
@keyframes anpop { 0% { transform: scale(.6); } 60% { transform: scale(1.15); } 100% { transform: scale(1); } }
.an-arrow { align-self: center; font-size: 22px; color: var(--muted); }
.anim-status { margin: 12px 0 6px; padding: 7px 10px; border-radius: 8px; background: var(--surface-2); font-size: 13.5px; min-height: 2.6em; }
.anim-code { margin: 6px 0; font: 12.5px/1.7 var(--mono); background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; overflow-x: auto; white-space: pre; }
.anim-code .ln { display: block; padding: 0 4px; border-radius: 4px; color: var(--muted); }
.anim-code .ln.hot { background: var(--accent-soft); color: var(--text); }
.anim-meter { display: grid; grid-template-columns: 96px 1fr 70px; gap: 6px 10px; align-items: center; font-size: 12.5px; margin-top: 8px; }
.anim-meter .pills { display: flex; flex-wrap: wrap; gap: 2px; min-height: 12px; }
.anim-meter .pill { width: 10px; height: 10px; border-radius: 2px; background: var(--f-acc); }
.anim-meter .pill.w { background: var(--f-todo); }
.anim-meter .pill.new { animation: anpop .4s ease-out; }
.anim-meter .num { font-family: var(--mono); text-align: right; }
.anim-lanes { display: flex; gap: 4px; justify-content: center; margin: 8px 0 2px; flex-wrap: wrap; }
.anim-lane { width: 44px; border-radius: 6px; border: 1px solid var(--border); font: 11px/1.4 var(--mono); padding: 2px 0; text-align: center; background: var(--surface); transition: background .2s, box-shadow .2s; }
.anim-lane.wait { box-shadow: 0 0 0 2px var(--f-warn); }
.anim-lane.now { background: var(--f-acc-bg); box-shadow: 0 0 0 2px var(--accent); }
.anim-lane.done { background: var(--f-pass-bg); color: var(--muted); }
.anim-banks { display: flex; gap: 4px; justify-content: center; flex-wrap: wrap; }
.anim-bank { width: 44px; min-height: 58px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); display: flex; flex-direction: column-reverse; align-items: center; gap: 2px; padding: 3px 0; font: 10.5px var(--mono); color: var(--muted); }
.anim-bank .q { width: 30px; height: 7px; border-radius: 2px; background: var(--f-warn); }
.anim-bank.busy { box-shadow: 0 0 0 2px var(--f-todo); }
.anim-note { font-size: 12px; color: var(--muted); margin-top: 6px; }
@media (prefers-reduced-motion: reduce) { .an-cell, .an-cell.pop, .anim-meter .pill.new { animation: none !important; transition: none !important; } }
</style>
<div class="anim" id="an-tr">
  <div class="anim-head">
    <b>转置动画（缩小版：8×8 的 tile，一个 warp 8 个 lane）</b>
    <button type="button" class="btn on" data-mode="tiled">tiled</button>
    <button type="button" class="btn" data-mode="naive">naive</button>
    <span style="width:8px"></span>
    <button type="button" class="btn" data-act="reset" title="重置">⏮</button>
    <button type="button" class="btn" data-act="prev" title="上一步">◀</button>
    <button type="button" class="btn" data-act="play" title="播放 / 暂停">▶ 播放</button>
    <button type="button" class="btn" data-act="next" title="下一步">▶|</button>
  </div>
  <div class="anim-body">
    <div class="anim-stage">
      <div class="anim-panel"><div class="cap">x（显存，行优先）</div><div class="anim-grid" data-g="x"></div></div>
      <div class="an-arrow" data-g="arrow1">→</div>
      <div class="anim-panel" data-g="smwrap"><div class="cap">tile[8][9]（shared memory）</div><div class="anim-grid" data-g="sm"></div></div>
      <div class="an-arrow" data-g="arrow2">→</div>
      <div class="anim-panel"><div class="cap">y = xᵀ（显存，行优先）</div><div class="anim-grid" data-g="y"></div></div>
    </div>
    <div class="anim-status" data-g="status"></div>
    <div class="anim-code" data-g="code"></div>
    <div class="anim-meter">
      <span>读 x 的事务</span><span class="pills" data-g="pr"></span><span class="num" data-g="nr">0</span>
      <span>写 y 的事务</span><span class="pills" data-g="pw"></span><span class="num" data-g="nw">0</span>
    </div>
    <div class="anim-note">格子里的两位数是元素在 x 里的 (行, 列)，颜色表示它在 x 里的行。每一步是一个 warp 发出的一条访存指令；一个小方块代表一次内存事务（这里假设一行 8 个 float 正好是一次事务）。</div>
  </div>
</div>
<script>
(function () {
  var root = document.getElementById("an-tr"), Nn = 8;
  function q(s) { return root.querySelector('[data-g="' + s + '"]'); }
  function mk(g, rows, cols, pad) {
    g.style.gridTemplateColumns = "repeat(" + (cols + (pad ? 1 : 0)) + ", 26px)";
    var cells = [];
    for (var r = 0; r < rows; r++) { cells.push([]); for (var c = 0; c < cols + (pad ? 1 : 0); c++) {
      var d = document.createElement("div"); d.className = "an-cell" + (c === cols ? " pad" : "");
      if (c === cols) d.textContent = "pad"; g.appendChild(d); cells[r].push(d); } }
    return cells;
  }
  var X = mk(q("x"), Nn, Nn), S = mk(q("sm"), Nn, Nn, true), Y = mk(q("y"), Nn, Nn);
  for (var r = 0; r < Nn; r++) for (var c = 0; c < Nn; c++) setv(X[r][c], r, c);
  function setv(el, r, c) { el.className = "an-cell fill"; el.style.setProperty("--h", (r * 45) % 360); el.textContent = "" + r + c; }
  function clear(el) { el.className = "an-cell"; el.textContent = ""; el.style.removeProperty("--h"); }
  var CODE = {
    tiled: ["// 阶段 1：按行读 x，写进 tile 的同一行", "tile[ty][tx] = x[r * N + c];", "__syncthreads();", "// 阶段 2：按列读 tile，按行写 y", "y[yr * M + yc] = tile[tx][ty];"],
    naive: ["// 每个线程：读 x[r][c]，写 y[c][r]", "y[c * M + r] = x[r * N + c];"]
  };
  var mode = "tiled", step = 0, timer = null;
  function maxStep() { return mode === "tiled" ? 17 : 8; }
  function render(animate) {
    var tiled = mode === "tiled";
    q("smwrap").style.display = tiled ? "" : "none"; q("arrow2").style.display = tiled ? "" : "none";
    for (var r = 0; r < Nn; r++) for (var c = 0; c < Nn; c++) {
      X[r][c].classList.remove("act", "src");
      var sf = tiled && r < Math.min(step, 8); if (sf) setv(S[r][c], r, c); else clear(S[r][c]);
      var yf = tiled ? r < step - 9 : c < step;               // y[r][c] = x[c][r]
      if (yf) setv(Y[r][c], c, r); else clear(Y[r][c]);
    }
    var hot = -1, msg = "", nr = 0, nw = 0, newR = 0, newW = 0;
    if (tiled) {
      nr = Math.min(step, 8); nw = Math.max(0, step - 9);
      if (step === 0) msg = "按 ▶ 播放，或用 ▶| 一步一步看。tiled 版本：先把整块 x 搬进 shared memory，再换个方向写出去。";
      else if (step <= 8) { var i = step - 1; hot = 1; newR = 1;
        for (var c = 0; c < Nn; c++) { X[i][c].classList.add("src"); S[i][c].classList.add("act"); if (animate) S[i][c].classList.add("pop"); }
        msg = "阶段 1，第 " + step + " 条指令：8 个 lane 读 x 第 " + i + " 行的 8 个<b>连续</b> float（1 次事务），写进 tile 第 " + i + " 行。"; }
      else if (step === 9) { hot = 2;
        for (var r = 0; r < Nn; r++) for (var c = 0; c < Nn; c++) { S[r][c].classList.add("act"); }
        msg = "barrier：所有线程都写完 tile 之后才放行。阶段 2 里每个线程要读<b>别的线程</b>写的数据。"; }
      else { var k = step - 10; hot = 4; newW = 1;
        for (var l = 0; l < Nn; l++) { S[l][k].classList.add("src"); Y[k][l].classList.add("act"); if (animate) Y[k][l].classList.add("pop"); }
        msg = "阶段 2，第 " + (step - 9) + " 条指令：lane l 读 tile[l][" + k + "]（按<b>列</b>读 shared memory，不怕跨步），写 y 第 " + k + " 行的 8 个<b>连续</b>位置（1 次事务）。"; }
    } else {
      nr = step; nw = step * 8;
      if (step === 0) msg = "naive 版本：每个线程直接读 x[r][c]、写 y[c][r]。看看写 y 的事务有多少。";
      else { var i = step - 1; hot = 1; newR = 1; newW = 8;
        for (var c = 0; c < Nn; c++) { X[i][c].classList.add("src"); Y[c][i].classList.add("act"); if (animate) Y[c][i].classList.add("pop"); }
        msg = "第 " + step + " 条指令：读 x 第 " + i + " 行（连续，1 次事务），但写的是 y 的第 " + i + " <b>列</b>：8 个地址各相隔一整行，要 <b>8 次</b>事务。"; }
    }
    if (step === maxStep()) msg += tiled ? "　<b>完成：读 8 次、写 8 次。</b>" : "　<b>完成：读 8 次、写 64 次。</b>";
    q("status").innerHTML = msg;
    q("code").innerHTML = CODE[mode].map(function (s, j) { return '<span class="ln' + (j === hot ? " hot" : "") + '">' + s.replace(/</g, "&lt;") + "</span>"; }).join("");
    pills(q("pr"), nr, "", animate ? newR : 0); pills(q("pw"), nw, "w", animate ? newW : 0);
    q("nr").textContent = nr; q("nw").textContent = nw;
  }
  function pills(box, n, cls, fresh) {
    box.innerHTML = ""; for (var i = 0; i < n; i++) { var p = document.createElement("span");
      p.className = "pill " + cls + (i >= n - fresh ? " new" : ""); box.appendChild(p); }
  }
  function go(s, animate) { step = Math.max(0, Math.min(maxStep(), s)); render(animate); }
  function stop() { if (timer) { clearInterval(timer); timer = null; } playBtn.textContent = "▶ 播放"; }
  var playBtn = root.querySelector('[data-act="play"]');
  root.querySelector('[data-act="play"]').onclick = function () {
    if (timer) return stop();
    if (step >= maxStep()) go(0, false);
    playBtn.textContent = "⏸ 暂停";
    timer = setInterval(function () { if (step >= maxStep()) return stop(); go(step + 1, true); }, 900);
  };
  root.querySelector('[data-act="next"]').onclick = function () { stop(); go(step + 1, true); };
  root.querySelector('[data-act="prev"]').onclick = function () { stop(); go(step - 1, false); };
  root.querySelector('[data-act="reset"]').onclick = function () { stop(); go(0, false); };
  root.querySelectorAll("[data-mode]").forEach(function (b) {
    b.onclick = function () { stop(); mode = b.dataset.mode;
      root.querySelectorAll("[data-mode]").forEach(function (x) { x.classList.toggle("on", x === b); }); go(0, false); };
  });
  go(0, false);
})();
</script>
:::

动画里最关键的是两种模式下“写 y 的事务”那一行：tiled 写 8 次，naive 写 64 次。真实的 32×32 tile 上差距是 32 倍。tiled 版本的代价是阶段 2 按**列**读 shared memory，这就是第 ④ 节 bank conflict 的来源。


### 阶段 1：把 x 的 tile 按行读进 shared memory

[[here:solutions/c3_transpose.cu@L33-L39|33–39 行]]

```cpp
const int c  = blockIdx.x * TILE + threadIdx.x;   // x 的列
const int r0 = blockIdx.y * TILE + threadIdx.y;   // 这个线程负责的第一行
for (int j = 0; j < TILE; j += ROWS) {            // j = 0, 8, 16, 24
  const int r = r0 + j;
  if (r < M && c < N) tile[threadIdx.y + j][threadIdx.x] = x[(int64_t)r * N + c];
}
```

block 只有 8 行线程，tile 有 32 行，所以每个线程隔 8 行搬一次，共 4 次：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 110px repeat(8, minmax(52px, 1fr)); min-width: 560px">
<div class="lbl">tile 行</div><div>0–7</div><div>8–15</div><div>16–23</div><div>24–31</div><div class="lbl"></div><div class="lbl"></div><div class="lbl"></div><div class="lbl"></div>
<div class="lbl">谁来搬</div><div class="c0">j = 0<small>ty 0–7</small></div><div class="c1">j = 8<small>ty 0–7</small></div><div class="c2">j = 16<small>ty 0–7</small></div><div class="c3">j = 24<small>ty 0–7</small></div><div class="lbl"></div><div class="lbl"></div><div class="lbl"></div><div class="lbl"></div>
</div></div>
<p class="legend">threadIdx.y = 1 的那一行线程依次搬 tile 的第 1、9、17、25 行。每次都是 32 个线程读 x 同一行的 32 个连续 float：合并。</p>
:::

为什么不直接开 32×32 = 1024 个线程、每人搬一个？可以，但每个线程只做一次 load/store，固定开销占比更高；block 太大，每个 SM 能同时驻留的 block 也更少。256 个线程、每人 4 个是常见的折中。

### barrier

[[here:solutions/c3_transpose.cu@L41|41 行]] `__syncthreads()`：阶段 2 里一个线程要读**别的线程**写进 tile 的数据（它读的是 `tile[threadIdx.x][...]`，行列反过来了）。必须等整个 tile 写完。

### 阶段 2：block 坐标对调，再按行写 y

[[here:solutions/c3_transpose.cu@L45-L52|45–52 行]]

```cpp
const int yc  = blockIdx.y * TILE + threadIdx.x;   // y 的列（= x 的行）
const int yr0 = blockIdx.x * TILE + threadIdx.y;   // y 的行（= x 的列）
for (int j = 0; j < TILE; j += ROWS) {
  const int yr = yr0 + j;
  if (yr < N && yc < M) y[(int64_t)yr * M + yc] = tile[threadIdx.x][threadIdx.y + j];
}
```

两件事同时发生：

1. **tile 级别对调**：x 的第 (by, bx) 个 tile 转置后是 y 的第 (bx, by) 个 tile。所以 y 的行用 `blockIdx.x` 算，y 的列用 `blockIdx.y` 算。
2. **tile 内部对调**：`threadIdx.x` 仍然走 y 的**列**（连续方向），保证写入合并；数据从 `tile[threadIdx.x][threadIdx.y + j]` 取，也就是在 shared memory 里**按列读**。

::: flow 一个 tile 的旅程：x 的 (by, bx) → y 的 (bx, by)
cfg cw=255 nw=180
n xt 0 0 "x 的 tile (by=5, bx=2)" "行 160–191，列 64–95" k=acc w=1.05
n sm 1 0 "tile[32][33]" "tile[i][j] = x[160+i][64+j]" k=teal w=1.05
n yt 2 0 "y 的 tile (2, 5)" "行 64–95，列 160–191" k=pass w=1.05
e xt -> sm "阶段 1"
e sm -> yt "阶段 2" k=pass
cap: 同一个 block (bx=2, by=5) 在阶段 1 读 x 的第 5 行第 2 列 tile，在阶段 2 写 y 的第 2 行第 5 列 tile。
:::

**拿一个具体线程验算一遍**：block (bx = 2, by = 5)，线程 (tx = 3, ty = 1)，只看 j = 0。

| 阶段 | 计算 | 结果 |
|---|---|---|
| 1 | c = 2·32 + 3 = 67，r = 5·32 + 1 = 161 | 读 x[161][67]，写进 **tile[1][3]** |
| 2 | yc = 5·32 + 3 = 163，yr = 2·32 + 1 = 65 | 写 y[65][163]，值取自 **tile[3][1]** |
| 验证 | tile[3][1] 是谁写的？阶段 1 里 tx = 1、ty = 3 的线程：r = 163，c = 65 | tile[3][1] = x[163][65]，而 y[65][163] 应该等于 x[163][65] ✓ |

注意同一个线程在两个阶段搬的**不是同一个元素**：它写进 tile[1][3]，却读出 tile[3][1]。这就是为什么中间必须有 barrier。

::: callout analogy 拼图块换位置
把整张图转置，就像把一幅拼图沿对角线翻过来。每一块拼图（tile）都要搬到对角位置：第 5 行第 2 列的那块搬到第 2 行第 5 列（block 坐标对调）。而且每一块本身也要翻过来（tile 内部的行列对调）。负责这块拼图的工人（block）先把它拿到工作台上（shared memory），翻过来，再放到新位置。工作台上一块拼图里的小格子谁拿都行，所以拿起来的人和放下去的人不必是同一个（不同线程，需要 barrier）。
:::

## ④ 那一列 padding：bank conflict

[[here:solutions/c3_transpose.cu@L27-L30|27–30 行]] `__shared__ float tile[TILE][TILE + 1];`

shared memory 被分成 **32 个 bank**，每个 bank 每周期能服务一个 4 字节的字。地址为 `a`（以 float 为单位）的元素在第 `a % 32` 个 bank。**一个 warp 里如果有多个线程访问同一个 bank 的不同地址，就要排队串行**，这叫 bank conflict。

#### 先用 4 个 bank 看懂：二维的 tile 其实是一条一维数组

先别管 32，缩小成 **4 个 bank、4×4 的 tile**。关键只有一条规则：**一个数在第几个 bank，只看它的地址除以 bank 数的余数**。

shared memory 其实是一条长长的一维数组，地址 0、1、2、3、4……依次排下去，bank = 地址 % 4，就是 0、1、2、3、0、1、2、3……一直循环：

```text
地址:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
bank:  0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3
```

二维的 `tile[i][j]` 是**按行一行接一行**放进这条数组的：地址 = i × 每行长度 + j。

**不加 padding（每行 4 个）**：地址 = 4i + j。把每一格的 bank 号画出来：

```text
         列0 列1 列2 列3
第0行:    0   1   2   3      ← 地址 0..3
第1行:    0   1   2   3      ← 地址 4..7
第2行:    0   1   2   3      ← 地址 8..11
第3行:    0   1   2   3      ← 地址 12..15
```

每一行都从 bank 0 开始：行长 4 正好等于 bank 数，转一圈又回到起点。读第 0 列时，4 个线程读地址 0、4、8、12，**全是 bank 0**。一个 bank 每拍只能服务一个线程，只能排队，读一次花了 4 拍。

**加 padding（每行 5 个，最后一个空着不用）**：地址 = 5i + j。

```text
         列0 列1 列2 列3 (空)
第0行:    0   1   2   3   0      ← 地址 0..4
第1行:    1   2   3   0   1      ← 地址 5..9
第2行:    2   3   0   1   2      ← 地址 10..14
第3行:    3   0   1   2   3      ← 地址 15..19
```

每一行多占了一格，**下一行的起点就往后挪了一个 bank**。再读第 0 列：地址 0、5、10、15，bank 是 0、1、2、3，各不相同，4 个 bank 同时工作，**1 拍读完**。

下面的演示把这个过程一步步做出来：先把 tile 一行一行地放进一维内存，再读第 0 列。切换“每行 4 个 / 每行 5 个”对比。

::: raw
<style>
.fl-cell { width: 36px; height: 46px; border-radius: 5px; border: 1px solid var(--border); background: var(--surface-2); display: flex; flex-direction: column;
  justify-content: space-between; align-items: center; padding: 2px 0; font: 10px/1.2 var(--mono); color: var(--muted); transition: background .25s, transform .2s, box-shadow .2s; }
.fl-cell .v { font-size: 12px; font-weight: 600; color: var(--text); }
.fl-cell.fill { background: hsl(var(--h) var(--an-s) var(--an-l)); }
.fl-cell.padc { background: none; border-style: dashed; }
.fl-cell.off { opacity: .35; }
.fl-cell.act { box-shadow: 0 0 0 2px var(--accent); transform: scale(1.06); }
.fl-cell.rd { box-shadow: 0 0 0 3px var(--f-todo); transform: scale(1.08); }
.fl-strip { display: grid; grid-template-columns: repeat(20, 36px); gap: 2px; justify-content: center; }
.fl-tile { display: grid; gap: 3px; justify-content: center; }
.fl-tile .fl-cell { width: 44px; }
.fl-rowlab { font: 11px/46px var(--mono); color: var(--muted); text-align: right; padding-right: 4px; }
</style>
<div class="anim" id="an-flat">
  <div class="anim-head">
    <b>二维 → 一维（4 个 bank，4×4 的 tile）</b>
    <button type="button" class="btn on" data-w="4">每行 4 个（不加 padding）</button>
    <button type="button" class="btn" data-w="5">每行 5 个（+1 个空位）</button>
    <span style="width:4px"></span>
    <button type="button" class="btn" data-act="reset" title="重置">⏮</button>
    <button type="button" class="btn" data-act="prev" title="上一步">◀</button>
    <button type="button" class="btn" data-act="play" title="播放 / 暂停">▶ 播放</button>
    <button type="button" class="btn" data-act="next" title="下一步">▶|</button>
  </div>
  <div class="anim-body">
    <div class="anim-panel"><div class="cap">shared memory 实际的样子：一条一维数组。每格上面是地址，下面是 bank = 地址 % 4</div>
      <div class="strip-scroll"><div class="fl-strip" data-g="strip"></div></div></div>
    <div class="anim-stage" style="margin-top:14px">
      <div class="anim-panel"><div class="cap">你在代码里看到的二维 tile[i][j]（格子下面是它的 bank）</div><div class="fl-tile" data-g="tile"></div></div>
      <div class="anim-panel"><div class="cap">读第 0 列时，每个 bank 要排几个请求</div><div class="anim-banks" data-g="banks" style="margin-top:8px"></div></div>
    </div>
    <div class="anim-status" data-g="status"></div>
    <div class="anim-note">颜色代表 bank：同一种颜色就是同一个 bank。“空”是 padding 占掉的位置，不存数据。</div>
  </div>
</div>
<script>
(function () {
  var root = document.getElementById("an-flat"), NB = 4, R = 4, C = 4, LEN = 20;
  function q(s) { return root.querySelector('[data-g="' + s + '"]'); }
  var W = 4, step = 0, timer = null, MAX = 5;
  var strip = [], bankEls = [];
  for (var a = 0; a < LEN; a++) { var d = document.createElement("div"); q("strip").appendChild(d); strip.push(d); }
  for (var b = 0; b < NB; b++) { var e = document.createElement("div"); e.className = "anim-bank"; q("banks").appendChild(e); bankEls.push(e); }
  function cell(el, top, val, bot, cls, bank) {
    el.className = "fl-cell" + (cls ? " " + cls : "");
    if (bank === undefined) el.style.removeProperty("--h"); else el.style.setProperty("--h", bank * 90);
    el.innerHTML = "<span>" + top + "</span><span class='v'>" + val + "</span><span>" + bot + "</span>";
  }
  function render() {
    var placed = Math.min(step, R), reading = step === 5;
    var col0 = []; for (var i = 0; i < R; i++) col0.push(i * W);
    // 一维内存
    for (var a = 0; a < LEN; a++) {
      var bank = a % NB, el = strip[a];
      if (a >= R * W) { cell(el, a, "—", "b" + bank, "off"); continue; }
      var i = Math.floor(a / W), j = a % W, isPad = j >= C;
      if (i < placed) {
        cell(el, a, isPad ? "空" : i + "," + j, "b" + bank, isPad ? "padc" : "fill", isPad ? undefined : bank);
        if (step >= 1 && step <= 4 && i === step - 1) el.classList.add("act");
        if (reading && col0.indexOf(a) >= 0) el.classList.add("rd");
      } else cell(el, a, "", "b" + bank, "");
    }
    // 二维 tile
    var t = q("tile"); t.innerHTML = ""; t.style.gridTemplateColumns = "48px repeat(" + W + ", 44px)";
    for (var i = 0; i < R; i++) {
      var lab = document.createElement("div"); lab.className = "fl-rowlab"; lab.textContent = "第" + i + "行"; t.appendChild(lab);
      for (var j = 0; j < W; j++) {
        var c = document.createElement("div"), a = i * W + j, bank = a % NB, isPad = j >= C;
        if (i < placed) {
          cell(c, "地址 " + a, isPad ? "空" : i + "," + j, "bank " + bank, isPad ? "padc" : "fill", isPad ? undefined : bank);
          if (step >= 1 && step <= 4 && i === step - 1) c.classList.add("act");
          if (reading && j === 0) c.classList.add("rd");
        } else cell(c, "", isPad ? "空" : i + "," + j, "", isPad ? "padc" : "");
        t.appendChild(c);
      }
    }
    // bank 队列
    var cnt = [0, 0, 0, 0]; if (reading) col0.forEach(function (a) { cnt[a % NB]++; });
    bankEls.forEach(function (el, b) {
      el.innerHTML = "<span>bank " + b + "</span>";
      for (var n = 0; n < cnt[b]; n++) { var s = document.createElement("span"); s.className = "q"; el.appendChild(s); }
      el.classList.toggle("busy", cnt[b] > 1);
    });
    // 说明
    var msg;
    if (step === 0) msg = "二维的 tile[i][j] 在内存里是按行一行接一行放的：<b>地址 = i × " + W + " + j</b>。上面这条就是内存，bank 号 0、1、2、3、0、1、2、3…… 循环。按 ▶| 把 tile 一行一行放进去。";
    else if (step <= 4) {
      var i = step - 1, s0 = i * W;
      msg = "放第 " + i + " 行：占地址 " + s0 + "～" + (s0 + W - 1) + "，这一行的<b>起点</b>在 bank " + (s0 % NB) + "。";
      if (i > 0) msg += W === 4 ? "　和上一行一样，<b>又从 bank 0 开始</b>：行长 4 正好转一圈回到原点。" : "　比上一行<b>往后错开了 1 个 bank</b>：因为每行多占了 1 个空位。";
      else msg += W === 4 ? "" : "　注意最后一格是空位，它把下一行往后推了一格。";
    } else {
      var banks = col0.map(function (a) { return a % NB; });
      msg = "读第 0 列：4 个线程读地址 " + col0.join("、") + "，bank 是 " + banks.join("、") + "。";
      msg += W === 4 ? "　<b>4 个请求全挤在 bank 0</b>，一拍只能服务一个，要排队 <b>4 拍</b>。" : "　<b>4 个 bank 各一个请求</b>，同时服务，<b>1 拍</b>读完。";
    }
    q("status").innerHTML = msg;
  }
  function go(s) { step = Math.max(0, Math.min(MAX, s)); render(); }
  var playBtn = root.querySelector('[data-act="play"]');
  function stop() { if (timer) { clearInterval(timer); timer = null; } playBtn.textContent = "▶ 播放"; }
  playBtn.onclick = function () {
    if (timer) return stop();
    if (step >= MAX) go(0);
    playBtn.textContent = "⏸ 暂停";
    timer = setInterval(function () { if (step >= MAX) return stop(); go(step + 1); }, 1200);
  };
  root.querySelector('[data-act="next"]').onclick = function () { stop(); go(step + 1); };
  root.querySelector('[data-act="prev"]').onclick = function () { stop(); go(step - 1); };
  root.querySelector('[data-act="reset"]').onclick = function () { stop(); go(0); };
  root.querySelectorAll("[data-w]").forEach(function (b) {
    b.onclick = function () { stop(); W = +b.dataset.w;
      root.querySelectorAll("[data-w]").forEach(function (x) { x.classList.toggle("on", x === b); }); go(0); };
  });
  render();
})();
</script>
:::

**一句话**：问题不在“读一列”本身，而在于**行长正好等于 bank 数时，每一行都从同一个 bank 开始，于是同一列的元素全落在同一个 bank 上**。每行多加一个空位，行长从 32 变成 33，每一行的起点就错开一个 bank，同一列的 32 个元素正好分散到 32 个不同的 bank。代价是每行浪费一个 float，32 行一共 128 字节。

::: callout analogy 店员排班
一家店有 4 个店员（bank），顾客按号码分配：0 号给店员 0，1 号给店员 1……4 号又回到店员 0，依次循环。

- 每批正好来 4 个人（每行 4 个），每批的**第一个人**拿到的号是 0、4、8、12，永远分给店员 0。你要找“每批的第一个人”（读一列），他们全挤在店员 0 那里排队。
- 每批 4 个人再加一个“空号”（padding），一批占 5 个号。每批第一个人的号变成 0、5、10、15，分别分给店员 0、1、2、3，4 个店员同时接待。
:::

把上面的 4 换成 32，就是真实的硬件：

阶段 2 里一个 warp 读的是 `tile[0..31][k]`（同一列，tx 变化）：

- **不加 padding**（每行 32 个）：`tile[i][k]` 的地址是 `32i + k`，bank = `k`。32 个线程全在同一个 bank，**32 路冲突**，读一次变成 32 次。
- **加 1 列 padding**（每行 33 个）：地址是 `33i + k`，bank = `(i + k) % 32`。i = 0..31 时恰好覆盖 32 个不同的 bank，**无冲突**。

#### 动画：bank conflict 是怎么排队的

::: raw
<div class="anim" id="an-bank">
  <div class="anim-head">
    <b>bank 动画（缩小版：8 个 bank，一个 warp 8 个 lane）</b>
    <button type="button" class="btn on" data-pad="0">tile[8][8]</button>
    <button type="button" class="btn" data-pad="1">tile[8][9]</button>
    <span style="width:4px"></span>
    <button type="button" class="btn on" data-dir="col">读一列（阶段 2）</button>
    <button type="button" class="btn" data-dir="row">读一行（阶段 1）</button>
    <span style="width:4px"></span>
    <button type="button" class="btn" data-act="reset" title="重置">⏮</button>
    <button type="button" class="btn" data-act="play" title="播放 / 暂停">▶ 播放</button>
    <button type="button" class="btn" data-act="next" title="下一个周期">▶|</button>
  </div>
  <div class="anim-body">
    <div class="anim-stage">
      <div class="anim-panel"><div class="cap" data-g="cap">tile 的每一格：所在的 bank</div><div class="anim-grid" data-g="tile"></div></div>
      <div>
        <div class="anim-panel"><div class="cap">8 个 lane 的请求</div><div class="anim-lanes" data-g="lanes"></div></div>
        <div class="anim-panel" style="margin-top:10px"><div class="cap">8 个 bank 的队列（每周期每个 bank 只能服务 1 个请求）</div><div class="anim-banks" data-g="banks"></div></div>
      </div>
    </div>
    <div class="anim-status" data-g="status"></div>
    <div class="anim-note">格子里的数字是 bank 号 = 地址 % 8，颜色也按 bank 着色。tile[8][9] 的最后一列是 padding，永远不存数据，只是把每一行往后错开一个 bank。</div>
  </div>
</div>
<script>
(function () {
  var root = document.getElementById("an-bank"), B = 8, L = 8;
  function q(s) { return root.querySelector('[data-g="' + s + '"]'); }
  var pad = 0, dir = "col", k = 0, cyc = 0, timer = null;
  var laneEls = [], bankEls = [];
  for (var l = 0; l < L; l++) { var d = document.createElement("div"); d.className = "anim-lane"; q("lanes").appendChild(d); laneEls.push(d); }
  for (var b = 0; b < B; b++) { var e = document.createElement("div"); e.className = "anim-bank"; q("banks").appendChild(e); bankEls.push(e); }
  function plan() {
    var cols = B + pad, req = [];
    for (var l = 0; l < L; l++) { var i = dir === "col" ? l : k, j = dir === "col" ? k : l, a = i * cols + j; req.push({ i: i, j: j, addr: a, bank: a % B }); }
    var groups = {}; req.forEach(function (r, l) { (groups[r.bank] = groups[r.bank] || []).push(l); });
    req.forEach(function (r, l) { r.slot = groups[r.bank].indexOf(l) + 1; });   // 这个请求在第几个周期被服务
    var cycles = Math.max.apply(null, req.map(function (r) { return r.slot; }));
    return { cols: cols, req: req, cycles: cycles, groups: groups };
  }
  function render(animate) {
    var P = plan(), g = q("tile"); g.innerHTML = ""; g.style.gridTemplateColumns = "repeat(" + P.cols + ", 26px)";
    var hit = {}; P.req.forEach(function (r, l) { hit[r.i + "," + r.j] = l; });
    for (var i = 0; i < 8; i++) for (var j = 0; j < P.cols; j++) {
      var c = document.createElement("div"), a = i * P.cols + j, bk = a % B;
      if (j === 8) { c.className = "an-cell pad"; c.textContent = "pad"; }
      else { c.className = "an-cell fill"; c.style.setProperty("--h", bk * 45); c.textContent = bk; }
      var l = hit[i + "," + j];
      if (l !== undefined) { var r = P.req[l]; if (r.slot === cyc) c.classList.add("act"); else if (r.slot > cyc) c.classList.add("src"); if (r.slot === cyc && animate) c.classList.add("pop"); }
      g.appendChild(c);
    }
    P.req.forEach(function (r, l) {
      var el = laneEls[l]; el.innerHTML = "lane " + l + "<br>bank " + r.bank;
      el.className = "anim-lane " + (cyc === 0 ? "wait" : r.slot < cyc ? "done" : r.slot === cyc ? "now" : "wait");
    });
    for (var b = 0; b < B; b++) {
      var lanes = P.groups[b] || [], left = lanes.filter(function (l) { return P.req[l].slot > cyc; }).length;
      var el = bankEls[b]; el.innerHTML = "";
      var lab = document.createElement("span"); lab.textContent = "bank " + b; el.appendChild(lab);
      for (var n = 0; n < left; n++) { var s = document.createElement("span"); s.className = "q"; el.appendChild(s); }
      el.classList.toggle("busy", lanes.some(function (l) { return P.req[l].slot === cyc; }));
    }
    var what = dir === "col" ? "8 个 lane 读第 " + k + " 列 tile[0..7][" + k + "]" : "8 个 lane 读第 " + k + " 行 tile[" + k + "][0..7]";
    var msg;
    if (cyc === 0) msg = what + "。地址 = 行 × " + P.cols + " + 列，bank = 地址 % 8。按 ▶ 看硬件怎么一个周期一个周期地服务这些请求。";
    else {
      var now = P.req.map(function (r, l) { return r.slot === cyc ? l : -1; }).filter(function (l) { return l >= 0; });
      msg = "第 " + cyc + " 个周期：服务 lane " + now.join("、") + "。";
      if (cyc === P.cycles) msg += P.cycles === 1 ? "　<b>1 个周期全部完成：无冲突。</b>" : "　<b>一共 " + P.cycles + " 个周期：" + P.cycles + " 路 bank conflict，读一次变成读 " + P.cycles + " 次。</b>";
    }
    q("status").innerHTML = msg;
    root._max = P.cycles;
  }
  function go(c, animate) { cyc = Math.max(0, Math.min(root._max || 8, c)); render(animate); }
  var playBtn = root.querySelector('[data-act="play"]');
  function stop() { if (timer) { clearInterval(timer); timer = null; } playBtn.textContent = "▶ 播放"; }
  playBtn.onclick = function () {
    if (timer) return stop();
    if (cyc >= root._max) go(0, false);
    playBtn.textContent = "⏸ 暂停";
    timer = setInterval(function () { if (cyc >= root._max) return stop(); go(cyc + 1, true); }, 700);
  };
  root.querySelector('[data-act="next"]').onclick = function () { stop(); go(cyc + 1, true); };
  root.querySelector('[data-act="reset"]').onclick = function () { stop(); go(0, false); };
  function toggle(attr, set) {
    root.querySelectorAll("[" + attr + "]").forEach(function (b) {
      b.onclick = function () { stop(); set(b.getAttribute(attr));
        root.querySelectorAll("[" + attr + "]").forEach(function (x) { x.classList.toggle("on", x === b); }); go(0, false); };
    });
  }
  toggle("data-pad", function (v) { pad = +v; });
  toggle("data-dir", function (v) { dir = v; });
  render(false);
})();
</script>
:::

试一试这四种组合：**tile[8][8] + 读一列**要 8 个周期（全部挤在 bank 0），**tile[8][9] + 读一列**只要 1 个周期，而**读一行**不管有没有 padding 都是 1 个周期。真实硬件上把 8 换成 32：不加 padding 是 32 路冲突。


读第 0 列时，前 8 个线程各自落在哪个 bank：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 150px repeat(8, minmax(46px, 1fr)); min-width: 560px">
<div class="lbl">线程（tile 行 i）</div><div>0</div><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>7</div>
<div class="lbl">[32][32] 的 bank</div><div class="c1">0</div><div class="c1">0</div><div class="c1">0</div><div class="c1">0</div><div class="c1">0</div><div class="c1">0</div><div class="c1">0</div><div class="c1">0</div>
<div class="lbl">[32][33] 的 bank</div><div class="c2">0</div><div class="c2">1</div><div class="c2">2</div><div class="c2">3</div><div class="c2">4</div><div class="c2">5</div><div class="c2">6</div><div class="c2">7</div>
</div></div>
<p class="legend">多出来的那一列永远不存数据，只是把每一行往后错开一个 bank。代价：32 × 4 = 128 字节 shared memory。</p>
:::

阶段 1 是按行写 `tile[i][0..31]`，bank = `(33i + k) % 32`，k 连续，本来就无冲突，padding 不影响它。

### 另一种做法：swizzle（异或）

padding 浪费一点空间，也会让地址不再对齐到 2 的幂，对向量化读写和 Tensor Core 不友好。生产代码更常用 **swizzle**：不多加列，而是把列号和行号异或，`物理列 = j ^ (i % 32)`。同一列 j 的 32 行被异或到 32 个不同的物理列，同样无冲突。

- CuTe 的通用实现：[[cutlass:include/cute/swizzle.hpp@L55|struct Swizzle]]，核心一行是 [[cutlass:include/cute/swizzle.hpp@L78|offset ^ shiftr(offset & yyy_msk, msk_sft)]]
- FlashInfer 的版本：[[flashinfer:include/flashinfer/permuted_smem.cuh@L48-L53|get_permuted_offset：i * stride + (j ^ (i % 8))]]（以 16 字节为单位做异或）

C4 和 G2 会再遇到这个技巧。

## ⑤ 边界：M、N 不是 32 的倍数时

grid 是 `⌈N/32⌉ × ⌈M/32⌉`，最右边和最下边的 tile 只有一部分在矩阵里。两个阶段各自做越界检查：

- 阶段 1 `if (r < M && c < N)`：越界的位置不读，tile 里那几格留着垃圾值。
- 阶段 2 `if (yr < N && yc < M)`：越界的位置不写。y 里合法的位置只会读到 tile 里合法的格子，垃圾值永远不会被写出去。

注意阶段 2 的条件是 `yr < N && yc < M`，因为 y 的形状是 `[N, M]`。M 和 N 写反是这题最常见的 bug，而方阵测试发现不了，所以测试里一定要有 M ≠ N 的形状。

## ⑥ 常见坑与自测

- **忘了 `__syncthreads()`**：小矩阵上可能碰巧对，大矩阵上随机出错。
- **阶段 2 没有对调 blockIdx**：写到 y 的 (by, bx) 位置，方阵上部分 tile 对、部分错；非方阵直接越界。
- **阶段 2 用了 `tile[threadIdx.y + j][threadIdx.x]`**：这就成了原样拷贝，没有转置。
- **把 padding 加在第一维**（`tile[33][32]`）：没用，行长度还是 32。

::: answer 自测 1：tile 改成 `float tile[32][34]`，阶段 2 按列读还有冲突吗？
bank = (34i + k) % 32 = (2i + k) % 32。i 从 0 到 31 时 2i % 32 只取 16 个不同的值（0, 2, 4, …, 30），每个 bank 被两个线程访问：**2 路冲突**。padding 必须让行长度和 32 互素，33 是最省空间的选择。
:::

::: answer 自测 2：M = 100，N = 50，tiled 版本的 grid 是多少？最后一个 block 有多少个线程真正干活？
grid = (⌈50/32⌉, ⌈100/32⌉) = (2, 4)。最后一个 block（bx = 1, by = 3）覆盖 x 的第 96–127 行、32–63 列，合法的只有行 96–99、列 32–49，即 4 × 18 = 72 个元素。阶段 1 里只有 72 次读真正发生。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/c3_transpose.cu|solutions/c3_transpose.cu]] · [[here:mine/c3_transpose.cu|mine/c3_transpose.cu]] · [[here:tests/test_c3_transpose.py|tests/test_c3_transpose.py]] · [[here:content/c3.md|content/c3.md]]

**对照实现**

- cuda-samples 的整套演进：[[cudasamples:cpp/6_Performance/transpose/transpose.cu@L126|transposeNaive]] → [[cudasamples:cpp/6_Performance/transpose/transpose.cu@L141|transposeCoalesced]]（有 bank conflict）→ [[cudasamples:cpp/6_Performance/transpose/transpose.cu@L168|transposeNoBankConflicts]]（`TILE_DIM + 1`）→ [[cudasamples:cpp/6_Performance/transpose/transpose.cu@L205|transposeDiagonal]]（处理 partition camping）
- swizzle：[[cutlass:include/cute/swizzle.hpp@L55|CuTe Swizzle]] · [[flashinfer:include/flashinfer/permuted_smem.cuh@L48|FlashInfer permuted_smem]]

**文档**

- [NVIDIA blog · An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [Programming Guide · Matrix Transpose Example Using Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#matrix-transpose-example-using-shared-memory) · [Shared Memory Bank Conflicts](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-bank-conflicts) · [Coalesced Access](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access)
