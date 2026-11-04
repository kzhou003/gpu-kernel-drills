title: CuTe 专题：用“布局代数”管住 kernel 里所有的下标
lede: CuTe 是 CUTLASS 3 和 CuTe DSL 的核心：它把 GPU kernel 里五花八门的“谁拿哪个数、数放在哪”统一写成同一种对象 Layout，再用一小套代数运算把它们组合起来。这一页先讲它要解决的问题和思路，再用四个可以动手玩的演示把 Layout、composition、logical_divide 和线程划分一步步拆开。演示背后跑的就是参考解 g2_cute_layout.py 的算法（移植到网页里，并和 Python 版逐个对拍过）。

::: callout info 怎么读这一页
这一页讲“为什么”和“怎么想”；[G2 讲解](g2_cute_layout.html) 讲参考解每个函数的细节，两页可以对照着看。所有演示都可以点：按钮切换例子，▶| 一步步走，鼠标悬停看细节。参考解：[[here:solutions/g2_cute_layout.py|g2_cute_layout.py]]。
:::

## ① 问题：kernel 里的“下标地狱”

翻一下你在这套题里写过的 kernel，真正难写、最容易错的不是乘法，而是**下标**：

| 题目 | 这一行在算什么 | 代码 |
|---|---|---|
| C3 转置 | 线程 → tile 里的位置，tile 在 shared memory 里的地址 | `tile[threadIdx.x][threadIdx.y + j]`，`tile[32][33]` |
| C4 GEMM | 协作加载时第 e 个元素在哪 | `m = e / BK, k = e % BK; As[k][m] = A[gm * K + gk]` |
| C4 GEMM | 每个线程负责哪个 4×4 小块 | `tx = tid % (BN/TN); ty = tid / (BN/TN); As[k][ty*TM + i]` |
| T4 Triton GEMM | program 负责 C 的哪个 tile | `pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)` |
| L2 paged decode | 第 p 页的 KV 在哪 | `phys * stride_cb + offs_t * stride_ct + kvh * stride_ch + offs_d` |
| G1 | 一个 lane 持有哪几个元素 | `rep*tile + warp*spt*tpw + lane*spt + s` |

每一行都是手写的整数算术。它们有三个共同的问题：

::: flow 手写下标的问题
cfg cw=240 nw=185
n a 0 0 "全局内存的布局" "行主序？列主序？stride？" k=todo
n b 0 1.2 "shared memory 的布局" "padding？swizzle？" k=todo
n c 0 2.4 "线程 → 数据" "每个线程拿哪几个？" k=todo
n d 0 3.6 "MMA 指令要求的寄存器排列" "每代硬件都不一样" k=todo w=1.05
n p1 1.25 1.2 "容易写错" "差一个 /、% 就错位" k=warn
n p2 1.25 2.4 "很难复用" "换个 tile 尺寸全部重写" k=warn
n p3 1.25 3.6 "很难组合" "四层映射要手动串起来" k=warn
e a -> p1
e b -> p1
e c -> p2
e d -> p3
cap: 一个高性能 GEMM 里，这四层映射要同时正确：数据在显存里怎么放、在 shared memory 里怎么放、每个线程搬哪些、每个线程的寄存器怎么喂给 Tensor Core。换一代 GPU（Ampere 的 mma.sync → Hopper 的 wgmma → Blackwell 的 tcgen05），最后两层就全变了。
:::

::: callout analogy 车同轨，书同文
战国时每个国家的车轴宽度、文字、度量衡都不一样，货物从一国运到另一国，每过一道边境都要换车、换秤、重新登记。手写下标就是这样：全局内存一套编号，shared memory 一套，线程一套，Tensor Core 又一套，每一层之间都靠人手写的公式“换车”。CuTe 做的事相当于统一度量衡：所有层都用同一种东西描述（Layout），层与层之间的转换也变成标准运算，不再是每次手写的一次性公式。
:::

## ② 思路：一切映射都是 Layout，再加一小套代数

CuTe 的核心观察是：**上面那些下标公式，本质上都是“坐标 → 编号”的函数**。不管是“矩阵的 (i, j) 在内存的第几个位置”，还是“第 t 个线程的第 v 个值对应矩阵的哪个位置”，都可以用同一种对象表示：

```text
Layout = (Shape, Stride)      一个函数：坐标 → 整数
         offset = Σ coord_i × stride_i
```

然后提供一小套运算，把这些函数**组合、切分、化简**：

::: flow CuTe 的层次
cfg cw=230 nw=180
n it 0 0 "IntTuple" "可以嵌套的整数元组" k=vio
n ly 1 0 "Layout" "Shape : Stride，坐标 → 偏移" k=acc
n al 2 0 "布局代数" "合成 · 补集 · 切分 · 化简" k=warn
n te 1 1.3 "Tensor" "指针 + Layout" k=acc
n op 2 1.3 "算法" "copy · gemm · partition" k=pass
n at 3.15 1.3 "Atom" "MMA / Copy 指令的 Layout" k=pass w=1.05
e it -> ly
e ly -> al
e ly -> te
e te -> op
e al -> op dash
e op -> at
cap: 越往右越高层，但每一层都建立在 Layout 上。参考解实现的就是最左边三层（IntTuple、Layout、布局代数），这也是理解整个 CuTe 的门槛。
:::

::: callout analogy 图形学里的变换矩阵
早期的图形程序里，平移、旋转、缩放、投影各写各的公式，组合起来很麻烦。后来大家发现它们都可以写成 4×4 矩阵，“先旋转再平移”就是两个矩阵相乘，求逆、复合都有现成的运算。CuTe 对下标做了同样的事：所有映射都是 Layout，“先切 tile 再分给线程”就是 Layout 的组合（composition / divide），你不再需要手推每一层的整数公式。
:::

## ③ Layout：从坐标到偏移的函数

`(4,2):(1,4)` 的意思是：shape 是 4 × 2，stride 是 (1, 4)。坐标 (i, j) 映射到偏移 `i·1 + j·4`。一个整数下标先按**列字典序**（第一个模式变化最快）拆成坐标，再算偏移。

在下面的演示里切换不同的 layout，用 ▶| 一步步看“下标 → 坐标 → 偏移”。也可以在输入框里自己写一个 layout（比如 `(3,4):(4,1)`）。

::: raw
<style>
:root { --an-l: 86%; --an-s: 62%; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --an-l: 30%; --an-s: 45%; } }
:root[data-theme="dark"] { --an-l: 30%; --an-s: 45%; }
.cw { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); box-shadow: var(--shadow); margin: 16px 0; overflow: hidden; }
.cw-head { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; padding: 7px 12px; background: var(--surface-2); border-bottom: 1px solid var(--border); font-size: 13.5px; }
.cw-head b { margin-right: auto; }
.cw-head .btn { font-size: 12px; padding: 1px 8px; }
.cw-head .btn.on { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.cw-bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding: 8px 12px 0; font-size: 12.5px; }
.cw-bar .btn { font: 12px var(--mono); padding: 1px 8px; }
.cw-bar .btn.on { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.cw-bar input { font: 13px var(--mono); padding: 2px 6px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); width: 190px; }
.cw-body { padding: 10px 12px 12px; }
.cw-row { display: flex; flex-wrap: wrap; gap: 18px; justify-content: center; align-items: flex-start; }
.cw-cap { font: 12px var(--mono); color: var(--muted); margin: 0 0 4px; text-align: center; }
.cw-grid { display: grid; gap: 2px; justify-content: center; }
.cw-c { min-width: 34px; height: 30px; padding: 0 3px; border: 1px solid var(--border); border-radius: 5px; background: var(--surface-2); font: 11.5px/28px var(--mono); text-align: center; color: var(--text); transition: box-shadow .15s, transform .15s; cursor: default; }
.cw-c.f { background: hsl(var(--h) var(--an-s) var(--an-l)); }
.cw-c.hole { background: none; border-style: dashed; color: var(--muted); }
.cw-c.dup { box-shadow: inset 0 0 0 2px var(--f-todo); }
.cw-c.hl { box-shadow: 0 0 0 2px var(--accent); transform: scale(1.08); position: relative; z-index: 1; }
.cw-c.hl2 { box-shadow: 0 0 0 2px var(--f-warn); }
.cw-lab { font: 11px/30px var(--mono); color: var(--muted); text-align: right; padding-right: 4px; }
.cw-status { margin-top: 10px; padding: 7px 10px; border-radius: 8px; background: var(--surface-2); font-size: 13.5px; min-height: 2.4em; }
.cw-status code { font-size: 12.5px; }
.cw-info { font-size: 12.5px; color: var(--muted); margin-top: 6px; }
.cw-err { color: var(--bad); font-size: 12.5px; }
.cw-chain { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 4px; margin-top: 8px; }
</style>
<script>
// CuTe layout algebra: a line-by-line port of solutions/g2_cute_layout.py (same algorithms, same results).
var Cute = (function () {
  function isT(x) { return Array.isArray(x); }
  function flatten(t) { return isT(t) ? t.reduce(function (a, e) { return a.concat(flatten(e)); }, []) : [t]; }
  function product(s) { return flatten(s).reduce(function (a, b) { return a * b; }, 1); }
  function idx2crd(idx, shape) {               // colexicographic: first mode fastest, follows nesting
    if (isT(shape)) { var crd = []; shape.forEach(function (s) { var n = product(s); crd.push(idx2crd(idx % n, s)); idx = Math.floor(idx / n); }); return crd; }
    return idx;
  }
  function inner(crd, shape, stride) {
    if (isT(shape)) {
      if (!isT(crd)) crd = idx2crd(crd, shape);
      var s = 0; for (var i = 0; i < shape.length; i++) s += inner(crd[i], shape[i], stride[i]); return s;
    }
    return crd * stride;
  }
  function L(shape, stride) { return { shape: shape, stride: stride }; }
  function call(l, crd) { return inner(crd, l.shape, l.stride); }
  function size(l) { return product(l.shape); }
  function cosize(l) { return size(l) ? call(l, size(l) - 1) + 1 : 0; }
  function unflatten(shape, it) { return isT(shape) ? shape.map(function (s) { return unflatten(s, it); }) : it.shift(); }
  function layoutLeft(shape) { var st = [], acc = 1; flatten(shape).forEach(function (s) { st.push(acc); acc *= s; }); return L(shape, unflatten(shape, st)); }
  function layoutRight(shape) { var fl = flatten(shape), st = new Array(fl.length), acc = 1;
    for (var i = fl.length - 1; i >= 0; i--) { st[i] = acc; acc *= fl[i]; } return L(shape, unflatten(shape, st)); }
  function coalesce(l) {
    var fs = flatten(l.shape), fd = flatten(l.stride), sh = [], st = [];
    for (var i = 0; i < fs.length; i++) {
      var s = fs[i], d = fd[i];
      if (s === 1) continue;
      if (sh.length && st[st.length - 1] * sh[sh.length - 1] === d) sh[sh.length - 1] *= s;
      else { sh.push(s); st.push(d); }
    }
    if (!sh.length) return L(1, 0);
    if (sh.length === 1) return L(sh[0], st[0]);
    return L(sh, st);
  }
  function composition(a, b) {
    if (isT(b.shape)) {
      var modes = b.shape.map(function (s, i) { return composition(a, L(s, b.stride[i])); });
      return L(modes.map(function (m) { return m.shape; }), modes.map(function (m) { return m.stride; }));
    }
    if (b.stride === 0) return L(b.shape, 0);
    var fa = coalesce(a), as = flatten(fa.shape), ad = flatten(fa.stride);
    var rs = b.shape, rd = b.stride, sh = [], st = [];
    for (var i = 0; i < as.length - 1; i++) {
      var s = as[i], d = ad[i];
      if (!(s % rd === 0 || rd % s === 0)) throw new Error("stride divisibility");
      var ns = Math.min(Math.max(1, Math.floor(s / rd)), rs);
      if (ns !== 1) { sh.push(ns); st.push(rd * d); }
      if (rs % ns !== 0) throw new Error("shape divisibility");
      rs = rs / ns; rd = Math.ceil(rd / s);
    }
    if (rs !== 1 || !sh.length) { sh.push(rs); st.push(rd * ad[ad.length - 1]); }
    return sh.length === 1 ? L(sh[0], st[0]) : L(sh, st);
  }
  function complement(a, cotarget) {
    var fs = flatten(a.shape), fd = flatten(a.stride), modes = [];
    for (var i = 0; i < fs.length; i++) if (fs[i] !== 1 && fd[i] !== 0) modes.push([fd[i], fs[i]]);
    modes.sort(function (x, y) { return x[0] - y[0] || x[1] - y[1]; });
    var sh = [], st = [], cur = 1;
    modes.forEach(function (m) { var d = m[0], s = m[1];
      if (d % cur !== 0) throw new Error("A must be injective with divisible strides");
      sh.push(d / cur); st.push(cur); cur = s * d; });
    sh.push(Math.ceil(cotarget / cur)); st.push(cur);
    return coalesce(L(sh, st));
  }
  function logicalDivide(a, tiler) {
    var c = complement(tiler, size(a));
    return composition(a, L([tiler.shape, c.shape], [tiler.stride, c.stride]));
  }
  function fmt(x) { return isT(x) ? "(" + x.map(fmt).join(",") + ")" : String(x); }
  function show(l) { return fmt(l.shape) + ":" + fmt(l.stride); }
  function parse(str) {                         // "((2,2),2):((1,4),2)" -> layout
    var parts = str.replace(/\s+/g, "").split(":");
    if (parts.length !== 2) throw new Error("写成 shape:stride，比如 (4,2):(1,4)");
    function tup(s) { var pos = 0;
      function val() { if (s[pos] === "(") { pos++; var out = [];
          while (s[pos] !== ")") { out.push(val()); if (s[pos] === ",") pos++; else if (s[pos] !== ")") throw new Error("括号或逗号不匹配"); }
          pos++; return out; }
        var m = /^\d+/.exec(s.slice(pos)); if (!m) throw new Error("应该是数字：" + s.slice(pos)); pos += m[0].length; return +m[0]; }
      var v = val(); if (pos !== s.length) throw new Error("多余的字符：" + s.slice(pos)); return v; }
    var sh = tup(parts[0]), st = tup(parts[1]);
    if (JSON.stringify(sh).replace(/\d+/g, "0") !== JSON.stringify(st).replace(/\d+/g, "0")) throw new Error("shape 和 stride 的嵌套结构要一样");
    return L(sh, st);
  }
  return { L: L, call: call, size: size, cosize: cosize, idx2crd: idx2crd, flatten: flatten, product: product,
           layoutLeft: layoutLeft, layoutRight: layoutRight, coalesce: coalesce, composition: composition,
           complement: complement, logicalDivide: logicalDivide, show: show, fmt: fmt, parse: parse };
})();
if (typeof module !== "undefined") module.exports = Cute;
window.CuteUI = {
  hue: function (i, n) { return Math.round((i / Math.max(n, 1)) * 300); },
  cell: function (txt, cls, hue, title) { var d = document.createElement("div"); d.className = "cw-c" + (cls ? " " + cls : "");
    if (hue !== undefined && hue !== null) d.style.setProperty("--h", hue); d.textContent = txt; if (title) d.title = title; return d; },
  lab: function (txt) { var d = document.createElement("div"); d.className = "cw-lab"; d.textContent = txt; return d; },
  player: function (root, getMax, render, ms) {
    var step = 0, timer = null, play = root.querySelector('[data-act="play"]');
    function stop() { if (timer) { clearInterval(timer); timer = null; } if (play) play.textContent = "▶ 播放"; }
    function go(s) { step = Math.max(0, Math.min(getMax(), s)); render(step); }
    root.querySelectorAll("[data-act]").forEach(function (b) { b.onclick = function () {
      var a = b.dataset.act;
      if (a === "play") { if (timer) return stop(); if (step >= getMax()) go(0); play.textContent = "⏸ 暂停";
        timer = setInterval(function () { if (step >= getMax()) return stop(); go(step + 1); }, ms || 900); }
      else { stop(); go(a === "next" ? step + 1 : a === "prev" ? step - 1 : 0); } }; });
    return { go: go, stop: stop, get step() { return step; } };
  }
};
</script>
<div class="cw" id="cw-layout">
  <div class="cw-head"><b>演示 1：Layout 浏览器</b>
    <button type="button" class="btn" data-act="reset">⏮</button><button type="button" class="btn" data-act="prev">◀</button>
    <button type="button" class="btn" data-act="play">▶ 播放</button><button type="button" class="btn" data-act="next">▶|</button></div>
  <div class="cw-bar" data-g="presets"></div>
  <div class="cw-bar"><span>自己写：</span><input data-g="inp" value="(4,2):(1,4)" spellcheck="false"><button type="button" class="btn" data-g="apply">应用</button><span class="cw-err" data-g="err"></span></div>
  <div class="cw-body">
    <div class="cw-row">
      <div><div class="cw-cap" data-g="gcap"></div><div class="cw-grid" data-g="grid"></div></div>
      <div><div class="cw-cap">内存（偏移 0 … cosize−1）里放的是哪个 (行, 列)</div><div class="cw-grid" data-g="mem"></div></div>
    </div>
    <div class="cw-status" data-g="status"></div>
    <div class="cw-info" data-g="info"></div>
  </div>
</div>
<script>
(function () {
  var C = Cute, U = CuteUI, root = document.getElementById("cw-layout");
  function q(s) { return root.querySelector('[data-g="' + s + '"]'); }
  var PRESETS = [
    ["(4,2):(1,4)", "4×2 列主序"], ["(4,2):(2,1)", "4×2 行主序"], ["(4,2):(1,5)", "列主序 + padding"],
    ["((2,2),(2,2)):((1,4),(2,8))", "4×4 分块"], ["(4,2):(0,1)", "广播（stride 0）"], ["8:2", "一维、跨步"]];
  var L = null, ok = true;
  PRESETS.forEach(function (p, k) { var b = document.createElement("button"); b.type = "button"; b.className = "btn"; b.textContent = p[0]; b.title = p[1];
    b.onclick = function () { q("inp").value = p[0]; apply(); }; q("presets").appendChild(b); });
  function modes(l) {                              // top-level modes as [shape, stride] pairs (rank 1 -> one mode)
    return Array.isArray(l.shape) ? l.shape.map(function (s, i) { return [s, l.stride[i]]; }) : [[l.shape, l.stride]];
  }
  function crdStr(c) { return C.fmt(c); }
  function apply() {
    try { L = C.parse(q("inp").value); ok = true; q("err").textContent = ""; }
    catch (e) { q("err").textContent = e.message; ok = false; return; }
    if (C.size(L) > 64 || C.cosize(L) > 96) { q("err").textContent = "演示只画小于 64 个元素的 layout"; ok = false; return; }
    root.querySelectorAll('[data-g="presets"] .btn').forEach(function (b) { b.classList.toggle("on", b.textContent === C.show(L)); });
    player.go(0);
  }
  function render(step) {
    if (!ok || !L) return;
    var md = modes(L), rank2 = md.length === 2, R = rank2 ? C.product(md[0][0]) : 1, Cn = rank2 ? C.product(md[1][0]) : C.size(L);
    var n = C.size(L), cos = C.cosize(L), offs = [], owners = {};
    for (var i = 0; i < n; i++) { var o = C.call(L, i); offs.push(o); (owners[o] = owners[o] || []).push(i); }
    var cur = step > 0 ? step - 1 : -1;
    // 坐标网格：(行坐标, 列坐标) -> 偏移
    var g = q("grid"); g.innerHTML = ""; g.style.gridTemplateColumns = "28px repeat(" + Cn + ", minmax(34px, auto))";
    q("gcap").textContent = rank2 ? "坐标网格：格子里是偏移（行 = 第 0 个模式，列 = 第 1 个模式）" : "一维：第 i 个元素的偏移";
    g.appendChild(U.lab(""));
    for (var c = 0; c < Cn; c++) { var h = U.lab(String(c)); h.style.textAlign = "center"; g.appendChild(h); }
    for (var r = 0; r < R; r++) {
      g.appendChild(U.lab(rank2 ? String(r) : "i"));
      for (var c = 0; c < Cn; c++) {
        var idx = rank2 ? r + c * R : c, o = offs[idx];
        var el = U.cell(String(o), "f", U.hue(o, cos), "下标 " + idx + " → 坐标 " + crdStr(C.idx2crd(idx, L.shape)) + " → 偏移 " + o);
        if (owners[o].length > 1) el.classList.add("dup");
        if (idx === cur) el.classList.add("hl");
        g.appendChild(el);
      }
    }
    // 内存条
    var m = q("mem"); m.innerHTML = ""; var per = Math.min(cos, 16); m.style.gridTemplateColumns = "repeat(" + per + ", minmax(34px, auto))";
    for (var a = 0; a < cos; a++) {
      var own = owners[a];
      var lc = function (i) { return rank2 ? "(" + (i % R) + "," + Math.floor(i / R) + ")" : String(i); };   // logical (row, col)
      var el = own ? U.cell(own.length > 1 ? own.length + "个" : lc(own[0]), "f" + (own.length > 1 ? " dup" : ""), U.hue(a, cos),
                            "偏移 " + a + (own.length === 1 ? "：坐标 " + crdStr(C.idx2crd(own[0], L.shape)) : ""))
                   : U.cell("·", "hole", null, "偏移 " + a + "：没有坐标映射到这里");
      if (cur >= 0 && offs[cur] === a) el.classList.add("hl");
      m.appendChild(el);
    }
    // 说明
    var msg;
    if (cur < 0) msg = "这个 layout 一共 <b>" + n + "</b> 个坐标（size），最大偏移是 " + (cos - 1) + "（cosize = " + cos + "）。按 ▶| 从下标 0 开始一步步看。";
    else {
      var crd = C.idx2crd(cur, L.shape), fs = C.flatten(crd), fd = C.flatten(L.stride);
      msg = "下标 <b>" + cur + "</b> → 按列字典序拆成坐标 <code>" + crdStr(crd) + "</code> → 偏移 = " +
        fs.map(function (x, k) { return x + "×" + fd[k]; }).join(" + ") + " = <b>" + offs[cur] + "</b>";
      if (owners[offs[cur]].length > 1) msg += "　（这个偏移被 " + owners[offs[cur]].length + " 个坐标共用：stride 为 0 的维度在“广播”）";
    }
    q("status").innerHTML = msg;
    var holes = cos - Object.keys(owners).length, dups = Object.keys(owners).filter(function (k) { return owners[k].length > 1; }).length;
    q("info").innerHTML = "layout <code>" + C.show(L) + "</code>　size = " + n + "　cosize = " + cos +
      (holes ? "　空位 " + holes + " 个（虚线格）" : "") + (dups ? "　被多个坐标共用的偏移 " + dups + " 个（红框）" : "") +
      "　coalesce 之后：<code>" + C.show(C.coalesce(L)) + "</code>";
  }
  var player = U.player(root, function () { return ok && L ? C.size(L) : 0; }, render, 700);
  q("apply").onclick = apply; q("inp").onkeydown = function (e) { if (e.key === "Enter") apply(); };
  apply();
})();
</script>
:::

几个值得点开看的例子：

- **列主序 vs 行主序**：同样是 4×2，只是 stride 不同，“第 5 个元素”在内存里的位置就不同。
- **列主序 + padding** `(4,2):(1,5)`：每列多空一格，内存条里出现虚线的空位。这就是 C3 的 `tile[32][33]`。
- **广播** `(4,2):(0,1)`：第一个模式的 stride 是 0，4 行全都映射到同一个位置。这就是 G1 里 SliceLayout 的广播，也是 CuTe 表达“同一个值给好几个线程”的方式。
- **分块** `((2,2),(2,2)):((1,4),(2,8))`：下一节专门讲。

::: callout analogy 储物柜
Layout 像商场储物柜的编号规则。你只知道“第 i 排第 j 列”（坐标），柜子上贴的却是一个号码（偏移）。stride 就是规则：往下一排号码加几，往右一列号码加几。同一排柜子，可以按“先往下数”编号（列主序），也可以按“先往右数”编号（行主序），还可以每列之间空一个柜子（padding），甚至让一整排共用一个柜子（广播）。柜子本身没动，变的只是编号规则。
:::

## ④ 为什么要嵌套：一个维度里藏着两层结构

CuTe 的 shape 可以嵌套，比如 `((2,2),(2,2))`。它表示一个逻辑上 4×4 的矩阵，但**每一维内部又分成了两层**：“块内第几个”和“第几个块”。

在演示 1 里点 `((2,2),(2,2)):((1,4),(2,8))`：

- 行坐标 r（0..3）被拆成 (r % 2, r / 2)，偏移贡献是 `(r%2)·1 + (r/2)·4`
- 列坐标 c（0..3）被拆成 (c % 2, c / 2)，偏移贡献是 `(c%2)·2 + (c/2)·8`

结果是：每个 2×2 小块在内存里是连续的 4 个数（0–3、4–7……），小块之间再按列主序排。网格里看起来还是普通的 4×4，内存条上每 4 个颜色相近的格子就是一个小块。

这正是 tiling 的表达方式：**逻辑坐标不变（还是 4×4），物理排列按块组织**。kernel 代码照样用 (r, c) 访问，块怎么排全藏在 layout 里。换一种分块方式，只需要换 layout，访问代码一行都不用改。

::: callout analogy 楼—单元—层—户
“5 栋 2 单元 3 楼 1 号”是一个四层嵌套的地址，但你也可以只说“第 37 户”。CuTe 的嵌套 shape 就是这样：一个逻辑维度（第几户）里藏着好几层物理结构（楼、单元、层）。代码可以只用“第几户”（扁平坐标），也可以用完整的四层地址（层次坐标），两者通过列字典序自动互相转换。
:::

## ⑤ 布局代数：每个运算解决一个具体问题

| 运算 | 要解决的问题 | 一句话 |
|---|---|---|
| `coalesce` | 层次很深的 layout 不好读、也不好优化 | 同一个函数，化简成最少的模式 |
| `composition` A∘B | “先按 B 选元素，再按 A 找地址” | 用一个 layout 去选另一个 layout 的元素 |
| `complement` | 选了一部分，剩下的怎么描述 | 找出 A 在 [0, M) 里没覆盖到的“空隙” |
| `logical_divide` | 怎么把一个大张量切成 tile | tile 的形状由 B 给，tile 怎么排由 complement 给 |

### composition：先选，再查地址

`R = A ∘ B` 的意思是 `R(i) = A(B(i))`：B 决定“要 A 的第几个元素”，A 决定“这个元素在内存哪里”。演示里左边是 B 选出的下标，右边是它们在 A 的内存里的位置。

::: raw
<div class="cw" id="cw-comp">
  <div class="cw-head"><b>演示 2：composition，R(i) = A(B(i))</b>
    <button type="button" class="btn" data-act="reset">⏮</button><button type="button" class="btn" data-act="prev">◀</button>
    <button type="button" class="btn" data-act="play">▶ 播放</button><button type="button" class="btn" data-act="next">▶|</button></div>
  <div class="cw-bar"><span>A（数据在内存里怎么放）：</span><span data-g="pa"></span></div>
  <div class="cw-bar"><span>B（要 A 的哪些元素）：</span><span data-g="pb"></span></div>
  <div class="cw-body">
    <div class="cw-row">
      <div><div class="cw-cap">i → B(i)（B 选出 A 的第几个）</div><div class="cw-grid" data-g="bsel"></div></div>
      <div><div class="cw-cap">A 的坐标网格（格子里是偏移），被选中的描边</div><div class="cw-grid" data-g="agrid"></div></div>
    </div>
    <div class="cw-status" data-g="status"></div>
    <div class="cw-info" data-g="info"></div>
  </div>
</div>
<script>
(function () {
  var C = Cute, U = CuteUI, root = document.getElementById("cw-comp");
  function q(s) { return root.querySelector('[data-g="' + s + '"]'); }
  var AS = [["(4,4):(4,1)", "4×4 行主序"], ["(4,4):(1,4)", "4×4 列主序"], ["(4,2):(2,1)", "4×2 行主序"]];
  var BS = [["4:2", "隔一个取一个"], ["(4,4):(4,1)", "按“行主序”读"], ["(2,2):(1,4)", "左上 2×2"], ["4:1", "前 4 个"]];
  var A = C.parse(AS[0][0]), B = C.parse(BS[1][0]), R = null, err = "";
  function mk(box, list, set) { list.forEach(function (p) { var b = document.createElement("button"); b.type = "button"; b.className = "btn"; b.textContent = p[0]; b.title = p[1];
    b.onclick = function () { set(C.parse(p[0])); box.querySelectorAll(".btn").forEach(function (x) { x.classList.toggle("on", x === b); }); recompute(); }; box.appendChild(b); });
    box.querySelector(".btn").classList.add("on"); }
  mk(q("pa"), AS, function (l) { A = l; }); mk(q("pb"), BS, function (l) { B = l; });
  q("pb").querySelectorAll(".btn").forEach(function (b, k) { b.classList.toggle("on", k === 1); });
  function recompute() { try { R = C.composition(A, B); err = ""; } catch (e) { R = null; err = e.message; }
    if (R === null || C.size(B) > C.size(A)) { err = err || "B 选的下标超出了 A 的大小"; } player.go(0); }
  function render(step) {
    var nA = C.size(A), nB = C.size(B), cur = step - 1;
    var md = Array.isArray(A.shape) ? A.shape : [A.shape], Rr = C.product(md[0]), Cc = md.length > 1 ? C.product(md[1]) : 1;
    var sel = {}; for (var i = 0; i < nB; i++) sel[C.call(B, i)] = i;
    var g = q("agrid"); g.innerHTML = ""; g.style.gridTemplateColumns = "repeat(" + Cc + ", 34px)";
    for (var r = 0; r < Rr; r++) for (var c = 0; c < Cc; c++) {
      var idx = r + c * Rr, o = C.call(A, idx), el = U.cell(String(o), sel[idx] !== undefined ? "f" : "", sel[idx] !== undefined ? U.hue(sel[idx], nB) : null,
        "A 的下标 " + idx + "，坐标 (" + r + "," + c + ")，偏移 " + o);
      if (cur >= 0 && C.call(B, cur) === idx) el.classList.add("hl");
      g.appendChild(el);
    }
    var bs = q("bsel"); bs.innerHTML = ""; bs.style.gridTemplateColumns = "50px 34px 34px 34px";
    ["i", "B(i)", "→", "R(i)"].forEach(function (t) { var h = U.lab(t); h.style.textAlign = "center"; bs.appendChild(h); });
    for (var i = 0; i < nB; i++) {
      var b = C.call(B, i), a = C.call(A, b), ri = R ? C.call(R, i) : "?";
      bs.appendChild(U.lab(String(i)));
      var e1 = U.cell(String(b), "f", U.hue(i, nB)); var e2 = U.lab("→"); e2.style.textAlign = "center";
      var e3 = U.cell(String(ri), "f" + (ri !== a ? " dup" : ""), U.hue(i, nB));
      if (i === cur) { e1.classList.add("hl"); e3.classList.add("hl"); }
      bs.appendChild(e1); bs.appendChild(e2); bs.appendChild(e3);
      if (i >= 15) break;
    }
    var msg;
    if (err) msg = "<span class='cw-err'>" + err + "</span>";
    else if (cur < 0) msg = "A = <code>" + C.show(A) + "</code>，B = <code>" + C.show(B) + "</code>。CuTe 不需要逐个元素去算：它直接算出合成后的 layout <b><code>" + C.show(R) + "</code></b>。按 ▶| 逐个验证 R(i) = A(B(i))。";
    else { var b = C.call(B, cur); msg = "i = " + cur + "：B(" + cur + ") = <b>" + b + "</b>，也就是 A 的第 " + b + " 个元素，坐标 <code>" + C.fmt(C.idx2crd(b, A.shape)) + "</code>，偏移 A(" + b + ") = <b>" + C.call(A, b) + "</b>。合成的 R = <code>" + C.show(R) + "</code> 直接给出 R(" + cur + ") = " + C.call(R, cur) + "。"; }
    q("status").innerHTML = msg;
    q("info").innerHTML = R ? "R = A ∘ B = <code>" + C.show(R) + "</code>　（右列红框表示 R(i) ≠ A(B(i))，正常情况下不会出现）" : "";
  }
  var player = U.player(root, function () { return R ? C.size(B) : 0; }, render, 700);
  recompute();
})();
</script>
:::

值得试的组合：

- A = 4×4 行主序，B = `(4,4):(4,1)`：用“行主序的读法”去读一个行主序矩阵，结果 R = `(4,4):(1,4)`，相当于**转置**。C3 里你手写的坐标对调，在 CuTe 里就是一次 composition。
- A = 4×2 行主序，B = `4:2`：B 的步子跨过了 A 的第一维边界，结果被拆成两个模式 `(2,2):(4,1)`（G2 讲解第 ④ 节的例子）。
- B = `(2,2):(1,4)`：从 4×4 里选出左上角 2×2 的子块。**取 tile 本质上就是 composition**。

::: callout analogy 两次换乘
composition 像坐地铁换乘：B 是第一段线路，把你从“起点站 i”送到“换乘站 B(i)”；A 是第二段线路，把你从换乘站送到“终点 A(B(i))”。CuTe 的本事是直接给你一张合并后的直达线路图 R，以后不用每次都换乘。
:::

### complement 和 logical_divide：把一个张量切成 tile

切 tile 需要回答两个问题：**一个 tile 长什么样**（tile 内部的元素怎么选），以及 **tile 之间怎么排**。第一个由你给的 tiler B 决定；第二个就是 B 的 complement：B 没覆盖到的“空隙”，正好是 tile 的重复方式。

```text
logical_divide(A, B) = A ∘ ( B , complement(B, size(A)) )
                             └ tile 内 ┘ └──── tile 之间 ────┘
```

::: raw
<div class="cw" id="cw-div">
  <div class="cw-head"><b>演示 3：logical_divide，把 A 切成 tile</b>
    <button type="button" class="btn" data-act="reset">⏮</button><button type="button" class="btn" data-act="prev">◀</button>
    <button type="button" class="btn" data-act="play">▶ 播放</button><button type="button" class="btn" data-act="next">▶|</button></div>
  <div class="cw-bar"><span>A：</span><span data-g="pa"></span></div>
  <div class="cw-bar"><span>tiler B：</span><span data-g="pb"></span></div>
  <div class="cw-body">
    <div class="cw-row">
      <div><div class="cw-cap">结果：每一行是一个 tile（模式 1），每一列是 tile 内第几个（模式 0），格子里是偏移</div><div class="cw-grid" data-g="tiles"></div></div>
      <div><div class="cw-cap">A 的内存：颜色 = 属于第几个 tile</div><div class="cw-grid" data-g="mem"></div></div>
    </div>
    <div class="cw-status" data-g="status"></div>
    <div class="cw-info" data-g="info"></div>
  </div>
</div>
<script>
(function () {
  var C = Cute, U = CuteUI, root = document.getElementById("cw-div");
  function q(s) { return root.querySelector('[data-g="' + s + '"]'); }
  var AS = [["16:1", "16 个连续元素"], ["32:1", "32 个连续元素"], ["16:2", "每隔一个存一个"]];
  var BS = [["4:1", "连续 4 个"], ["4:2", "隔一个取一个"], ["(2,2):(1,8)", "两段各 2 个"], ["8:1", "连续 8 个"]];
  var A = C.parse("16:1"), B = C.parse("4:1"), D = null, comp = null, err = "";
  function mk(box, list, set, on) { list.forEach(function (p, k) { var b = document.createElement("button"); b.type = "button"; b.className = "btn" + (k === on ? " on" : ""); b.textContent = p[0]; b.title = p[1];
    b.onclick = function () { set(C.parse(p[0])); box.querySelectorAll(".btn").forEach(function (x) { x.classList.toggle("on", x === b); }); recompute(); }; box.appendChild(b); }); }
  mk(q("pa"), AS, function (l) { A = l; }, 0); mk(q("pb"), BS, function (l) { B = l; }, 0);
  function recompute() { try { comp = C.complement(B, C.size(A)); D = C.logicalDivide(A, B); err = ""; } catch (e) { D = null; err = e.message; } player.go(0); }
  // the 2-mode result: mode 0 = tile, mode 1 = repetition
  function render(step) {
    if (!D) { q("status").innerHTML = "<span class='cw-err'>" + err + "</span>"; return; }
    var T = C.size(B), N = C.size(A) / T, cur = step - 1;
    var tileOf = {}, g = q("tiles"); g.innerHTML = ""; g.style.gridTemplateColumns = "54px repeat(" + T + ", 34px)";
    g.appendChild(U.lab("")); for (var e = 0; e < T; e++) { var h = U.lab(String(e)); h.style.textAlign = "center"; g.appendChild(h); }
    for (var t = 0; t < N; t++) {
      g.appendChild(U.lab("tile " + t));
      for (var e = 0; e < T; e++) { var o = C.call(D, [e, t]); tileOf[o] = t;
        var el = U.cell(String(o), "f", U.hue(t, N), "tile " + t + " 的第 " + e + " 个：偏移 " + o);
        if (t === cur) el.classList.add("hl"); g.appendChild(el); }
    }
    var cos = C.cosize(A), m = q("mem"); m.innerHTML = ""; m.style.gridTemplateColumns = "repeat(" + Math.min(cos, 16) + ", 34px)";
    for (var a = 0; a < cos; a++) { var t = tileOf[a];
      var el = t === undefined ? U.cell("·", "hole", null, "偏移 " + a + "：A 不使用") : U.cell(String(a), "f", U.hue(t, N), "偏移 " + a + " 属于 tile " + t);
      if (t !== undefined && t === cur) el.classList.add("hl"); m.appendChild(el); }
    var msg = cur < 0 ? "tiler B = <code>" + C.show(B) + "</code> 规定一个 tile 选哪几个；它在 [0, " + C.size(A) + ") 里的空隙是 complement = <code>" + C.show(comp) + "</code>，规定 tile 之间怎么排。两者拼起来再和 A 合成，就是结果。按 ▶| 一个 tile 一个 tile 地看。"
      : "tile " + cur + "：起点由 complement 决定（第 " + cur + " 个 = " + C.call(comp, cur) + "），tile 内的元素由 B 决定（" + Array.from({length: T}, function (_, e) { return C.call(B, e); }).join("、") + "）。经过 A 映射到偏移 " + Array.from({length: T}, function (_, e) { return C.call(D, [e, cur]); }).join("、") + "。";
    q("status").innerHTML = msg;
    q("info").innerHTML = "complement(B, " + C.size(A) + ") = <code>" + C.show(comp) + "</code>　logical_divide(A, B) = <code>" + C.show(D) + "</code>";
  }
  var player = U.player(root, function () { return D ? C.size(A) / C.size(B) : 0; }, render, 900);
  recompute();
})();
</script>
:::

- tiler `4:1`：最普通的切法，4 个连续的一块，complement 是 `4:4`（每块起点相差 4）。
- tiler `4:2`：每个 tile 是“隔一个取一个”的 4 个元素，complement 是 `(2,2):(1,8)`，所以 tile 0 和 tile 1 交错，tile 2、3 在后半段。
- A = `16:2`：A 本身是跨步存放的，切出来的偏移也跟着翻倍，tile 的结构不变。这就是“tile 的形状”和“数据怎么存”被分开描述的好处。

二维的情况（GEMM 里把 M×N 的矩阵切成 BM×BN 的 tile）是**对每一维分别做一次 divide**，CuTe 叫它 `zipped_divide` / `local_tile`。

::: callout analogy 切蛋糕
tiler 是你手里的模具，决定每一块蛋糕的形状（4 个连续的，或者隔一个取一个的 4 个）；complement 是“模具一次次往哪里放”，决定这些块在整个蛋糕上怎么排开，保证不重叠、不遗漏。logical_divide 就是“用这个模具把整个蛋糕切完”，结果是一张“第几块的第几口在哪”的表。蛋糕本身放在什么盘子里（A 的 stride），模具不用关心。
:::

## ⑥ 用起来：Tensor、local_tile 和线程划分

有了 Layout，CuTe 的 **Tensor = 指针 + Layout**。在 Tensor 上做 divide，得到的还是 Tensor，所以“切 tile”和“分给线程”都变成了对 Tensor 的操作。CUTLASS 的教程 [[cutlass:examples/cute/tutorial/sgemm_1.cu@L52|sgemm_1.cu]] 就是 C4 那个 GEMM 的 CuTe 版本：

::: flow 同一个 GEMM：C4 手写的下标 vs CuTe
cfg cw=270 nw=210
n c1 0 0 "C4：A[gm * K + gk]" "手写的全局地址" k=todo
n c2 0 1.2 "C4：blockIdx.y * BM" "手写 tile 的起点" k=todo
n c3 0 2.4 "C4：e / BK, e % BK" "手写的协作加载" k=todo
n c4 0 3.6 "C4：tid % 16, tid / 16" "手写线程到 4×4 小块" k=todo
n k1 1 0 "make_tensor(ptr, shape, stride)" "Tensor mA" k=pass w=1.05
n k2 1 1.2 "local_tile(mA, tiler, coord)" "Tensor gA：本 block 的 tile" k=pass w=1.05
n k3 1 2.4 "local_partition(gA, tA, tid)" "Tensor tAgA：本线程搬的部分" k=pass w=1.05
n k4 1 3.6 "local_partition(sA, tC, tid, …)" "本线程算的部分" k=pass w=1.05
e c1 -> k1
e c2 -> k2
e c3 -> k3
e c4 -> k4
cap: 左边每一行都是一条手写的整数公式；右边每一行都是一次 layout 运算。CuTe 版本里没有任何 / 和 %：tile 尺寸和线程排布都写在 layout 里（host 端几行 make_layout），换配置只改 layout，kernel 主体不变。
:::

对应的源码行：

- 构造全局张量：[[cutlass:examples/cute/tutorial/sgemm_1.cu@L98-L100|make_tensor(make_gmem_ptr(A), shape, stride)]]
- 取本 block 的 tile：[[cutlass:examples/cute/tutorial/sgemm_1.cu@L104-L106|local_tile(mA, cta_tiler, cta_coord, …)]]
- 每个线程搬哪些：[[cutlass:examples/cute/tutorial/sgemm_1.cu@L120-L124|local_partition(gA, tA, threadIdx.x)]]
- 每个线程算哪些：[[cutlass:examples/cute/tutorial/sgemm_1.cu@L138-L142|local_partition(sA, tC, threadIdx.x, Step…)]]
- host 端的配置全是 layout：[[cutlass:examples/cute/tutorial/sgemm_1.cu@L273-L286|bM、bN、bK，sA、sB，tA = 32×8 个线程，tC = 16×16 个线程]]

### 线程划分：同一个 tile，换一种线程 layout

`local_partition(tile, thr_layout, tid)` 的意思是：把 tile 按线程 layout 的形状切开，第 tid 个线程拿走属于它的那些元素。下面的演示把一个 8×8 的 tile 分给几个线程，对比两种排法。kernel 代码完全一样，变的只是一个 layout。

::: raw
<div class="cw" id="cw-part">
  <div class="cw-head"><b>演示 4：把 8×8 的 tile 分给线程</b></div>
  <div class="cw-bar"><span>线程 layout：</span><span data-g="pt"></span></div>
  <div class="cw-bar"><span>分法：</span><span data-g="pm"></span></div>
  <div class="cw-body">
    <div class="cw-row">
      <div><div class="cw-cap">每格写的是哪个线程拿它（悬停一个格子，高亮这个线程的全部元素）</div><div class="cw-grid" data-g="grid"></div></div>
    </div>
    <div class="cw-status" data-g="status"></div>
  </div>
</div>
<script>
(function () {
  var C = Cute, U = CuteUI, root = document.getElementById("cw-part");
  function q(s) { return root.querySelector('[data-g="' + s + '"]'); }
  var TS = [[2, 2], [4, 2], [2, 4], [8, 1]], tm = 2, tn = 2, mode = "raked", hover = -1;
  TS.forEach(function (t, k) { var b = document.createElement("button"); b.type = "button"; b.className = "btn" + (k === 0 ? " on" : ""); b.textContent = "(" + t[0] + "," + t[1] + ")";
    b.title = t[0] * t[1] + " 个线程"; b.onclick = function () { tm = t[0]; tn = t[1]; q("pt").querySelectorAll(".btn").forEach(function (x) { x.classList.toggle("on", x === b); }); render(); }; q("pt").appendChild(b); });
  [["raked", "交错（local_partition）"], ["blocked", "分块（每个线程一整块）"]].forEach(function (p, k) { var b = document.createElement("button"); b.type = "button"; b.className = "btn" + (k === 0 ? " on" : ""); b.textContent = p[1];
    b.onclick = function () { mode = p[0]; q("pm").querySelectorAll(".btn").forEach(function (x) { x.classList.toggle("on", x === b); }); render(); }; q("pm").appendChild(b); });
  function owner(r, c) {
    var thr = C.L([tm, tn], [1, tm]);                    // thread layout (tm,tn):(1,tm): thread id of a thread coordinate
    if (mode === "raked") return C.call(thr, [r % tm, c % tn]);           // element (r,c) goes to thread (r mod tm, c mod tn)
    return C.call(thr, [Math.floor(r / (8 / tm)), Math.floor(c / (8 / tn))]);   // each thread owns one (8/tm)x(8/tn) block
  }
  function render() {
    var n = tm * tn, g = q("grid"); g.innerHTML = ""; g.style.gridTemplateColumns = "28px repeat(8, 34px)";
    g.appendChild(U.lab("")); for (var c = 0; c < 8; c++) { var h = U.lab(String(c)); h.style.textAlign = "center"; g.appendChild(h); }
    for (var r = 0; r < 8; r++) { g.appendChild(U.lab(String(r)));
      for (var c = 0; c < 8; c++) { var t = owner(r, c), el = U.cell("t" + t, "f", U.hue(t, n), "(" + r + "," + c + ") → 线程 " + t);
        if (t === hover) el.classList.add("hl");
        (function (t) { el.onmouseenter = function () { hover = t; render(); }; })(t);
        g.appendChild(el); } }
    g.onmouseleave = function () { if (hover !== -1) { hover = -1; render(); } };
    var per = 64 / n, ex = [];
    for (var r = 0; r < 8; r++) for (var c = 0; c < 8; c++) if (owner(r, c) === 0) ex.push("(" + r + "," + c + ")");
    q("status").innerHTML = (mode === "raked"
      ? "交错分法：元素 (r, c) 给线程 (r mod " + tm + ", c mod " + tn + ")。相邻的元素属于相邻的线程，所以一个 warp 读一行时地址连续，<b>合并访存</b>。这就是 sgemm_1 里 local_partition 的默认分法，也是 C1、C2 里 grid-stride 的“发牌”思路。"
      : "分块分法：每个线程拿一整块 " + (8 / tm) + "×" + (8 / tn) + "。每个线程自己的数据连续，适合放进寄存器做外积（C4 的 4×4 寄存器 tile 就是这样），但同一时刻相邻线程读的地址相隔很远。") +
      "　共 " + n + " 个线程，每个线程 " + per + " 个元素。线程 0 拿：" + ex.slice(0, 8).join(" ") + (ex.length > 8 ? " …" : "");
  }
  render();
})();
</script>
:::

真实的 kernel 往往两种都用：**搬数据用交错分法**（合并访存），**算的时候用分块分法**（寄存器里复用）。sgemm_1 里搬运用 `tA`（[[cutlass:examples/cute/tutorial/sgemm_1.cu@L120|local_partition(gA, tA, …)]]），计算用另一个线程 layout `tC`（[[cutlass:examples/cute/tutorial/sgemm_1.cu@L138-L142|local_partition(sA, tC, …)]]）。两者是两个不同的 layout，同一份数据在两次划分之间通过 shared memory 交接，这正是 C4 里协作加载和寄存器 tile 的关系。

::: callout analogy 工位分配表
一批货（tile）进仓库，要分给几个工人（线程）。“交错分法”像流水线：一排货依次滑过，第 0 个工人拿第 0 件，第 1 个拿第 1 件，大家同时伸手，一次拿走一整排（合并访存）。“分块分法”像包干到人：每人认领一个货架，自己的货都在一起，加工时不用走动（寄存器复用）。CuTe 里换分法只是换一张“工位分配表”（线程 layout），工人干活的步骤（kernel 代码）一个字都不用改。
:::

## ⑦ 硬件也用 Layout 说话

CuTe 最强的地方在最后一层：**GPU 的矩阵指令和拷贝指令，本身就有固定的数据排列要求**，CuTe 把这些要求也写成 layout。

::: flow 指令的要求也是 layout
cfg cw=240 nw=190
n mma 0 0 "MMA Atom" "mma.sync / wgmma / tcgen05" k=vio
n tv 1 0 "TV layout" "(线程, 值) → 矩阵坐标" k=acc
n cp 0 1.3 "Copy Atom" "cp.async / TMA / ldmatrix" k=vio
n sw 1 1.3 "smem layout ∘ Swizzle" "无 bank conflict" k=acc w=1.05
n k 2.1 0.65 "kernel" "只组合 layout，不手写下标" k=pass
e mma -> tv
e cp -> sw
e tv -> k
e sw -> k
cap: 一条 mma.sync m16n8k16 指令要求 32 个线程的寄存器按一种特定的方式持有 A、B、C 的元素。CuTe 把这个要求写成一个“(线程号, 值号) → (行, 列)”的 layout（TV layout）。换到 Hopper 的 wgmma，只是换一个 Atom，上层用同样的 divide / partition 去对齐它。
:::

- **TV layout**：G1 里你写的“lane 持有哪几个元素”的计算器，就是一个 TV layout。CuTe 用它描述每条 MMA 指令的寄存器排列。
- **Swizzle**：C3 讲过的异或技巧也是一个 layout（[[cutlass:include/cute/swizzle.hpp@L55|Swizzle]]），和 shared memory 的 layout 组合起来就消除了 bank conflict，不需要 padding。
- 文档：[[cutlass:media/docs/cpp/cute/0t_mma_atom.md|0t_mma_atom.md]] · [[cutlass:media/docs/cpp/cute/0x_gemm_tutorial.md|0x_gemm_tutorial.md]] · [[cutlass:media/docs/cpp/cute/0z_tma_tensors.md|0z_tma_tensors.md]]

这也是为什么 FlashAttention-3/4、FlashInfer 的 Hopper / Blackwell 后端都建立在 CuTe 上：新一代硬件来了，只需要补上新指令的 Atom（它的 layout），上层的 tiling 和 partition 代码大部分可以复用。CuTe DSL 则把同一套概念搬到了 Python 里。

## ⑧ 参考解 ↔ CuTe

| 参考解 [[here:solutions/g2_cute_layout.py|g2_cute_layout.py]] | CuTe C++ | pycute（官方 Python 参考） |
|---|---|---|
| [[here:solutions/g2_cute_layout.py@L46-L58|Layout、__call__、size、cosize]] | [[cutlass:include/cute/layout.hpp|layout.hpp]] | [[cutlass:python/pycute/layout.py@L51|class Layout]] |
| [[here:solutions/g2_cute_layout.py@L26-L35|_idx2crd（列字典序）]] | `idx2crd` | — |
| [[here:solutions/g2_cute_layout.py@L89-L104|coalesce]] | [[cutlass:include/cute/layout.hpp@L867|coalesce]] | [[cutlass:python/pycute/layout.py@L137|coalesce]] |
| [[here:solutions/g2_cute_layout.py@L107-L135|composition]] | [[cutlass:include/cute/layout.hpp@L1136|composition]] | [[cutlass:python/pycute/layout.py@L190|composition]] |
| [[here:solutions/g2_cute_layout.py@L138-L152|complement]] | [[cutlass:include/cute/layout.hpp@L1274|complement]] | [[cutlass:python/pycute/layout.py@L232|complement]] |
| [[here:solutions/g2_cute_layout.py@L155-L159|logical_divide]] | [[cutlass:include/cute/layout.hpp@L1599|logical_divide]] | [[cutlass:python/pycute/layout.py@L297|logical_divide]] |

这一页的四个演示用的就是参考解这六个函数的 JavaScript 移植版，和 Python 版在 55 个例子上逐个对拍过，结果完全一致。

::: answer 自测 1：(4,4):(1,4) 和 (4,4):(4,1) 合成（前者 ∘ 后者）是什么？意味着什么？
A = 列主序，B 按“行主序”的顺序选下标：R(i) = A(B(i))。B 把下标 i 当作行主序的 (i/4, i%4) 去取 A 的元素，结果是 `(4,4):(4,1)`。在演示 2 里选 A = `(4,4):(1,4)`、B = `(4,4):(4,1)` 就能看到。和第 ⑤ 节的“转置”例子正好对称。
:::

::: answer 自测 2：logical_divide(32:1, 8:1) 是什么？complement 是什么？
complement(8:1, 32) = `4:8`（4 个 tile，起点相差 8）。logical_divide = `(8,4):(1,8)`：4 个连续的 tile，每个 8 个元素。在演示 3 里选 A = `32:1`、B = `8:1` 验证。
:::

## 源码与资料索引

**本仓库**：[[here:solutions/g2_cute_layout.py|solutions/g2_cute_layout.py]] · [[here:tests/test_g2_cute_layout.py|tests/test_g2_cute_layout.py]] · [[here:content/g2.md|content/g2.md]] · [G2 讲解](g2_cute_layout.html) · [G1 讲解](g1_layouts.html)

**CuTe 文档**：[[cutlass:media/docs/cpp/cute/00_quickstart.md|00_quickstart]] · [[cutlass:media/docs/cpp/cute/01_layout.md|01_layout]] · [[cutlass:media/docs/cpp/cute/02_layout_algebra.md|02_layout_algebra]] · [[cutlass:media/docs/cpp/cute/03_tensor.md|03_tensor]] · [[cutlass:media/docs/cpp/cute/0x_gemm_tutorial.md|0x_gemm_tutorial]]

**代码**：[[cutlass:examples/cute/tutorial/sgemm_1.cu@L52|sgemm_1.cu]] · [[cutlass:python/pycute/layout.py|pycute/layout.py]] · [[cutlass:include/cute/layout.hpp|layout.hpp]] · [[cutlass:examples/python/CuTeDSL/cute/notebooks/cute_layout_algebra.ipynb|CuTe DSL notebook]]
