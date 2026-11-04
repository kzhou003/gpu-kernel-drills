"""Build index.html (one self-contained page) from content/*.md + the code in mine/, tests/, solutions/.

    python build/build.py        # needs markdown-it-py

Markdown extensions (fenced containers, nesting allowed; fenced code is skipped when scanning):
    ::: hint 1 标题        L1/L2/L3 collapsible hint
    ::: solution 标题      collapsible reference solution (red, "write yours first")
    ::: answer 标题        click-to-reveal answer
    ::: callout info 标题  boxed note; kinds: info | warn | ok | idea
    ::: raw                emitted verbatim
    :::                    closes the innermost container
Line macro:
    {{code: path/to/file [title]}}   embeds the file as a highlighted code block with a copy button
Each content file starts with `lede: one sentence`.
"""
import html
import json
import re
import shlex
from pathlib import Path

from markdown_it import MarkdownIt

import sources as SRC

LINK_ERRORS: list[str] = []

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"

# (id, file stem or None for a lab page, title, minutes, difficulty, track)
DRILLS = [
    ("C1", "c1_elementwise", "Vector add + 融合 bias-GELU：索引、launch 配置、fusion", 35, "易", "C"),
    ("C2", "c2_reduce_sum", "Reduction：warp shuffle + shared memory + atomic", 45, "中", "C"),
    ("C3", "c3_transpose", "矩阵转置：coalescing 与 bank conflict", 40, "中", "C"),
    ("C4", "c4_gemm", "GEMM 三连：naive → shared-memory tiling → register tiling", 90, "难", "C"),
    ("T1", "t1_add_gelu", "Triton 入门：融合 add + GELU", 25, "易", "T"),
    ("T2", "t2_softmax", "融合 row softmax：数值稳定 + 一次读写", 35, "中", "T"),
    ("T3", "t3_layernorm", "LayerNorm / RMSNorm forward", 40, "中", "T"),
    ("T4", "t4_matmul", "Triton GEMM：tl.dot、grouped ordering、autotune", 60, "中", "T"),
    ("T5", "t5_flash_attn", "FlashAttention forward：tiling + online softmax", 90, "难", "T"),
    ("L1", "l1_paged_kv_write", "Paged KV cache：slot mapping + 写 cache", 35, "易", "L"),
    ("L2", "l2_paged_decode", "Paged decode attention + GQA", 60, "难", "L"),
    ("L3", "l3_split_kv_decode", "Split-KV decode（Flash-Decoding）+ LSE 合并", 60, "难", "L"),
    ("F1", "f1_flashinfer_paged", "FlashInfer 分页格式（CSR）+ plan / run decode", 40, "易", "F"),
    ("F2", "f2_merge_state", "merge_state / merge_states：以 2 为底的 LSE 合并", 40, "中", "F"),
    ("F3", "f3_fused_norm_act", "fused_add_rmsnorm（原地）+ silu_and_mul", 35, "中", "F"),
    ("F4", "f4_topk_topp_renorm", "不排序的 top-k / top-p：对浮点位模式二分", 60, "难", "F"),
    ("F5", None, "FlashInfer 贡献实战：源码地图、JIT、测试、benchmark、PR 规范", 180, "中", "F"),
    ("G1", "g1_layouts", "Layout 计算器：BlockedLayout / SliceLayout 到线程的映射", 45, "中", "G"),
    ("G2", "g2_cute_layout", "CuTe 布局代数：coalesce、composition、complement、divide", 60, "难", "G"),
    ("M1", "m1_moe_align", "MoE 路由 + moe_align_block_size", 40, "中", "M"),
    ("M2", "m2_fused_moe", "Fused MoE grouped GEMM（vLLM fused_moe 结构）", 75, "难", "M"),
    ("M3", "m3_fp8_block_gemm", "FP8 细粒度量化 + block-scaled GEMM（DeepSeek-V3 / DeepGEMM 配方）", 60, "难", "M"),
    ("X1", "x1_metal_elementwise", "MLX 自定义 Metal kernel 入门：融合 add + GELU", 30, "易", "X"),
    ("X2", "x2_metal_softmax", "Metal 归约：simdgroup + threadgroup 内存的 softmax / RMSNorm", 45, "中", "X"),
    ("X3", "x3_metal_matmul", "Metal GEMM：threadgroup tiling → simdgroup_matrix", 75, "难", "X"),
]
# track -> (title, description, tier)
TRACKS = {
    "C": ("Track C · CUDA C++", "硬件映射：线程索引、合并访存、shared memory、warp 原语、数据复用。", "核心"),
    "T": ("Track T · Triton", "block 级编程、mask、融合、reduction、tl.dot、autotune、FlashAttention。", "核心"),
    "L": ("Track L · 推理 attention 从零写", "Paged KV cache、decode attention、GQA、split-KV：真实 serving 的形状。", "核心"),
    "F": ("Track F · FlashInfer", "对着 FlashInfer 的真实 API 实现并逐数值对拍，最后进入它的源码和贡献流程。", "核心"),
    "G": ("Track G · Layout / CuTe", "数据在 warp / lane / register 上怎么分布；CuTe 的布局代数。", "进阶"),
    "M": ("Track M · MoE 与低精度", "路由与对齐、grouped GEMM、FP8 block scaling：DeepSeek-V3 / DeepGEMM 的配方。", "进阶"),
    "X": ("Track X · MLX / Metal", "Apple silicon 上的自定义 kernel：elementwise、simdgroup 归约、simdgroup_matrix GEMM。", "进阶"),
}
BASICS = ["intro", "basics_hw", "basics_cuda", "basics_triton", "basics_workflow", "landscape"]
HINT_NOTES = {"1": "方向（轻度剧透）", "2": "拆解与常见坑（中度剧透）", "3": "几乎是答案（重度剧透）"}

MD = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable(["table", "strikethrough"])
OPEN_RE = re.compile(r"^:::\s+(\w+)\s*(.*)$")
CLOSE_RE = re.compile(r"^:::\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
CODE_RE = re.compile(r"^\{\{code:\s*(\S+)\s*(.*?)\}\}\s*$")
LANG = {".py": "python", ".cu": "cpp", ".sh": "bash"}


def split_blocks(lines):
    out, i, in_fence = [], 0, False
    while i < len(lines):
        line = lines[i]
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        m = None if in_fence else OPEN_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue
        depth, j, inner, f = 1, i + 1, [], False
        while j < len(lines):
            ln = lines[j]
            if FENCE_RE.match(ln):
                f = not f
            elif not f and OPEN_RE.match(ln):
                depth += 1
            elif not f and CLOSE_RE.match(ln):
                depth -= 1
                if depth == 0:
                    break
            inner.append(ln)
            j += 1
        if depth:
            raise SystemExit(f"unclosed ::: {m.group(1)} near {line!r}")
        out.append((m.group(1), m.group(2).strip(), inner))
        i = j + 1
    return out


def code_block(path: str, title: str = "") -> str:
    p = ROOT / path
    text = p.read_text(encoding="utf-8").rstrip("\n")
    lang = LANG.get(p.suffix, "plaintext")
    label = html.escape(title or path)
    n = text.count("\n") + 1
    return (f'<div class="codebox"><div class="code-head"><span class="path">{label}</span>'
            f'<span class="muted small">{n} 行</span><button type="button" class="btn copy">复制</button></div>'
            f'<pre><code class="language-{lang}">{html.escape(text)}</code></pre></div>')



def parse_flow(title, lines):
    """Tiny diagram DSL -> JSON for the React-Flow-style renderer in template.html.

        cfg cw=200 rh=90 nw=160               column width, row height, default node width (px)
        n id col row "title" ["sub"] [k=acc] [w=1.4]
        e a[:side] -> b[:side] ["label"] [k=acc] [anim] [dash]     side: t b l r (auto if omitted)
        g col0 row0 col1 row1 "label" [k=acc]                       group box around a grid range
        cap: caption text (markdown inline allowed)
    """
    spec = {"title": title, "nodes": [], "edges": [], "groups": [], "cfg": {}}
    caption = ""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("cap:"):
            caption = line[4:].strip()
            continue
        tok = shlex.split(line)
        kind, rest = tok[0], tok[1:]
        is_opt = lambda t: re.fullmatch(r"(k|w|cw|rh|nw)=\S+", t) is not None   # only real options, not "i = 2·256"
        opts = {k: v for k, _, v in (t.partition("=") for t in rest if is_opt(t))}
        flags = {t for t in rest if not is_opt(t)}
        if kind == "cfg":
            spec["cfg"] = {k: float(v) for k, v in opts.items()}
        elif kind == "n":
            nid, c, r = rest[0], float(rest[1]), float(rest[2])
            texts = [t for t in rest[3:] if not is_opt(t)]
            spec["nodes"].append({"id": nid, "c": c, "r": r, "t": texts[0] if texts else nid,
                                  "s": texts[1] if len(texts) > 1 else "", "k": opts.get("k", ""),
                                  "w": float(opts.get("w", 1))})
        elif kind == "e":
            a, arrow, b = rest[0], rest[1], rest[2]
            assert arrow == "->", line
            texts = [t for t in rest[3:] if not is_opt(t) and t not in ("anim", "dash")]
            spec["edges"].append({"a": a, "b": b, "t": texts[0] if texts else "", "k": opts.get("k", ""),
                                  "anim": "anim" in flags, "dash": "dash" in flags})
        elif kind == "g":
            c0, r0, c1, r1 = map(float, rest[:4])
            texts = [t for t in rest[4:] if not is_opt(t)]
            spec["groups"].append({"c0": c0, "r0": r0, "c1": c1, "r1": r1, "t": texts[0] if texts else "",
                                   "k": opts.get("k", "")})
        else:
            raise SystemExit(f"flow: unknown line {line!r}")
    ids = {n["id"] for n in spec["nodes"]}
    cw, rh, nw = spec["cfg"].get("cw", 200), spec["cfg"].get("rh", 88), spec["cfg"].get("nw", 160)
    boxes = [(n["id"], n["c"] * cw, n["r"] * rh, n["c"] * cw + nw * n["w"], n["r"] * rh + (52 if n["s"] else 38)) for n in spec["nodes"]]
    for i, A in enumerate(boxes):                       # layout lint: nodes must not overlap (12 px gap)
        for B in boxes[i + 1:]:
            if A[1] < B[3] + 12 and B[1] < A[3] + 12 and A[2] < B[4] + 8 and B[2] < A[4] + 8:
                raise SystemExit(f"flow {title!r}: nodes {A[0]!r} and {B[0]!r} overlap")
    for e in spec["edges"]:
        for end in (e["a"], e["b"]):
            if end.split(":")[0] not in ids:
                raise SystemExit(f"flow {title!r}: edge references unknown node {end!r}")
    cap = MD.renderInline(SRC.render(caption, LINK_ERRORS)) if caption else ""
    return (f'<figure class="flow"><div class="flow-head">{html.escape(title)}<span class="flow-hint">悬停节点查看连接</span></div>'
            f'<div class="flow-body" data-flow="{html.escape(json.dumps(spec, ensure_ascii=False))}"></div>'
            + (f"<figcaption>{cap}</figcaption>" if cap else "") + "</figure>")


def md(text: str) -> str:
    out = MD.render(SRC.render(text, LINK_ERRORS))
    out = out.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return out


def render(lines) -> str:
    parts, buf = [], []

    def flush():
        if buf:
            parts.append(md("\n".join(buf)))
            buf.clear()

    for b in split_blocks(lines):
        if isinstance(b, str):
            m = CODE_RE.match(b)
            if m:
                flush()
                parts.append(code_block(m.group(1), m.group(2)))
            else:
                buf.append(b)
            continue
        flush()
        name, args, inner = b
        if name == "raw":
            parts.append("\n".join(inner))
        elif name == "hint":
            lvl, _, title = args.partition(" ")
            parts.append(f'<details class="hint l{lvl}"><summary><span class="lvl">L{lvl}</span>{html.escape(title)}'
                         f'<span class="spoiler-note">{HINT_NOTES.get(lvl, "")}</span></summary>'
                         f'<div class="body">{render(inner)}</div></details>')
        elif name == "solution":
            parts.append(f'<details class="solution"><summary><span class="lvl">解</span>{html.escape(args or "参考解")}'
                         f'<span class="spoiler-note">先自己写完、跑过测试再看</span></summary>'
                         f'<div class="body">{render(inner)}</div></details>')
        elif name == "answer":
            parts.append(f'<details class="answer"><summary>{html.escape(args or "显示答案")}</summary>'
                         f'<div class="body">{render(inner)}</div></details>')
        elif name == "flow":
            parts.append(parse_flow(args, inner))
        elif name == "callout":
            kind, _, title = args.partition(" ")
            label = "类比 · " if kind == "analogy" else ""
            t = f'<div class="callout-title">{label}{html.escape(title)}</div>' if (title or label) else ""
            parts.append(f'<div class="callout {kind}">{t}{render(inner)}</div>')
        else:
            raise SystemExit(f"unknown container ::: {name}")
    flush()
    return "\n".join(parts)


def read_src(name):
    lines = (CONTENT / f"{name}.md").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("lede:"), f"{name}.md must start with 'lede:'"
    return lines[0][5:].strip(), lines[1:]


def with_ids(body: str, prefix: str):
    toc = []

    def sub(m):
        toc.append((f"{prefix}-{len(toc) + 1}", html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))))
        return f'<h3 id="{toc[-1][0]}">{m.group(1)}</h3>'

    return re.sub(r"<h2>(.*?)</h2>", sub, body), toc


def basics_section(name, idx):
    lede, lines = read_src(name)
    body, toc = with_ids(render(lines), name)
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("# ") else name
    if lines and lines[0].startswith("# "):
        body = body.split("</h1>", 1)[1] if "</h1>" in body else body
    sub = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc)
    return (f'<section class="part" id="{name}"><div class="kicker">Part 0 · 基础 {idx}</div>'
            f'<h2>{html.escape(title)}</h2><p class="lede">{html.escape(lede)}</p>'
            + (f'<nav class="mini-toc"><ol>{sub}</ol></nav>' if toc else "") + body + "</section>"), title, toc


HINTS = ROOT / "hints"


def hint_page(did, stem):
    """hints/<stem>.html for a drill (F5 has no stem: match by its id prefix), or None."""
    if stem and (HINTS / f"{stem}.html").exists():
        return f"hints/{stem}.html"
    for p in sorted(HINTS.glob(f"{did.lower()}_*.html")):
        return f"hints/{p.name}"
    return None


def hint_topics():
    """Hint pages that belong to no drill: (href, title from hints/src/<name>.md)."""
    drill_pages = {hint_page(d[0], d[1]) for d in DRILLS}
    out = []
    for src in sorted((HINTS / "src").glob("*.md")):
        href = f"hints/{src.stem}.html"
        if href not in drill_pages and (ROOT / href).exists():
            out.append((href, src.read_text(encoding="utf-8").splitlines()[0][6:].strip()))
    return out


def drill_section(k):
    did, stem, title, mins, diff, track = DRILLS[k]
    lede, lines = read_src(did.lower())
    body, toc = with_ids(render(lines), did.lower())
    ext = ".cu" if track == "C" else ".py"
    sub = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc)
    if stem:
        files = (f'<span class="badge">你写 <code>mine/{stem}{ext}</code></span>'
                 f'<span class="badge">验收 <code>./check.sh {did.lower()}</code></span>'
                 f'<span class="badge">参考解 <code>solutions/{stem}{ext}</code></span>')
    else:
        files = '<span class="badge">实践 · 在 FlashInfer 仓库里动手，没有 check.sh</span>'
    hp = hint_page(did, stem)
    if hp:
        files += f'<a class="badge hint-link" href="{hp}">📖 <b>逐行讲解</b></a>'
    head = f"""<div class="drill-head">
<div class="kicker">{TRACKS[track][2]} · {TRACKS[track][0]} · 第 {k + 1} 题</div>
<h2><span class="did">{did}</span>{html.escape(title)}</h2>
<p class="lede">{html.escape(lede)}</p>
<div class="badges">
<span class="badge">⏱ <b>{mins} min</b></span>
<span class="badge diff-{diff}">难度 <b>{diff}</b></span>
{files}
<label class="badge done-toggle"><input type="checkbox" class="done-box" data-id="{did}">标记完成</label>
</div>
<nav class="mini-toc"><ol>{sub}</ol></nav>
</div>"""
    return f'<section class="drill" id="{did.lower()}" data-id="{did}">{head}{body}</section>', toc


def page():
    parts, nav = [], []
    # Part 0
    nav.append('<div class="nav-tier">Part 0</div><div class="nav-group">基础与全景</div>')
    for i, name in enumerate(BASICS):
        html_, title, _ = basics_section(name, i)
        parts.append(html_)
        nav.append(f'<a class="nav-item" href="#{name}">{html.escape(title)}</a>')
    # drills
    tier_seen = set()
    for t, (tname, tdesc, tier) in TRACKS.items():
        if tier not in tier_seen:
            tier_seen.add(tier)
            nav.append(f'<div class="nav-tier">{tier}</div>')
            parts.append(f'<div class="tier-banner" id="tier-{tier}">{tier}</div>')
        nav.append(f'<div class="nav-group">{html.escape(tname)}</div>')
        parts.append(f'<div class="track-banner" id="track-{t}"><h2>{html.escape(tname)}</h2><p>{html.escape(tdesc)}</p></div>')
        for k, d in enumerate(DRILLS):
            if d[5] != t:
                continue
            sec, _ = drill_section(k)
            parts.append(sec)
            nav.append(f'<a class="nav-item drill-link" href="#{d[0].lower()}" data-id="{d[0]}">'
                       f'<span class="nid">{d[0]}</span>{html.escape(d[2].split("：")[0])}<span class="nmin">{d[3]}′</span></a>')
    # stand-alone explainer pages (hints/, built by build_hints.py)
    nav.append('<div class="nav-tier">提示与讲解</div>'
               '<a class="nav-item" href="hints/index.html">全部讲解页（每题一页）</a>'
               '<div class="nav-group">专题</div>')
    for href, t in hint_topics():
        nav.append(f'<a class="nav-item" href="{href}">{html.escape(t.split("：")[0])}</a>')
    html_, title, _ = basics_section("interview", "")
    parts.append(html_.replace("Part 0 · 基础 ", "面试"))
    nav.append(f'<div class="nav-tier">收尾</div><a class="nav-item" href="#interview">{html.escape(title)}</a>')
    lede, lines = read_src("appendix")
    app_body, _ = with_ids(render(lines), "appendix")
    parts.append(f'<section class="part" id="appendix"><div class="kicker">附录</div><h2>附录：下一步与参考资料</h2>'
                 f'<p class="lede">{html.escape(lede)}</p>{app_body}</section>')
    nav.append('<a class="nav-item" href="#appendix">附录：下一步与参考资料</a>')

    total = sum(d[3] for d in DRILLS)
    template = (ROOT / "build" / "template.html").read_text(encoding="utf-8")
    return (template.replace("{{NAV}}", "\n".join(nav))
            .replace("{{BODY}}", "\n".join(parts))
            .replace("{{N_DRILLS}}", str(len(DRILLS)))
            .replace("{{TOTAL_H}}", f"{total // 60} 小时 {total % 60} 分"))


if __name__ == "__main__":
    out = ROOT / "index.html"
    html_out = page()
    if LINK_ERRORS:
        raise SystemExit("unresolved source links:\n  " + "\n  ".join(sorted(set(LINK_ERRORS))))
    out.write_text(html_out, encoding="utf-8")
    print("built", out, f"{out.stat().st_size // 1024} KB")
