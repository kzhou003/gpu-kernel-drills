"""Build the stand-alone explainer pages in hints/ from hints/src/*.md.

    python3 build/build_hints.py            # every hints/src/*.md -> hints/<name>.html

Same Markdown dialect as content/*.md (see build.py: ::: flow / callout / answer / raw, [[repo:path@symbol]]),
plus two link macros that build.py does not have:
    [[here:solutions/c1_elementwise.cu@L16|label]]   this repo, pinned to the last commit that touched the file
    [[llvm:clang/lib/Headers/x.h@L79|label]]         llvm/llvm-project at LLVM_SHA
`@L123` / `@L10-L20` works for every repo alias (exact line instead of a symbol search).
The first line of each source is `title: ...`, the second `lede: ...`.
"""
import html
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build as B  # noqa: E402  (reuses render(), the flow DSL and the pinned source links)

ROOT = B.ROOT
SRC_DIR = ROOT / "hints" / "src"
OUT_DIR = ROOT / "hints"
REPO = "kzhou003/gpu-kernel-drills"
LLVM_SHA = "3c4aa885091c12572281a7117dd254907d8cd732"
EXTRA = re.compile(r"\[\[(here|llvm):([^\]|@]+)(?:@(L\d+(?:-L\d+)?))?(?:\|([^\]]+))?\]\]")


def last_commit(path):
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", path], cwd=ROOT, capture_output=True, text=True)
    sha = r.stdout.strip()
    if not sha:
        raise SystemExit(f"{path}: not committed, cannot pin a link to it")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path], cwd=ROOT).returncode
    if dirty:
        print(f"warning: {path} has uncommitted changes; line links point at {sha[:7]}")
    return sha


def extra_links(text):
    def sub(m):
        alias, path, line, label = m.group(1), m.group(2).strip(), m.group(3), m.group(4)
        if alias == "here":
            if not (ROOT / path).exists():
                B.LINK_ERRORS.append(f"here: no file {path}")
                return m.group(0)
            repo, sha = REPO, last_commit(path)
        else:
            repo, sha = "llvm/llvm-project", LLVM_SHA
        url = f"https://github.com/{repo}/blob/{sha}/{path}" + (f"#{line}" if line else "")
        name = label or path.split("/")[-1]
        return (f'<a class="src" href="{url}" title="{repo} @ {sha[:7]} · {path}" target="_blank" rel="noopener">'
                f'<span class="src-repo">{"drills" if alias == "here" else alias}</span>{html.escape(name)}</a>')
    # leave inline/fenced code alone, like sources.render does
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    return "".join(p if i % 2 else EXTRA.sub(sub, p) for i, p in enumerate(parts))


_md = B.md
B.md = lambda text: _md(extra_links(text))   # render() calls md() for every Markdown chunk

PAGE_CSS = """
.page { max-width: 900px; }
.qnav { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 4px; }
.qnav a { border: 1px solid var(--border); background: var(--surface); border-radius: 999px; padding: 1px 10px; font-size: 13px; }
.qnav a:hover { border-color: var(--accent); text-decoration: none; }
.q { font-size: 13px; color: var(--muted); margin: -4px 0 10px; }
.q b { color: var(--text); }
/* element/thread cell strips */
.cells { display: grid; gap: 3px; margin: 8px 0; }
.cells > div { border: 1px solid var(--border); border-radius: 5px; padding: 2px 0; text-align: center; font: 11.5px/1.35 var(--mono); background: var(--surface); }
.cells > div small { display: block; color: var(--muted); font-size: 10px; }
.cells .lbl { border: 0; background: none; text-align: right; padding-right: 6px; color: var(--muted); font-family: var(--font); }
.c0 { background: var(--f-acc-bg) !important; border-color: var(--f-acc) !important; }
.c1 { background: var(--f-warn-bg) !important; border-color: var(--f-warn) !important; }
.c2 { background: var(--f-pass-bg) !important; border-color: var(--f-pass) !important; }
.c3 { background: var(--f-vio-bg) !important; border-color: var(--f-vio) !important; }
.cx { background: repeating-linear-gradient(135deg, var(--surface), var(--surface) 4px, var(--surface-2) 4px, var(--surface-2) 8px) !important; color: var(--muted); border-style: dashed !important; }
.strip-scroll { overflow-x: auto; }
td:first-child code { white-space: nowrap; }
.legend { font-size: 12.5px; color: var(--muted); margin: 2px 0 10px; }
.legend span { display: inline-block; width: 11px; height: 11px; border-radius: 3px; vertical-align: -1px; margin: 0 3px 0 10px; border: 1px solid var(--border); }
"""


def drill_of(name):
    for d in B.DRILLS:
        if d[1] == name or (d[1] is None and name.startswith(d[0].lower() + "_")):
            return d
    return None


def kicker(name):
    d = drill_of(name)
    if not d:
        return "提示与讲解"
    ext = ".cu" if d[5] == "C" else ".py"
    files = f" · 对照 <code>solutions/{d[1]}{ext}</code> 逐行读" if d[1] else ""
    return f"{d[0]} 提示与讲解 · {html.escape(B.TRACKS[d[5]][0])}{files}"


def index_page(names):
    rows, cur = [], None
    for d in B.DRILLS:
        n = next((x for x in names if drill_of(x) is d), None)
        if d[5] != cur:
            cur = d[5]
            rows.append(f'<h3>{html.escape(B.TRACKS[cur][0])}</h3><p class="muted small">{html.escape(B.TRACKS[cur][1])}</p>')
        title = html.escape(d[2])
        link = f'<a href="{n}.html">{title}</a>' if n else f'<span class="muted">{title}（待写）</span>'
        rows.append(f'<div class="hrow"><span class="did">{d[0]}</span>{link}<span class="muted small">{d[3]} min · {d[4]}</span></div>')
    extra = [x for x in names if drill_of(x) is None]
    if extra:
        rows.append('<h3>专题</h3><p class="muted small">跨题目的系统讲解。</p>')
        for x in extra:
            t = (SRC_DIR / f"{x}.md").read_text(encoding="utf-8").splitlines()[0][6:].strip()
            rows.append(f'<div class="hrow"><span class="did">专题</span><a href="{x}.html">{html.escape(t)}</a></div>')
    css = """.hrow { display: flex; gap: 10px; align-items: baseline; padding: 6px 2px; border-bottom: 1px dashed var(--border); }
.hrow .did { font: 700 12px var(--mono); color: #fff; background: var(--accent); border-radius: 6px; padding: 0 7px; }
.hrow .small { margin-left: auto; white-space: nowrap; }"""
    return wrap("Kernel Dev 训练营 · 提示与讲解", "每道题一页：背景、参考解逐段讲解、图解、类比和钉死版本的源码链接。建议先自己写，卡住了再来看。",
                "提示与讲解", "".join(rows), "", css)


def page(name):
    lines = (SRC_DIR / f"{name}.md").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("title:") and lines[1].startswith("lede:"), f"{name}.md must start with title:/lede:"
    title, lede = lines[0][6:].strip(), lines[1][5:].strip()
    body, toc = B.with_ids(B.render(lines[2:]), name)
    qnav = "".join(f'<a href="#{a}">{html.escape(t)}</a>' for a, t in toc)
    return wrap(title, lede, kicker(name), body, qnav)


def wrap(title, lede, kick, body, qnav, extra_css=""):
    template = (ROOT / "build" / "template.html").read_text(encoding="utf-8")
    style = template[template.index("<style>"): template.index("</style>")] + PAGE_CSS + extra_css + "</style>"
    flow_js = template[template.index("<script>"): template.index("</script>") + 9]   # the diagram renderer
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(lede)}">
{style}
</head>
<body>
<header class="topbar">
  <a class="home" href="../index.html">Kernel Dev 训练营</a><a href="index.html">/ 提示与讲解</a>
  <span class="spacer"></span>
  <button type="button" class="btn" id="theme-btn" title="切换明暗">◐</button>
</header>
<main class="page" id="top">
<section class="hero">
  <div class="kicker">{kick}</div>
  <h1>{html.escape(title)}</h1>
  <p class="lede">{html.escape(lede)}</p>
  {f'<nav class="qnav">{qnav}</nav>' if qnav else ""}
</section>
{body}
</main>
{flow_js}
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>
(function () {{
  var root = document.documentElement, KEY = "gkd-theme";
  try {{ var t = localStorage.getItem(KEY); if (t) root.setAttribute("data-theme", t); }} catch (e) {{}}
  document.getElementById("theme-btn").addEventListener("click", function () {{
    var dark = root.getAttribute("data-theme") ? root.getAttribute("data-theme") === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    var next = dark ? "light" : "dark"; root.setAttribute("data-theme", next);
    try {{ localStorage.setItem(KEY, next); }} catch (e) {{}}
  }});
  document.querySelectorAll(".codebox .copy").forEach(function (b) {{
    b.addEventListener("click", function () {{
      var code = b.closest(".codebox").querySelector("code").innerText;
      (navigator.clipboard ? navigator.clipboard.writeText(code) : Promise.reject()).then(function () {{
        b.textContent = "已复制"; setTimeout(function () {{ b.textContent = "复制"; }}, 1200);
      }}, function () {{ b.textContent = "复制失败"; }});
    }});
  }});
  if (window.hljs) document.querySelectorAll("pre code").forEach(function (el) {{
    if (!el.className) el.classList.add("language-plaintext");
    try {{ hljs.highlightElement(el); }} catch (e) {{}}
  }});
}})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in SRC_DIR.glob("*.md"))
    for n in names:
        out = page(n)
        left = re.findall(r"\[\[(?:here|llvm|" + "|".join(B.SRC.pins()) + r"):[^\]]{0,60}", out)
        if left:
            B.LINK_ERRORS.extend(f"{n}: unexpanded link (brackets in label?): {x}" for x in left)
        if B.LINK_ERRORS:
            raise SystemExit("unresolved source links:\n  " + "\n  ".join(sorted(set(B.LINK_ERRORS))))
        (OUT_DIR / f"{n}.html").write_text(out, encoding="utf-8")
        print("built", OUT_DIR / f"{n}.html")
    allnames = sorted(p.stem for p in SRC_DIR.glob("*.md"))
    (OUT_DIR / "index.html").write_text(index_page(allnames), encoding="utf-8")
    print("built", OUT_DIR / "index.html")
