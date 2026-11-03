"""Pinned source links for the drill pages.

Content syntax (inside Markdown, outside code):
    [[flashinfer:include/flashinfer/norm.cuh]]                 file link, label = file name
    [[flashinfer:include/flashinfer/norm.cuh@RMSNorm]]         jump to the line defining RMSNorm
    [[flashinfer:include/flashinfer/norm.cuh@RMSNorm|label]]   custom label
    [[flashinfer:include/flashinfer/attention/]]               directory (tree) link
    [[flashinfer]]                                             repository root at the pinned commit

Every link points at a fixed commit (build/sources/pins.txt), so line numbers never drift.

    python3 build/sources.py fetch    # (needs `gh`) download every referenced file into build/sources/files/
    python3 build/sources.py check    # verify paths + symbols resolve (offline, from the cache)
build.py calls render() and fails if anything does not resolve.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "sources"
CONTENT = Path(__file__).resolve().parent.parent / "content"
MACRO = re.compile(r"\[\[([a-z0-9]+)(?::([^\]|@#]*))?(?:@([^\]|]+))?(?:\|([^\]]+))?\]\]")


def pins():
    out = {}
    for line in (HERE / "pins.txt").read_text().split("\n"):
        if line.strip():
            alias, repo, sha = line.split()
            out[alias] = (repo, sha)
    return out


def tree(alias):
    return set((HERE / f"{alias}.tree").read_text().split("\n"))


def cached(alias, path):
    return HERE / "files" / alias / path


def find_line(text, symbol):
    """1-based line of the most definition-like occurrence of `symbol` (or None)."""
    sym = re.escape(symbol)
    lines = text.split("\n")
    patterns = [
        rf"^\s*(async\s+)?def\s+{sym}\b", rf"^\s*class\s+{sym}\b", rf"^\s*struct\s+{sym}\b",
        rf"\b(__global__|__device__|cudaError_t|kernel)\b.*\b{sym}\s*[(<]",
        rf"^\s*(template\s*<.*>\s*)?([\w:<>,\*&\s]+\s)?{sym}\s*\(",          # C/C++ function definition
        rf"^\s*{sym}\s*=", rf"^#+\s.*{sym}",                               # assignment / markdown heading
        rf"\b{sym}\b",
    ]
    for pat in patterns:
        for i, ln in enumerate(lines):
            if re.search(pat, ln):
                return i + 1
    return None


def resolve(alias, path, symbol):
    """-> (url, error)"""
    P = pins()
    if alias not in P:
        return None, f"unknown repo alias {alias!r}"
    repo, sha = P[alias]
    if not path:
        return f"https://github.com/{repo}/tree/{sha}", None
    if path.endswith("/"):
        if not any(p.startswith(path) for p in tree(alias)):
            return None, f"{alias}: no directory {path}"
        return f"https://github.com/{repo}/tree/{sha}/{path.rstrip('/')}", None
    if path not in tree(alias):
        return None, f"{alias}: no file {path}"
    url = f"https://github.com/{repo}/blob/{sha}/{path}"
    if symbol:
        f = cached(alias, path)
        if not f.exists():
            return None, f"{alias}:{path} not fetched (run: python3 build/sources.py fetch)"
        line = find_line(f.read_text(errors="replace"), symbol)
        if line is None:
            return None, f"{alias}:{path}: symbol {symbol!r} not found"
        url += ("?plain=1" if path.endswith((".md", ".rst", ".ipynb")) else "") + f"#L{line}"
    return url, None


def _protect(text):
    """Do not expand macros inside fenced code or inline code spans."""
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    return parts


def render(text, errors):
    out = []
    for i, part in enumerate(_protect(text)):
        if i % 2:                     # code: leave untouched
            out.append(part)
            continue

        def sub(m):
            alias, path, symbol, label = m.group(1), (m.group(2) or "").strip(), m.group(3), m.group(4)
            url, err = resolve(alias, path, symbol)
            if err:
                errors.append(err)
                return m.group(0)
            repo = pins()[alias][0]
            name = label or (symbol if symbol and not path.endswith("/") else None) or \
                (path.rstrip("/").split("/")[-1] + ("/" if path.endswith("/") else "") if path else repo)
            tip = f"{repo} @ {pins()[alias][1][:7]}" + (f" · {path}" if path else "")
            return (f'<a class="src" href="{url}" title="{tip}" target="_blank" rel="noopener">'
                    f'<span class="src-repo">{alias}</span>{name}</a>')
        out.append(MACRO.sub(sub, part))
    return "".join(out)


def referenced():
    refs = set()
    for f in CONTENT.glob("*.md"):
        for i, part in enumerate(_protect(f.read_text())):
            if i % 2 == 0:
                for m in MACRO.finditer(part):
                    if m.group(2) and m.group(3) and not m.group(2).endswith("/"):
                        refs.add((m.group(1), m.group(2).strip()))
    return refs


def fetch():
    P = pins()
    for alias, path in sorted(referenced()):
        f = cached(alias, path)
        if f.exists():
            continue
        repo, sha = P[alias]
        f.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["gh", "api", f"repos/{repo}/contents/{path}?ref={sha}",
                            "-H", "Accept: application/vnd.github.raw"], capture_output=True)
        if r.returncode:
            print("FAIL", alias, path, r.stderr.decode()[:200])
            continue
        f.write_bytes(r.stdout)
        print("fetched", alias, path)


def check():
    errors = []
    n = 0
    for f in sorted(CONTENT.glob("*.md")):
        before = len(errors)
        text = f.read_text()
        n += sum(len(MACRO.findall(p)) for i, p in enumerate(_protect(text)) if i % 2 == 0)
        render(text, errors)
        for e in errors[before:]:
            print(f"{f.name}: {e}")
    print(f"{n} links, {len(errors)} errors")
    return not errors


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "fetch":
        fetch()
    sys.exit(0 if check() else 1)
