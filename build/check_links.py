"""Check every external http(s) link in content/*.md (markdown links) responds. Usage: python3 build/check_links.py"""
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

C = Path(__file__).resolve().parent.parent / "content"
urls = sorted({u for f in C.glob("*.md") for u in re.findall(r"\]\((https?://[^)\s]+)\)", f.read_text())})


def status(u):
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-L", "--max-time", "25", "-A", "Mozilla/5.0 (link checker)",
                        "-w", "%{http_code}", u], capture_output=True, text=True)
    return u, r.stdout.strip()


bad = 0
with ThreadPoolExecutor(12) as ex:
    for u, code in ex.map(status, urls):
        ok = code == "200"
        bad += not ok
        print(("ok  " if ok else "BAD ") + code + "  " + u)
print(f"{len(urls)} external links, {bad} not 200")
sys.exit(1 if bad else 0)
