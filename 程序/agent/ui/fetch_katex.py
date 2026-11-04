"""把 KaTeX 拉到本地，保证离线可用。"""
import re
import subprocess
import sys
from pathlib import Path

BASE = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
OUT = Path(__file__).resolve().parent / "vendor" / "katex"
(OUT / "fonts").mkdir(parents=True, exist_ok=True)


def fetch(url, dest):
    proc = subprocess.run(["curl.exe", "-sSL", "--max-time", "60", "-o", str(dest), url],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"failed {url}: {proc.stderr}")
    return dest.stat().st_size


for name in ("katex.min.js", "katex.min.css"):
    size = fetch(f"{BASE}/{name}", OUT / name)
    print(f"{name}: {size} bytes")

css = (OUT / "katex.min.css").read_text(encoding="utf-8")
fonts = sorted(set(re.findall(r"fonts/(KaTeX_[A-Za-z0-9\-]+\.woff2)", css)))
print(f"fonts referenced: {len(fonts)}")
total = 0
for f in fonts:
    try:
        total += fetch(f"{BASE}/fonts/{f}", OUT / "fonts" / f)
    except SystemExit as exc:
        print("  skip", f, exc)
print(f"fonts downloaded: {total} bytes")
print("done ->", OUT)
