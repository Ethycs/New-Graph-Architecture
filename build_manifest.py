#!/usr/bin/env python3
"""Build a manifest of all zettels for the Sonnet link pass."""
import json
import re
from pathlib import Path

D = Path(__file__).parent / "docs"
clusters = ["arch", "exp", "drivers", "open"]

manifest = {}
for c in clusters:
    folder = D / c
    items = []
    for p in sorted(folder.glob("*.md")):
        text = p.read_text()
        # extract title (first H1)
        m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = m.group(1).strip() if m else p.stem
        # extract first paragraph of `## What`
        wm = re.search(r"##\s+What\s*\n+(.+?)(?:\n##|\Z)", text, re.DOTALL)
        what = " ".join((wm.group(1) if wm else "").split())[:200]
        items.append({"file": p.name, "title": title, "what": what})
    manifest[c] = items

out = D / "_manifest.json"
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"wrote {out} with {sum(len(v) for v in manifest.values())} entries")
for c, items in manifest.items():
    print(f"  {c}: {len(items)} entries")
