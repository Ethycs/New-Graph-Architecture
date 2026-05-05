#!/usr/bin/env python3
"""Merge cleaned batch JSONLs (skipping malformed lines) + recovery → chunks_clean.jsonl.
Fall back to original chunk text for any chunk still missing."""
import json
from pathlib import Path

D = Path(__file__).parent

with (D / "chunks.jsonl").open() as f:
    rows = [json.loads(l) for l in f]
orig = {r["id"]: r for r in rows}

cleaned: dict[int, dict] = {}

# Read all batch files, skipping malformed lines
for n in range(7):
    p = D / f"chunks_clean_batch_{n}.jsonl"
    if not p.exists():
        continue
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                cleaned[r["id"]] = r
            except json.JSONDecodeError:
                pass  # malformed line; recovery batch will fill in

# Recovery batch (overrides any successful clean for the recovered IDs)
rec_path = D / "chunks_clean_recovery.jsonl"
if rec_path.exists():
    with rec_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                cleaned[r["id"]] = r
            except json.JSONDecodeError:
                pass

# Fallback: use original for any chunk not cleaned
filled_with_original = []
for cid in range(72):
    if cid not in cleaned:
        cleaned[cid] = orig[cid]
        filled_with_original.append(cid)

# Write final, sorted by id
out = D / "chunks_clean.jsonl"
with out.open("w") as f:
    for cid in sorted(cleaned):
        f.write(json.dumps(cleaned[cid], ensure_ascii=False) + "\n")

print(f"chunks_clean.jsonl: {len(cleaned)} records (expected 72)")
print(f"  fell back to original for: {filled_with_original}")
print(f"  size: {out.stat().st_size:,} bytes")
