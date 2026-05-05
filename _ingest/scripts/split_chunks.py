#!/usr/bin/env python3
"""Pre-split chunks.jsonl into per-batch files so each Haiku agent reads a small file."""
import json
from pathlib import Path

D = Path(__file__).parent
batches = [
    (0, range(0, 10)),
    (1, range(10, 20)),
    (2, range(20, 30)),
    (3, range(30, 40)),
    (4, range(40, 50)),
    (5, range(50, 60)),
    (6, range(60, 72)),
]

with (D / "chunks.jsonl").open() as f:
    chunks = {json.loads(l)["id"]: json.loads(l) for l in f}
with (D / "chunks.jsonl").open() as f:
    rows = [json.loads(l) for l in f]
chunks = {r["id"]: r for r in rows}

for n, rng in batches:
    out = D / f"chunks_input_batch_{n}.jsonl"
    with out.open("w") as f:
        for cid in rng:
            f.write(json.dumps(chunks[cid], ensure_ascii=False) + "\n")
    print(f"  {out.name}: {out.stat().st_size:,} bytes, {len(list(rng))} records")
