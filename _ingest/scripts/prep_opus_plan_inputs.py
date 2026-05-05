#!/usr/bin/env python3
"""Build compact per-piece input files for Opus subsection-planning agents."""
import json
from pathlib import Path

D = Path(__file__).parent

with (D / "highlights.jsonl").open() as f:
    highlights = {json.loads(l)["chunk_id"]: json.loads(l) for l in f}
# Re-read because dict above consumed iterator partially? Actually generator + dict comp uses one pass per call. Use explicit.
with (D / "highlights.jsonl").open() as f:
    rows = [json.loads(l) for l in f]
highlights = {r["chunk_id"]: r for r in rows}

sections = json.loads((D / "sections.json").read_text())

for piece, bundle in sections.items():
    plan_input = {"piece": piece, "sections": []}
    for sec in bundle["sections"]:
        plan_input["sections"].append({
            "section_name": sec["section_name"],
            "rationale": sec["rationale"],
            "owned_chunks": [
                {"chunk_id": cid, "highlight": highlights[cid]["highlight"]}
                for cid in sec["owned_chunk_ids"]
            ],
        })
    out = D / f"plan_input_{piece}.json"
    out.write_text(json.dumps(plan_input, indent=2, ensure_ascii=False))
    print(f"  {out.name}: {out.stat().st_size:,} bytes, "
          f"{len(plan_input['sections'])} sections, "
          f"{sum(len(s['owned_chunks']) for s in plan_input['sections'])} chunks")
