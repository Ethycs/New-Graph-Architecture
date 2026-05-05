#!/usr/bin/env python3
"""Stage 2 post-processing: merge per-piece sections.json into a single sections.json
with primary/cross-ref split resolved, then write per-piece input files for Stage 3 (Opus).

Rules:
- Each chunk lives verbatim in exactly one piece — its `primary` from highlights.jsonl.
- If Sonnet's section list for piece X contains a chunk whose primary != X, that becomes
  a cross-reference (to be rendered as a one-line pointer in piece X's final file).
- secondary tags (from Stage 1) are an additional source of cross-references — but only
  if the chunk isn't already in piece X's section list.
"""
import json
from pathlib import Path

D = Path(__file__).parent

with (D / "highlights.jsonl").open() as f:
    highlights = [json.loads(l) for l in f]

primary_of = {r["chunk_id"]: r["primary"] for r in highlights}
secondary_of = {r["chunk_id"]: r.get("secondary") for r in highlights}
highlight_of = {r["chunk_id"]: r["highlight"] for r in highlights}

PIECES = ["Mathematics", "Architecture", "Experiments"]

per_piece = {}
for piece in PIECES:
    sec = json.loads((D / f"sections_{piece}.json").read_text())
    per_piece[piece] = sec

# For each piece, split chunk_ids in each section into "owned" (primary == piece)
# and "xref" (primary != piece, but Sonnet placed it here). Also gather extra xrefs
# from secondary tags pointing at this piece that aren't already in any section.
resolved = {}
for piece in PIECES:
    sections = []
    placed_xref_ids = set()
    for sec in per_piece[piece]["sections"]:
        owned = [cid for cid in sec["chunk_ids"] if primary_of[cid] == piece]
        xrefs = [cid for cid in sec["chunk_ids"] if primary_of[cid] != piece]
        placed_xref_ids.update(xrefs)
        sections.append({
            "section_name": sec["section_name"],
            "rationale": sec["rationale"],
            "owned_chunk_ids": owned,
            "xref_chunk_ids": [
                {
                    "chunk_id": cid,
                    "lives_in": primary_of[cid],
                    "highlight": highlight_of[cid],
                }
                for cid in xrefs
            ],
        })
    # Extra xrefs: chunks with secondary == piece, primary != piece, not already placed
    extra_xrefs = []
    for cid, sec_tag in secondary_of.items():
        if sec_tag == piece and primary_of[cid] != piece and cid not in placed_xref_ids:
            extra_xrefs.append({
                "chunk_id": cid,
                "lives_in": primary_of[cid],
                "highlight": highlight_of[cid],
            })
    resolved[piece] = {
        "piece": piece,
        "sections": sections,
        "extra_xrefs": extra_xrefs,  # rendered as a "Cross-references" appendix
    }

(D / "sections.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False))

# Sanity print
print("Per-piece resolution:")
for piece in PIECES:
    r = resolved[piece]
    owned_total = sum(len(s["owned_chunk_ids"]) for s in r["sections"])
    inline_xref = sum(len(s["xref_chunk_ids"]) for s in r["sections"])
    print(f"  {piece}: {len(r['sections'])} sections, "
          f"{owned_total} owned chunks, "
          f"{inline_xref} inline xrefs, "
          f"{len(r['extra_xrefs'])} appendix xrefs")

# Cross-check: every chunk owned exactly once across all pieces
owned_everywhere = []
for piece in PIECES:
    for s in resolved[piece]["sections"]:
        owned_everywhere.extend(s["owned_chunk_ids"])
print(f"\nTotal owned across all pieces: {len(owned_everywhere)} (should be 72)")
print(f"Unique: {len(set(owned_everywhere))}")
missing = set(range(72)) - set(owned_everywhere)
print(f"Missing chunk_ids: {sorted(missing) if missing else 'none'}")

# Write per-piece input bundles for Opus, including chunk text for owned chunks
with (D / "chunks.jsonl").open() as f:
    chunks = {json.loads(l)["id"]: json.loads(l) for l in f.readlines()}
# Reload because dict comprehension above consumes iterator twice
with (D / "chunks.jsonl").open() as f:
    chunks_list = [json.loads(l) for l in f]
chunks = {c["id"]: c for c in chunks_list}

for piece in PIECES:
    bundle = {
        "piece": piece,
        "sections": [],
        "extra_xrefs": resolved[piece]["extra_xrefs"],
    }
    for sec in resolved[piece]["sections"]:
        bundle_sec = {
            "section_name": sec["section_name"],
            "rationale": sec["rationale"],
            "xref_chunk_ids": sec["xref_chunk_ids"],
            "owned_chunks": [
                {
                    "chunk_id": cid,
                    "line_start": chunks[cid]["line_start"],
                    "line_end": chunks[cid]["line_end"],
                    "text": chunks[cid]["text"],
                }
                for cid in sec["owned_chunk_ids"]
            ],
        }
        bundle["sections"].append(bundle_sec)
    out = D / f"opus_input_{piece}.json"
    out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    print(f"  wrote {out.name} ({out.stat().st_size:,} bytes)")
