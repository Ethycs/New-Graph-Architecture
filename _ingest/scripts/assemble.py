#!/usr/bin/env python3
"""Stage 4: assemble final Markdown files from Opus subsection plans.

Reads:
  plan_<Piece>.json     — subsection plan from Opus
  chunks_clean.jsonl (preferred) or chunks.jsonl
  sections.json         — cross-reference data
"""
import json
import sys
from pathlib import Path

D = Path(__file__).parent

CHUNK_FILE = "chunks_clean.jsonl" if (D / "chunks_clean.jsonl").exists() else "chunks.jsonl"
with (D / CHUNK_FILE).open() as f:
    chunks_list = [json.loads(l) for l in f]
chunks = {c["id"]: c for c in chunks_list}

sections_meta = json.loads((D / "sections.json").read_text())

PIECES = sys.argv[1:] if len(sys.argv) > 1 else ["Mathematics", "Architecture", "Experiments"]
TOTAL_CHUNKS = len(chunks)
print(f"Using chunk source: {CHUNK_FILE}")
print(f"Re-emitting pieces: {PIECES}")


def render_piece(piece: str) -> tuple[str, dict]:
    plan = json.loads((D / f"plan_{piece}.json").read_text())
    meta = sections_meta[piece]

    owned_in_meta = {}  # section_name -> {"xref_chunk_ids": [...]}
    for sec in meta["sections"]:
        owned_in_meta[sec["section_name"]] = sec

    owned_count = sum(
        len(ss["chunk_ids"]) for s in plan["sections"] for ss in s["subsections"]
    )

    lines: list[str] = []
    lines.append(f"# {piece}")
    lines.append("")
    lines.append(
        f"> Source: chat-export-2026-05-05T04-35-07.md, owned chunks of the "
        f"{piece} piece ({owned_count} chunks of {TOTAL_CHUNKS} total)."
    )
    lines.append("")

    seen_chunks = set()
    n_subsections = 0
    for sec in plan["sections"]:
        lines.append(f"## {sec['section_name']}")
        lines.append("")
        if sec.get("preface"):
            lines.append(f"*{sec['preface']}*")
            lines.append("")
        for ss in sec["subsections"]:
            n_subsections += 1
            lines.append(f"### {ss['subsection_name']}")
            lines.append("")
            for cid in ss["chunk_ids"]:
                if cid in seen_chunks:
                    raise RuntimeError(f"chunk {cid} appears twice in plan_{piece}.json")
                seen_chunks.add(cid)
                text = chunks[cid]["text"]
                lines.append(text.rstrip())
                lines.append("")

        # inline xrefs at end of section
        meta_sec = owned_in_meta.get(sec["section_name"])
        if meta_sec and meta_sec.get("xref_chunk_ids"):
            for x in meta_sec["xref_chunk_ids"]:
                lines.append(
                    f"> See also chunk #{x['chunk_id']} in "
                    f"[{x['lives_in']}.md]({x['lives_in']}.md): {x['highlight']}"
                )
            lines.append("")

    # appendix cross-references
    if meta.get("extra_xrefs"):
        lines.append("## Cross-references")
        lines.append("")
        lines.append(
            f"The following chunks live in other pieces but are {piece.lower()}-relevant:"
        )
        lines.append("")
        for x in meta["extra_xrefs"]:
            lines.append(
                f"- chunk #{x['chunk_id']} (in [{x['lives_in']}.md]({x['lives_in']}.md)): "
                f"{x['highlight']}"
            )
        lines.append("")

    md = "\n".join(lines).rstrip() + "\n"
    stats = {
        "sections": len(plan["sections"]),
        "subsections": n_subsections,
        "owned_chunks": len(seen_chunks),
        "inline_xrefs": sum(
            len(s.get("xref_chunk_ids", [])) for s in meta["sections"]
        ),
        "appendix_xrefs": len(meta.get("extra_xrefs", [])),
        "bytes": len(md),
    }
    return md, stats


def main() -> None:
    overall_owned = 0
    for piece in PIECES:
        plan_path = D / f"plan_{piece}.json"
        if not plan_path.exists():
            print(f"  SKIP {piece}: plan_{piece}.json not found")
            continue
        md, stats = render_piece(piece)
        out = D / f"{piece}.md"
        out.write_text(md)
        overall_owned += stats["owned_chunks"]
        print(
            f"  {out.name}: "
            f"sections={stats['sections']}, "
            f"subsections={stats['subsections']}, "
            f"owned_chunks={stats['owned_chunks']}, "
            f"inline_xref={stats['inline_xrefs']}, "
            f"appendix_xref={stats['appendix_xrefs']}, "
            f"size={stats['bytes']:,}B"
        )
    print(f"\nTotal owned chunks across pieces: {overall_owned} (expected {TOTAL_CHUNKS})")


if __name__ == "__main__":
    main()
