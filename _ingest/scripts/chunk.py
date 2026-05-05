#!/usr/bin/env python3
"""Stage 0: split chat-export into ~500-word chunks on paragraph boundaries.

Output: chunks.jsonl, one record per line:
    {"id": int, "line_start": int, "line_end": int, "word_count": int, "text": str}
"""
import json
import re
from pathlib import Path

SRC = Path(__file__).parent / "chat-export-2026-05-05T04-35-07.md"
OUT = Path(__file__).parent / "chunks.jsonl"

TARGET = 500
TOLERANCE = 100  # snap to a paragraph break within ±TOLERANCE words

def word_count(s: str) -> int:
    return len(re.findall(r"\S+", s))

def main() -> None:
    raw = SRC.read_text()
    lines = raw.splitlines(keepends=True)

    # Build paragraph-aware index: list of (line_start, line_end_exclusive, text, wc)
    paragraphs: list[tuple[int, int, str, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        # Skip leading blanks
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break
        start = i
        while i < n and lines[i].strip() != "":
            i += 1
        end = i  # exclusive
        text = "".join(lines[start:end])
        paragraphs.append((start, end, text, word_count(text)))

    # Greedy pack paragraphs into chunks targeting 500 words.
    chunks = []
    cur_paras: list[tuple[int, int, str, int]] = []
    cur_words = 0
    cid = 0
    for p in paragraphs:
        s, e, t, wc = p
        if cur_words + wc <= TARGET + TOLERANCE or not cur_paras:
            cur_paras.append(p)
            cur_words += wc
            if cur_words >= TARGET - TOLERANCE and cur_words >= TARGET:
                # close chunk
                line_start = cur_paras[0][0] + 1  # 1-indexed
                line_end = cur_paras[-1][1]       # exclusive in 0-index = inclusive in 1-index
                text = "".join(p[2] for p in cur_paras)
                chunks.append({
                    "id": cid,
                    "line_start": line_start,
                    "line_end": line_end,
                    "word_count": cur_words,
                    "text": text,
                })
                cid += 1
                cur_paras = []
                cur_words = 0
        else:
            # close current and start new with this paragraph
            line_start = cur_paras[0][0] + 1
            line_end = cur_paras[-1][1]
            text = "".join(p[2] for p in cur_paras)
            chunks.append({
                "id": cid,
                "line_start": line_start,
                "line_end": line_end,
                "word_count": cur_words,
                "text": text,
            })
            cid += 1
            cur_paras = [p]
            cur_words = wc

    if cur_paras:
        line_start = cur_paras[0][0] + 1
        line_end = cur_paras[-1][1]
        text = "".join(p[2] for p in cur_paras)
        chunks.append({
            "id": cid,
            "line_start": line_start,
            "line_end": line_end,
            "word_count": cur_words,
            "text": text,
        })

    with OUT.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    total_words = sum(c["word_count"] for c in chunks)
    print(f"chunks: {len(chunks)}")
    print(f"total words: {total_words}")
    print(f"min/median/max words: "
          f"{min(c['word_count'] for c in chunks)}/"
          f"{sorted(c['word_count'] for c in chunks)[len(chunks)//2]}/"
          f"{max(c['word_count'] for c in chunks)}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
