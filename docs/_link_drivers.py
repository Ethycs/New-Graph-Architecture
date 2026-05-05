#!/usr/bin/env python3
"""Convert prose mentions of driver filenames into clickable Markdown links in arch/ and exp/."""
import re
from pathlib import Path

D = Path(__file__).parent

# Patterns: (regex, replacement). Operates outside code fences only.
# Use negative lookbehind to avoid double-linking already-linked items.
DRIVER_LINKS = [
    # Token : driver path (relative from arch/ or exp/ — same up-one)
    (r"`metrics\.jsonl`", "[`metrics.jsonl`](../drivers/metrics-jsonl.md)"),
    (r"`results\.jsonl`", "[`results.jsonl`](../drivers/results-jsonl.md)"),
    (r"`config\.yaml`", "[`config.yaml`](../drivers/config.md)"),
    (r"`graph_fsm\.json`", "[`graph_fsm.json`](../drivers/graph-fsm-spec.md)"),
    (r"`graph[-_]fsm[-_]spec`", "[graph-fsm-spec](../drivers/graph-fsm-spec.md)"),
    (r"`ablation[-_]flags?`", "[ablation flags](../drivers/ablation-flags.md)"),
    (r"\bpython run\.py\b", "[`python run.py`](../drivers/cli-runner.md)"),
    (r"\bCLI runner\b", "[CLI Runner](../drivers/cli-runner.md)"),
]

CODE_FENCE = re.compile(r"^```")
LINK_RE = re.compile(r"\]\(")


def link_drivers(text: str) -> tuple[str, int]:
    """Walk lines; outside code fences, apply driver-link substitutions.
    Skip lines that already contain a Markdown link target right after the matched token,
    to avoid re-wrapping already-linked references."""
    out = []
    in_fence = False
    n_subs = 0
    for line in text.split("\n"):
        if CODE_FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        new_line = line
        for pat, repl in DRIVER_LINKS:
            # Skip if already a link to drivers/ on this line
            if "drivers/" in new_line and pat.replace("\\b", "").replace("\\.", ".") in new_line:
                # check if the token already appears as a link target
                pass
            # Avoid linking inside an existing markdown link: skip if pattern is part of "(...token...)"
            def replace_once(m):
                nonlocal n_subs
                start = m.start()
                # if the next 4 chars contain "](", we're inside a link; skip
                rest = new_line[m.end():m.end()+5]
                if rest.startswith("](") or "](" in new_line[max(0, start-30):start]:
                    return m.group(0)
                # if already followed by ](, skip
                n_subs += 1
                return repl
            new_line = re.sub(pat, replace_once, new_line)
        out.append(new_line)
    return "\n".join(out), n_subs


total = 0
for folder in ("arch", "exp"):
    for p in (D / folder).glob("*.md"):
        if p.name == "_index.md":
            continue
        text = p.read_text()
        new_text, n = link_drivers(text)
        if n > 0:
            p.write_text(new_text)
            total += n
            print(f"  {p.relative_to(D)}: +{n} driver links")
print(f"\nTotal driver links inserted: {total}")
