# <Atomic concept name>

**Cluster:** arch | exp | driver | open
**Status:** spec | stub | implemented | tested
**Tags:** #example-tag

## What
One paragraph. A single idea. If the note splits cleanly into two ideas, write two notes instead.

## Why
What does this enable? What goes wrong without it? Be concrete.

## Interface
- For **drivers**: the exact contract — fields, types, file format, file path conventions. This is the API both arch and exp depend on.
- For **arch atoms**: inputs (what driver/contract it consumes) → outputs (what it produces, often into another driver).
- For **exp atoms**: what it reads (datasets, configs, prior results) and what metrics it writes.
- For **open**: the question, the constraints that make it unresolved, candidate answers under consideration.

## Build steps
- bullet 1: concrete coding/config task
- bullet 2: …
- bullet 3: …
Not pseudocode. The actual work to do.

## Links
- **See also:** stubbed wikilinks like [[arch.sibling]] — Sonnet will expand these to real relative Markdown links.
- **Drives:** [[exp.experiment-id]] — what experiment exercises this piece.
- **Driven by:** [[drv.driver-id]] — what shared contract this consumes.
- **Math:** [Mathematics.md §section](../Mathematics.md#anchor) — link out, do not duplicate.
- **Open:** [[open.qNN-slug]] — pending decisions that affect this note.
