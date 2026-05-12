# Axis Quantizer

**Cluster:** arch
**Status:** implemented
**Tags:** #quantizer #typed-axis #no-raw-floats #audit

## What

A deterministic projection $\mathbb{R} \to \text{vertex_set}$ that maps a continuous measurement (σ, energy, margin, or any scalar signal) to a typed-axis node ID. Each axis is a typed graph with a totally ordered vertex set; the quantiser defines the bin boundaries that determine which vertex a measurement falls into. Same input ⇒ same node ID, by construction.

## Why

The TPN commitment "no raw floats cross interfaces" requires every continuous signal to be discretised into a typed-axis vertex before it enters the decision trace, the σ ensemble, or downstream routing. Without an axis quantiser, σ scores would be raw floats in the audit trail, and the trace would no longer be a tuple of typed nodes — it would be a mixture of types and reals, defeating the audit-by-construction guarantee. The quantiser is the gate that keeps the typed-axis discipline intact.

## Interface

- **Constructor:** `AxisQuantizer(vertex_ids, edges)` — pass the ordered vertex set and the bin-boundary thresholds (one threshold per pair of adjacent vertices).
- **Method:** `quantize(x: float) -> str` — returns the vertex ID whose half-open bin contains $x$. Out-of-range inputs clip to the leftmost / rightmost vertex.
- **Determinism:** the projection is a step function with stable bin edges; reproducible across runs and across machines.

## Build steps

- Validate that the vertex set is totally ordered (the bin-boundary list must have length `len(vertex_ids) - 1`).
- `quantize(x)`: binary-search the threshold list for the bin containing $x$; clip to the endpoints if out of range.
- Provide a `bins` property exposing the half-open bin intervals for documentation and trace records.
- Provide a `vertex_to_centroid` map (optional) for inverse-quantisation needed by the σ ensemble.

## Links

- **See also:** [Singularity Detector](./singularity-detector.md), [Decision Trace JSONL](../drivers/decision-trace-jsonl.md), [Graph FSM](./graph-fsm.md).
- **Drives:** every runner that emits a decision trace with σ axis vertices; the cross-attention proposal requires axis-quantisation to re-typify continuous attention outputs.
- **Driven by:** the typed-axis spec (which defines vertex IDs and bin-boundary thresholds).
- **Math:** a deterministic step function $\mathbb{R} \to V$; the discretisation is a Borel-measurable retract of $\mathbb{R}$ onto a finite set.
- **Open:** [[open.qNN-quantizer-bin-choice]] — how to choose bin boundaries: equal-width, equal-frequency on a held-out sample, or hand-authored from domain knowledge.
