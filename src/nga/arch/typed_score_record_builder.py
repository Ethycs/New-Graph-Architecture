"""Typed Score Record builder arch atom.

Packs a TypedFieldOutput (the in-memory result of one pipeline step) into a
TypedScoreRecord (the JSON-lines wire format defined in
drivers/typed_score_record.py).

This is a pure constructor - no IO, no side effects. All field conversions
(numpy -> list[float], timestamp defaulting) happen here so callers never need
to touch the wire-format model directly.

Phase 1 notes:
- z_H is always None (hyperbolic embedding arrives in Phase 3).
- embedding is always None in the wire record for Phase 1 (the in-memory
  TypedFieldOutput.embedding field is reserved for future phases).

See docs/arch/typed-score-record.md for the full spec.
"""

from __future__ import annotations

from datetime import datetime, timezone

from nga.arch.typed_field_pipeline import TypedFieldOutput
from nga.drivers.typed_score_record import TypedScoreRecord

__all__ = ["build_typed_score_record"]


def build_typed_score_record(
    field: TypedFieldOutput,
    *,
    run_id: str,
    experiment: str,
    ablation: str,
    seed: int,
    step: int,
    sample_id: str,
    margin: float,
    z_H: list[float] | None = None,
    timestamp: str | None = None,
) -> TypedScoreRecord:
    """Pack a TypedFieldOutput into the wire-format TypedScoreRecord.

    Parameters
    ----------
    field:
        Output of TypedFieldPipeline.predict_one / predict_batch[i].
    run_id:
        Unique identifier for this run (e.g. "E0_A0_seed42_20260504T120000Z").
    experiment:
        Experiment label (e.g. "E0").
    ablation:
        Ablation label matching ^A\\d+$ (e.g. "A0").
    seed:
        Integer random seed used for this run.
    step:
        Non-negative integer step index within the run.
    sample_id:
        Unique identifier for the observation (e.g. a dataset row id).
    margin:
        Top-1 minus top-2 probability; supplied by the margin-uncertainty atom.
    z_H:
        Hyperbolic embedding coordinates; None until Phase 3.
    timestamp:
        ISO-8601 timestamp string. Defaults to the current UTC time when None.

    Returns
    -------
    TypedScoreRecord
        Fully validated wire-format record ready for serialisation.
    """
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc).isoformat()

    dist_list: list[float] = field.distribution.tolist()

    return TypedScoreRecord(
        run_id=run_id,
        experiment=experiment,
        ablation=ablation,
        seed=seed,
        step=step,
        sample_id=sample_id,
        predicted_state=field.predicted_state,
        stratum_label=field.stratum_label,
        confidence=field.confidence,
        margin=margin,
        dist=dist_list,
        z_H=z_H,
        embedding=None,  # Phase 3 will populate from field.embedding
        timestamp=timestamp,
    )
