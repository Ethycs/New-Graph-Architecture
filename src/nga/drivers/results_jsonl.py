"""ResultsRecord pydantic v2 model and helpers for the results.jsonl append-only stream.

Schema changelog
----------------
1.0 (initial): base fields through singular_flag + behavioral_stratum/stratum_bitmask.
1.0 (Phase 2, additive): sigma_score optional field added after singular_flag.
  Adding an optional field is non-breaking per the project versioning policy;
  schema_version remains "1.0".
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from nga.drivers.jsonl_writer import JsonlWriter, read_jsonl


class ResultsRecord(BaseModel):
    """One per-sample prediction record written to results.jsonl."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    run_id: str
    experiment: str
    ablation: str
    seed: int
    step: int
    sample_id: str
    y_true: str
    y_hat: str
    margin: float = Field(..., ge=-1.0, le=1.0)
    singular_flag: bool = False
    sigma_score: float | None = Field(None, ge=0.0, le=1.0)
    behavioral_stratum: str | None = None
    stratum_bitmask: int | None = None
    transition_legal: bool | None = None
    cost: float | None = None
    timestamp: str | None = None


def open_results_writer(path: Path) -> JsonlWriter[ResultsRecord]:
    """Open an append-only JSONL writer for ResultsRecord objects."""
    return JsonlWriter(path, ResultsRecord)


def read_results(path: Path) -> list[ResultsRecord]:
    """Read all ResultsRecord lines from a results.jsonl file."""
    return read_jsonl(path, ResultsRecord)


__all__ = ["ResultsRecord", "open_results_writer", "read_results"]
