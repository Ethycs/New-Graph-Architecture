"""Typed Score Record driver schema - pydantic v2 model for the JSON-lines wire format.

Each record binds a predicted type label with confidence, a probability distribution
array over all vertices, optional hyperbolic coordinates, and provenance fields
(experiment, ablation, seed, step). Architecture emits one record per inference
sample; experiments consume these records to evaluate margins, singularities, and
transition legality.

JSON-line semantics: use to_json() to serialise a single record and from_json() to
parse one line from scores.jsonl.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nga.drivers._version import TYPED_SCORE_RECORD_SCHEMA_VERSION, check_supported


class TypedScoreRecord(BaseModel):
    """Pydantic v2 model for a single typed score record.

    dist validator: the distribution must be non-empty and must sum to 1.0 within
    an absolute tolerance of 1e-6. This mirrors the softmax guarantee; any
    pre-normalised distribution from the architecture should satisfy it.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=TYPED_SCORE_RECORD_SCHEMA_VERSION)
    run_id: str
    experiment: str
    ablation: str = Field(..., pattern=r"^A\d+$")
    seed: int
    step: int = Field(..., ge=0)
    sample_id: str
    predicted_state: str
    stratum_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    margin: float = Field(..., ge=-1.0, le=1.0)
    dist: list[float]
    z_H: list[float] | None = None
    embedding: list[float] | None = None
    timestamp: str | None = None

    @field_validator("dist", mode="after")
    @classmethod
    def _dist_must_be_normalised(cls, v: list[float]) -> list[float]:
        """dist must be non-empty and sum to 1.0 within 1e-6."""
        if len(v) == 0:
            raise ValueError("dist must be non-empty")
        total = sum(v)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"dist must sum to 1.0 (within 1e-6), got sum={total}"
            )
        return v

    def to_json(self) -> str:
        """Serialise this record to a JSON string (one JSON-line)."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> "TypedScoreRecord":
        """Parse a JSON string (one JSON-line) into a TypedScoreRecord.

        Validates schema_version before constructing the model.
        """
        import json

        raw: Any = json.loads(s)
        if not isinstance(raw, dict):
            raise ValueError(f"TypedScoreRecord JSON must be a mapping, got {type(raw)}")
        check_supported(str(raw.get("schema_version", "")), "TypedScoreRecord")
        return cls.model_validate(raw)


__all__ = ["TypedScoreRecord"]
