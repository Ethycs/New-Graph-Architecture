"""Ablation Flags driver schema - pydantic v2 models for configs/ablations.yaml.

AblationTuple holds the eight boolean feature flags for one ablation variant.
AblationFile wraps the full ablations.yaml file (schema_version + ablations dict).

Functions:
  load(path)          - read YAML, validate, return AblationFile
  get(file, id)       - look up one AblationTuple by id (e.g. "A0")
  assert_one_off_from_a0(file) - structural integrity check for the standard matrix
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nga.drivers._version import ABLATION_FLAGS_SCHEMA_VERSION, check_supported


class AblationTuple(BaseModel):
    """Eight boolean flags controlling which architectural components are active.

    A0 (the full-system baseline) has all flags set to True.
    A1-A9 each flip exactly one flag relative to A0 (except A4, which keeps all
    flags identical to A0 - its difference from A0 is at the singularity-detector
    consumer level, not in the flag tuple itself).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    graph_mask_enabled: bool
    typed_scores_enabled: bool
    singularity_detector_enabled: bool
    idf_weighting_enabled: bool
    hyperbolic_geometry_enabled: bool
    group_quotient_enabled: bool
    reservoir_frozen: bool
    trace_history_enabled: bool

    def _bool_flags(self) -> dict[str, bool]:
        """Return only the boolean flag fields as a dict."""
        return {
            "graph_mask_enabled": self.graph_mask_enabled,
            "typed_scores_enabled": self.typed_scores_enabled,
            "singularity_detector_enabled": self.singularity_detector_enabled,
            "idf_weighting_enabled": self.idf_weighting_enabled,
            "hyperbolic_geometry_enabled": self.hyperbolic_geometry_enabled,
            "group_quotient_enabled": self.group_quotient_enabled,
            "reservoir_frozen": self.reservoir_frozen,
            "trace_history_enabled": self.trace_history_enabled,
        }


class AblationFile(BaseModel):
    """Top-level model for configs/ablations.yaml."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=ABLATION_FLAGS_SCHEMA_VERSION)
    ablations: dict[str, AblationTuple]

    @field_validator("ablations", mode="after")
    @classmethod
    def _keys_must_match_pattern(
        cls, v: dict[str, AblationTuple]
    ) -> dict[str, AblationTuple]:
        """All ablation keys must match ^A\\d+$."""
        bad = [k for k in v if not re.match(r"^A\d+$", k)]
        if bad:
            raise ValueError(
                f"Ablation keys must match ^A\\d+$; invalid keys: {bad}"
            )
        return v


def load(path: Path) -> AblationFile:
    """Read an AblationFile YAML, validate schema_version, and return the model."""
    with path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"AblationFile {path} must be a YAML mapping, got {type(raw)}"
        )
    check_supported(str(raw.get("schema_version", "")), "AblationFile")
    return AblationFile.model_validate(raw)


def get(file: AblationFile, ablation_id: str) -> AblationTuple:
    """Return the AblationTuple for ablation_id, or raise KeyError on miss."""
    if ablation_id not in file.ablations:
        raise KeyError(
            f"Ablation '{ablation_id}' not found in file. "
            f"Available: {sorted(file.ablations)}"
        )
    return file.ablations[ablation_id]


def assert_one_off_from_a0(file: AblationFile) -> None:
    """Verify the structural integrity of the standard A0-A9 ablation matrix.

    Rules:
      - A0 must exist and have all eight boolean flags set to True.
      - A1, A2, A3, A5, A6, A7, A8, A9 must each differ from A0 on exactly one
        boolean flag (i.e. exactly one flag is False where A0 has True).
      - A4 is intentionally equal to A0 in flag values. The difference between A3
        and A4 is at the singularity-detector consumer level (A4 computes sigma(x)
        but ignores it in control routing), not in the flag tuple itself. A4 is
        therefore exempted from the one-off requirement.
      - Ablations beyond A9 (if present) are not checked.

    Raises ValueError if any invariant is violated.
    """
    if "A0" not in file.ablations:
        raise ValueError("A0 must be present in the ablation file")

    a0 = file.ablations["A0"]
    a0_flags = a0._bool_flags()

    # A0 must have all flags True
    false_in_a0 = [k for k, v in a0_flags.items() if not v]
    if false_in_a0:
        raise ValueError(
            f"A0 must have all boolean flags True; these are False: {false_in_a0}"
        )

    # Standard one-off ablations (A4 is exempt - same flags as A0 by design)
    one_off_ids = ["A1", "A2", "A3", "A5", "A6", "A7", "A8", "A9"]

    for aid in one_off_ids:
        if aid not in file.ablations:
            # Non-fatal: only validate ablations that exist
            continue
        at = file.ablations[aid]
        flags = at._bool_flags()
        diffs = [k for k in a0_flags if a0_flags[k] != flags[k]]
        if len(diffs) != 1:
            raise ValueError(
                f"{aid} must differ from A0 on exactly one boolean flag, "
                f"but differs on {len(diffs)}: {diffs}"
            )

    # A4 must have the same flags as A0
    if "A4" in file.ablations:
        a4_flags = file.ablations["A4"]._bool_flags()
        diffs = [k for k in a0_flags if a0_flags[k] != a4_flags[k]]
        if diffs:
            raise ValueError(
                f"A4 must have the same boolean flags as A0 (its difference from "
                f"A0 is at the consumer level, not the flag level). "
                f"Flag differences found: {diffs}"
            )


__all__ = ["AblationTuple", "AblationFile", "load", "get", "assert_one_off_from_a0"]
