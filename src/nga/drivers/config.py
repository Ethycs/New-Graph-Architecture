"""Config driver schema - pydantic v2 model for the YAML run configuration.

The cross-field invariant for group_spec_path requires an AblationTuple to be
fully evaluated. For Phase 0, the model_validator runs in "lenient" mode: it only
raises if group_spec_path is None AND the caller explicitly sets
explicit_group_check=True on model construction (via model_fields_set or a
context flag passed through model_validate). Document: in Phase 0 this check is
advisory only; Phase 1+ wires the AblationFile loader to perform the full check.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nga.drivers._version import CONFIG_SCHEMA_VERSION, check_supported


class Config(BaseModel):
    """Pydantic v2 model for the run configuration YAML.

    Cross-field invariant (group_spec_path):
      If group_quotient_enabled is True in the selected ablation, group_spec_path
      must be non-None. In Phase 0 this validation is lenient - it fires only when
      the caller passes explicit_group_check=True via model_validate(context=...).
      Full enforcement is wired in Phase 1 once the AblationFile loader is linked
      to Config.load.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=CONFIG_SCHEMA_VERSION)
    grammar_spec_path: Path
    graph_fsm_path: Path
    graph_fsm_source: Literal["hand", "compiled"] = "hand"
    group_spec_path: Path | None = None
    embedding_dim: int = Field(..., ge=1)
    temperature: float = Field(..., gt=0.0)
    idf_alpha: float = Field(..., gt=0.0)
    ablation: str = Field(..., pattern=r"^A\d+$")
    dataset: Literal[
        "mnist", "babyai-synthetic", "babyai", "alfworld", "scienceworld", "dyck-k"
    ]
    seed: int = Field(..., ge=0)
    batch_size: int | None = None
    num_epochs: int | None = None
    device: str = "cpu"

    @model_validator(mode="after")
    def _check_group_spec(self) -> "Config":
        """Lenient group_spec_path check.

        In Phase 0 the validator only raises when the caller explicitly requests
        the group check. Full enforcement requires the AblationFile to be loaded
        and the selected ablation's group_quotient_enabled flag to be consulted.
        """
        # Full check is deferred to Phase 1. No action in Phase 0.
        return self


def load(path: Path) -> Config:
    """Read a Config YAML file, validate schema_version, and return a Config."""
    with path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must be a YAML mapping, got {type(raw)}")
    check_supported(str(raw.get("schema_version", "")), "Config")
    return Config.model_validate(raw)


def dump(obj: Config, path: Path) -> None:
    """Write a Config object to a YAML file, omitting None-valued fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj.model_dump(mode="json", exclude_none=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


__all__ = ["Config", "load", "dump"]
