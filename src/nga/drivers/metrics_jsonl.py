"""MetricsRecord pydantic v2 model and helpers for the metrics.jsonl append-only stream."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from nga.drivers.jsonl_writer import JsonlWriter, read_jsonl


class MetricsRecord(BaseModel):
    """One scalar metric observation written to metrics.jsonl."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    run_id: str
    experiment: str
    ablation: str
    seed: int
    step: int
    split: Literal["train", "val", "test", "all"]
    metric_name: str
    value: float
    timestamp: str | None = None


def open_metrics_writer(path: Path) -> JsonlWriter[MetricsRecord]:
    """Open an append-only JSONL writer for MetricsRecord objects."""
    return JsonlWriter(path, MetricsRecord)


def read_metrics(path: Path) -> list[MetricsRecord]:
    """Read all MetricsRecord lines from a metrics.jsonl file."""
    return read_jsonl(path, MetricsRecord)


__all__ = ["MetricsRecord", "open_metrics_writer", "read_metrics"]
