"""Generic append-only JSONL writer and reader for pydantic v2 records."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonlWriter(Generic[T]):
    """Append-only JSONL writer for pydantic v2 records.

    - Validates each record before writing (writer is the last line of defense
      against schema drift).
    - One record per line, JSON-serialized via model_dump_json().
    - Optionally fsync after each write (set in constructor).
    - Tracks count of records written for diagnostic purposes.
    """

    def __init__(self, path: Path, model_cls: type[T], fsync: bool = False) -> None:
        self._path = path
        self._model_cls = model_cls
        self._fsync = fsync
        self._count = 0
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def append(self, record: T) -> None:
        """Validate and append one record as a JSON line."""
        # Re-validate via round-trip to catch any drift between construction and write.
        validated = self._model_cls.model_validate(record.model_dump())
        self._file.write(validated.model_dump_json() + "\n")
        if self._fsync:
            self._file.flush()
            os.fsync(self._file.fileno())
        self._count += 1

    def append_many(self, records: Iterable[T]) -> None:
        """Validate and append multiple records."""
        for record in records:
            self.append(record)

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def count(self) -> int:
        """Number of records successfully written in this session."""
        return self._count

    def __enter__(self) -> "JsonlWriter[T]":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()


def read_jsonl(path: Path, model_cls: type[T]) -> list[T]:
    """Read every line from a JSONL file and parse via model_cls.model_validate_json.

    Returns [] for an empty file. Raises FileNotFoundError for a missing file.
    Raises ValidationError (pydantic) on any malformed line.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: list[T] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(model_cls.model_validate_json(line))
    return records


__all__ = ["JsonlWriter", "read_jsonl"]
