"""Graph FSM Spec driver schema - pydantic v2 models for the task state-machine YAML.

The GraphFSMSpec captures vertices, edges, geometric coordinates, and optional
validation metadata. On load the following cross-field invariants are checked:
  - vertex_count == len(vertices)
  - edge_count == len(edges)
  - Every edge.source and edge.target references a real vertex id
  - If coordinates.node_embeddings is set, its length equals vertex_count and each
    row has coordinates.dimension floats
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nga.drivers._version import GRAPH_FSM_SCHEMA_VERSION, check_supported


class Vertex(BaseModel):
    """A node in the task state-machine graph."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    layer: int | None = None
    w_v: float = Field(default=1.0, ge=0.0)
    g_v: float = 0.0
    m_v: str | None = None


class Edge(BaseModel):
    """A directed transition in the task state-machine graph."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    label: str | None = None
    weight: float = Field(default=1.0, ge=0.0)


class Coordinates(BaseModel):
    """Geometric embedding space for the graph vertices."""

    model_config = ConfigDict(extra="forbid")

    space: Literal["euclidean", "hyperbolic"] = "hyperbolic"
    dimension: int = Field(..., ge=1)
    # node_embeddings is a dict in the YAML (vertex_id -> float list),
    # stored as list[list[float]] after normalization by the parent validator.
    node_embeddings: dict[str, list[float]] | None = None


class Validation(BaseModel):
    """Optional sanity-check metadata; loaders enforce when present."""

    model_config = ConfigDict(extra="forbid")

    acyclic: bool | None = None
    start_nodes: list[str] = Field(default_factory=list)
    end_nodes: list[str] = Field(default_factory=list)
    invariant_edges: list[str] = Field(default_factory=list)


class GraphFSMSpec(BaseModel):
    """Pydantic v2 model for the Graph FSM Spec YAML.

    Invariants checked after construction:
      - vertex_count == len(vertices)
      - edge_count == len(edges)
      - Every edge endpoint references a real vertex id
      - If coordinates.node_embeddings is present, each entry has
        coordinates.dimension floats and all vertex ids are covered
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=GRAPH_FSM_SCHEMA_VERSION)
    name: str
    vertex_count: int = Field(..., ge=1)
    edge_count: int = Field(..., ge=0)
    vertices: list[Vertex]
    edges: list[Edge]
    coordinates: Coordinates
    validation: Validation = Field(default_factory=Validation)

    @model_validator(mode="after")
    def _check_invariants(self) -> "GraphFSMSpec":
        """Enforce cross-field structural invariants."""
        vertex_ids: set[str] = {v.id for v in self.vertices}

        if self.vertex_count != len(self.vertices):
            raise ValueError(
                f"vertex_count={self.vertex_count} does not match "
                f"len(vertices)={len(self.vertices)}"
            )

        if self.edge_count != len(self.edges):
            raise ValueError(
                f"edge_count={self.edge_count} does not match "
                f"len(edges)={len(self.edges)}"
            )

        for edge in self.edges:
            if edge.source not in vertex_ids:
                raise ValueError(
                    f"Edge source '{edge.source}' does not reference a known vertex id"
                )
            if edge.target not in vertex_ids:
                raise ValueError(
                    f"Edge target '{edge.target}' does not reference a known vertex id"
                )

        if self.coordinates.node_embeddings is not None:
            emb = self.coordinates.node_embeddings
            if len(emb) != self.vertex_count:
                raise ValueError(
                    f"coordinates.node_embeddings has {len(emb)} entries but "
                    f"vertex_count={self.vertex_count}"
                )
            for vid, row in emb.items():
                if len(row) != self.coordinates.dimension:
                    raise ValueError(
                        f"coordinates.node_embeddings['{vid}'] has {len(row)} floats "
                        f"but coordinates.dimension={self.coordinates.dimension}"
                    )

        return self

    def build_legality_matrix(self) -> list[list[bool]]:
        """Build a (V, V) boolean adjacency matrix.

        Returns a list of V lists, each of length V. Entry [i][j] is True iff
        there is an edge from vertices[i].id to vertices[j].id.
        """
        index: dict[str, int] = {v.id: i for i, v in enumerate(self.vertices)}
        v_count = len(self.vertices)
        matrix: list[list[bool]] = [[False] * v_count for _ in range(v_count)]
        for edge in self.edges:
            i = index[edge.source]
            j = index[edge.target]
            matrix[i][j] = True
        return matrix


def load(path: Path) -> GraphFSMSpec:
    """Read a GraphFSMSpec YAML file, validate schema_version, and return the spec."""
    with path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"GraphFSMSpec file {path} must be a YAML mapping, got {type(raw)}"
        )
    check_supported(str(raw.get("schema_version", "")), "GraphFSMSpec")
    return GraphFSMSpec.model_validate(raw)


def dump(obj: GraphFSMSpec, path: Path) -> None:
    """Write a GraphFSMSpec object to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj.model_dump(mode="json", exclude_none=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


__all__ = [
    "Vertex",
    "Edge",
    "Coordinates",
    "Validation",
    "GraphFSMSpec",
    "load",
    "dump",
]
