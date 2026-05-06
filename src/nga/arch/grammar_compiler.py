"""Grammar compiler - DSL -> GraphFSMSpec translator.

A small declarative YAML DSL describing the state types and legal transitions
of a typed task graph, compiled into a fully validated GraphFSMSpec ready for
GraphFSM, attention, and energy modules.

DSL shape (schema_version "1.0"):

    schema_version: "1.0"
    name: <graph name>
    states:
      - name: <state id>          # required; used as Vertex.id and Vertex.label
        layer: <int>               # optional; populates Vertex.layer (default null)
        w_v: <float>               # optional; default 1.0
        g_v: <float>               # optional; default 0.0
      - ...
    transitions:
      - from: <state name>
        to: <state name>
        weight: <float>            # optional; default 1.0
      - ...
    embedding:
      space: euclidean | hyperbolic   # default hyperbolic
      dimension: <int>                # required, >= 1
      default_coordinate: [floats]    # optional; default zero-vector of dimension
    validation:                       # optional
      acyclic: <bool>
      start_nodes: [<state name>, ...]
      end_nodes: [<state name>, ...]

Compilation rules:
  - The compiler emits Vertex(id=name, label=name, layer=layer, w_v=w_v, g_v=g_v)
    in the order they appear in `states`.
  - Each transition emits an Edge(source, target, weight,
    label=f"{source}_to_{target}").
  - coordinates.node_embeddings is filled with the default_coordinate (zero
    vector of `dimension` floats by default) for every state.
  - validation passes through unchanged when present.
  - Unknown state references in `transitions` raise pydantic ValidationError.

The compiled GraphFSMSpec is the canonical persisted format and is fully
interchangeable with hand-authored .fsm.yaml files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nga.drivers._version import check_supported
from nga.drivers.graph_fsm_spec import (
    Coordinates,
    Edge,
    GraphFSMSpec,
    Validation,
    Vertex,
    dump as _dump_spec,
)

__all__ = [
    "GrammarState",
    "GrammarTransition",
    "GrammarEmbedding",
    "GrammarValidation",
    "GrammarSpec",
    "load_grammar",
    "compile_grammar",
    "compile_to_yaml",
]


class GrammarState(BaseModel):
    """A typed state declaration in the grammar DSL."""

    model_config = ConfigDict(extra="forbid")

    name: str
    layer: int | None = None
    w_v: float = Field(default=1.0, ge=0.0)
    g_v: float = 0.0


class GrammarTransition(BaseModel):
    """A directed legal transition between two declared states."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    src: str = Field(alias="from")
    dst: str = Field(alias="to")
    weight: float = Field(default=1.0, ge=0.0)


class GrammarEmbedding(BaseModel):
    """Geometric embedding configuration for the compiled graph."""

    model_config = ConfigDict(extra="forbid")

    space: Literal["euclidean", "hyperbolic"] = "hyperbolic"
    dimension: int = Field(..., ge=1)
    default_coordinate: list[float] | None = None

    @model_validator(mode="after")
    def _check_default_coordinate(self) -> "GrammarEmbedding":
        if self.default_coordinate is not None and len(self.default_coordinate) != self.dimension:
            raise ValueError(
                f"default_coordinate has {len(self.default_coordinate)} floats but "
                f"dimension={self.dimension}"
            )
        return self


class GrammarValidation(BaseModel):
    """Optional validation metadata passed through to GraphFSMSpec.validation."""

    model_config = ConfigDict(extra="forbid")

    acyclic: bool | None = None
    start_nodes: list[str] = Field(default_factory=list)
    end_nodes: list[str] = Field(default_factory=list)
    invariant_edges: list[str] = Field(default_factory=list)


class GrammarSpec(BaseModel):
    """Top-level grammar DSL document."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    name: str
    states: list[GrammarState] = Field(..., min_length=1)
    transitions: list[GrammarTransition] = Field(default_factory=list)
    embedding: GrammarEmbedding
    validation: GrammarValidation = Field(default_factory=GrammarValidation)

    @model_validator(mode="after")
    def _check_state_references(self) -> "GrammarSpec":
        names = [s.name for s in self.states]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate state names: {names}")
        known = set(names)
        for tr in self.transitions:
            if tr.src not in known:
                raise ValueError(
                    f"transition.from='{tr.src}' references unknown state; "
                    f"known states: {sorted(known)}"
                )
            if tr.dst not in known:
                raise ValueError(
                    f"transition.to='{tr.dst}' references unknown state; "
                    f"known states: {sorted(known)}"
                )
        for vid in self.validation.start_nodes + self.validation.end_nodes:
            if vid not in known:
                raise ValueError(
                    f"validation references unknown state '{vid}'; "
                    f"known states: {sorted(known)}"
                )
        return self


def load_grammar(path: Path) -> GrammarSpec:
    """Read a grammar YAML file, validate schema_version, and return the GrammarSpec."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"GrammarSpec file {path} must be a YAML mapping, got {type(raw)}"
        )
    check_supported(str(raw.get("schema_version", "")), "GrammarSpec")
    return GrammarSpec.model_validate(raw)


def compile_grammar(spec: GrammarSpec) -> GraphFSMSpec:
    """Compile a GrammarSpec into a fully validated GraphFSMSpec."""
    vertices = [
        Vertex(
            id=s.name,
            label=s.name,
            layer=s.layer,
            w_v=s.w_v,
            g_v=s.g_v,
            m_v=None,
        )
        for s in spec.states
    ]

    edges = [
        Edge(
            source=tr.src,
            target=tr.dst,
            label=f"{tr.src}_to_{tr.dst}",
            weight=tr.weight,
        )
        for tr in spec.transitions
    ]

    default_coord = (
        list(spec.embedding.default_coordinate)
        if spec.embedding.default_coordinate is not None
        else [0.0] * spec.embedding.dimension
    )
    node_embeddings: dict[str, list[float]] = {
        s.name: list(default_coord) for s in spec.states
    }

    coordinates = Coordinates(
        space=spec.embedding.space,
        dimension=spec.embedding.dimension,
        node_embeddings=node_embeddings,
    )

    validation = Validation(
        acyclic=spec.validation.acyclic,
        start_nodes=list(spec.validation.start_nodes),
        end_nodes=list(spec.validation.end_nodes),
        invariant_edges=list(spec.validation.invariant_edges),
    )

    return GraphFSMSpec(
        schema_version=spec.schema_version,
        name=spec.name,
        vertex_count=len(vertices),
        edge_count=len(edges),
        vertices=vertices,
        edges=edges,
        coordinates=coordinates,
        validation=validation,
    )


def compile_to_yaml(grammar_path: Path, out_path: Path) -> GraphFSMSpec:
    """Convenience: load a grammar YAML, compile it, dump the GraphFSMSpec to disk."""
    spec = load_grammar(Path(grammar_path))
    compiled = compile_grammar(spec)
    _dump_spec(compiled, Path(out_path))
    return compiled
