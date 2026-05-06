"""Decision-trace JSONL: per-step interpretability surface.

Each record exposes the FULL chain of signals that drove a single
prediction:

    typed scores -> mask -> margin -> stratum tag -> sigma signals -> energy
    -> partition -> control decision -> final action

The point is interpretability, not loss reduction. A run with N steps
produces N decision-trace rows; an analyst can read one row and reconstruct
exactly why the agent did what it did at that step. This is the data
backing the architecture's claim that the system is auditable in a way that
a flat classifier is not.

Companion to results.jsonl (which is the model-output stream) and
metrics.jsonl (which is the aggregate stream). decision_trace.jsonl is the
WHY stream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nga.drivers._version import (
    SUPPORTED_MAJOR,
    check_supported,
)
from nga.drivers.jsonl_writer import JsonlWriter, read_jsonl

DECISION_TRACE_SCHEMA_VERSION = "1.0"

__all__ = [
    "DECISION_TRACE_SCHEMA_VERSION",
    "DecisionTraceRecord",
    "MaskAction",
    "ControlAction",
    "open_decision_trace_writer",
    "read_decision_trace",
]


class MaskAction(BaseModel):
    """Record of what the legality mask did this step.

    Captures whether the graph mask was enabled, which illegal indices were
    zeroed, what the classifier wanted before masking, and whether the mask
    actually changed the prediction. Together these fields let an analyst
    reconstruct the mask's causal contribution to the final action.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    """Whether ablation.graph_mask_enabled was True for this step."""

    current_state: str | None
    """The FSM state at the start of this step, or None on the first step."""

    illegal_indices_zeroed: list[int] = Field(default_factory=list)
    """Indices in the softmax distribution that were zeroed by the mask."""

    pre_mask_argmax: str | None = None
    """The classifier's argmax prediction before the mask was applied."""

    post_mask_argmax: str | None = None
    """The classifier's argmax prediction after the mask was applied."""

    mask_changed_prediction: bool = False
    """True iff post_mask_argmax != pre_mask_argmax."""


class ControlAction(BaseModel):
    """The control surface's verdict for this step.

    Phase 5 v1: a categorical decision. ROUTE_NORMAL means the trained
    classifier's prediction stands. ROUTE_RECOVERY means sigma exceeded its
    threshold and the runner is expected to invoke a recovery branch.
    ABSTAIN means even recovery is not appropriate (sigma very high or
    illegal mass dominates) and the runner returns no action.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["ROUTE_NORMAL", "ROUTE_RECOVERY", "ABSTAIN"]
    """The control surface's categorical verdict for this step."""

    reason: str
    """Short human-readable explanation for why this decision was reached."""

    sigma_threshold_used: float
    """The sigma threshold that separates ROUTE_NORMAL from ROUTE_RECOVERY."""

    sigma_observed: float
    """The actual sigma value that was compared against the threshold."""

    abstain_threshold_used: float | None = None
    """The sigma threshold that separates ROUTE_RECOVERY from ABSTAIN, if applicable."""


class DecisionTraceRecord(BaseModel):
    """One row per step. The full WHY of a single prediction.

    Fields are grouped by which architectural layer they come from:

      - identification: schema_version, run_id, experiment, ablation, seed,
                        step, sample_id
      - typed scoring: confidence, top1_state, top2_state, top1_prob, top2_prob,
                       margin
      - graph mask: mask (MaskAction sub-record)
      - behavioral stratum: stratum_tag, stratum_bitmask, stratum_confidence
      - sigma breakdown: sigma_total, sigma_signals (dict of named signals to
                         their contribution)
      - energy: energy_total, energy_breakdown (dict)
      - partition (when available): P_per_stratum (dict), free_energy
      - monodromy (when available): monodromy_class (int), in_closed_walk (bool)
      - control: control (ControlAction sub-record)
      - audit: timestamp

    All Phase-4-and-up fields are Optional. A Phase 1 runner that does not
    have monodromy or partition data writes None for those fields. The trace
    structure is forward-compatible.

    sigma_signals is a free dict (not a model) because the signal set
    evolves. Documented keys: margin, decision_tie, illegal, loop,
    stabilizer, catastrophe_bias. Additional signals can be added without
    a schema bump.

    P_lambda and free_energy are run-level scalars repeated on each row.
    This is intentionally redundant: it lets a single trace row stand alone
    for analysis without joining against an aggregate file.
    """

    model_config = ConfigDict(extra="forbid")

    # -- identification --------------------------------------------------
    schema_version: str = DECISION_TRACE_SCHEMA_VERSION
    run_id: str
    """Composite run identifier, e.g. 'E1_A0_seed42'."""

    experiment: str
    """Experiment label, e.g. 'E1'."""

    ablation: str
    """Ablation label, e.g. 'A0'."""

    seed: int
    """RNG seed for this run."""

    step: int
    """Step index within the episode or evaluation batch."""

    sample_id: str
    """Stable sample identifier; joins to results.jsonl on the same key."""

    # -- typed scoring ---------------------------------------------------
    confidence: float = Field(..., ge=0.0, le=1.0)
    """max(softmax) after masking; the classifier's confidence in top1_state."""

    top1_state: str
    """The argmax state after the legality mask was applied."""

    top2_state: str | None = None
    """The second-highest-scoring state after masking, if available."""

    top1_prob: float = Field(..., ge=0.0, le=1.0)
    """Softmax probability assigned to top1_state."""

    top2_prob: float = Field(..., ge=0.0, le=1.0)
    """Softmax probability assigned to top2_state (0.0 if top2_state is None)."""

    margin: float = Field(..., ge=-1.0, le=1.0)
    """top1_prob - top2_prob. Negative only when forced by masking."""

    # -- graph mask ------------------------------------------------------
    mask: MaskAction
    """Full record of what the legality mask did this step."""

    # -- behavioral stratum ----------------------------------------------
    stratum_tag: str | None = None
    """String name of the SingularityType that fired (e.g. 'nominal', 'low_margin')."""

    stratum_bitmask: int | None = None
    """Integer bitmask recording every rule that fired, per TagResult.bitmask."""

    stratum_confidence: float | None = None
    """Magnitude of the strongest firing signal in [0, 1], per TagResult.confidence."""

    # -- sigma -----------------------------------------------------------
    sigma_total: float | None = None
    """The aggregate singularity score sigma(x) in [0, 1]."""

    sigma_signals: dict[str, float] | None = None
    """Per-signal contributions to sigma before weighting.

    Documented keys (Phase 2): margin, decision_tie, illegal, loop,
    stabilizer, catastrophe_bias. New signals can be added in later phases
    without a schema version bump.
    """

    # -- energy + partition ----------------------------------------------
    energy_total: float | None = None
    """Scalar sum of all weighted energy contributions for this step."""

    energy_breakdown: dict[str, float] | None = None
    """Per-term energy contributions, e.g. {'cost': 0.05, 'uncertainty': 0.25, 'progress': -0.5}."""

    P_lambda: dict[str, float] | None = None
    """Per-stratum probability from the partition function when run-level Z is available."""

    free_energy: float | None = None
    """Free energy F = -T log Z at the run level. Repeated on each row for self-contained rows."""

    # -- monodromy (Phase 4 v2) -----------------------------------------
    monodromy_class: int | None = None
    """Opaque integer encoding (rho_orbit_length, rho_tau_orbit_length) for the active dart."""

    in_closed_walk: bool | None = None
    """True iff the current dart lies on a closed walk (rho*tau orbit length == 1)."""

    # -- control ---------------------------------------------------------
    control: ControlAction
    """The control surface's verdict for this step."""

    # -- audit -----------------------------------------------------------
    timestamp: str | None = None
    """ISO 8601 UTC wall-clock time at which this record was written."""


def open_decision_trace_writer(path: Path) -> JsonlWriter[DecisionTraceRecord]:
    """Open an append-only writer for DecisionTraceRecord rows.

    Parameters
    ----------
    path:
        Filesystem path to the decision_trace.jsonl file. The file is
        opened in append mode; existing rows are preserved.

    Returns
    -------
    JsonlWriter[DecisionTraceRecord]
        A context-manager-compatible writer. Call .append(record) for each
        step; use as a context manager to ensure the file is flushed and
        closed on exit.
    """
    return JsonlWriter(path, DecisionTraceRecord)


def read_decision_trace(path: Path) -> list[DecisionTraceRecord]:
    """Read every row from a decision_trace.jsonl file, validating each against the schema.

    Pydantic validation runs on every line. Corrupted or schema-drifted rows
    raise pydantic.ValidationError rather than silently returning partial data.
    An empty file returns an empty list. A missing file raises FileNotFoundError.

    Parameters
    ----------
    path:
        Filesystem path to the decision_trace.jsonl file to read.

    Returns
    -------
    list[DecisionTraceRecord]
        All validated rows in file order.
    """
    return read_jsonl(path, DecisionTraceRecord)
