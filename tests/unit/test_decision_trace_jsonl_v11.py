"""Tests for decision_trace.jsonl schema 1.1 (additive node-tuple commitment).

Schema 1.1 adds five optional fields that make the
"all-output-must-be-a-node" commitment first-class:

    output_node_tuple, edge_traversed, mask_version_id,
    posterior_summary, axis_node_ids

These tests verify:
  - v1.0-shaped records (no new fields) still validate and round-trip.
  - Each new field round-trips correctly through JSON.
  - The writer correctly persists v1.1 records and read_decision_trace
    reconstitutes them.
  - DECISION_TRACE_SCHEMA_VERSION advanced to "1.1".

Construction is purely additive; existing required fields remain required.
"""
from __future__ import annotations

from pathlib import Path

from nga.drivers.decision_trace_jsonl import (
    DECISION_TRACE_SCHEMA_VERSION,
    ControlAction,
    DecisionTraceRecord,
    MaskAction,
    open_decision_trace_writer,
    read_decision_trace,
)


def _make_minimal_record(**overrides: object) -> DecisionTraceRecord:
    """Build a v1.0-shaped DecisionTraceRecord with all required fields populated.

    Tests can override any field via kwargs.
    """
    base: dict[str, object] = {
        "run_id": "E1_A0_seed42",
        "experiment": "E1",
        "ablation": "A0",
        "seed": 42,
        "step": 0,
        "sample_id": "sample-0",
        "confidence": 0.9,
        "top1_state": "q_open",
        "top2_state": "q_closed",
        "top1_prob": 0.9,
        "top2_prob": 0.1,
        "margin": 0.8,
        "mask": MaskAction(
            enabled=True,
            current_state="q_init",
            illegal_indices_zeroed=[2, 5],
            pre_mask_argmax="q_open",
            post_mask_argmax="q_open",
            mask_changed_prediction=False,
        ),
        "control": ControlAction(
            decision="ROUTE_NORMAL",
            reason="sigma below threshold",
            sigma_threshold_used=0.5,
            sigma_observed=0.1,
        ),
    }
    base.update(overrides)
    return DecisionTraceRecord(**base)  # type: ignore[arg-type]


def _round_trip(record: DecisionTraceRecord) -> DecisionTraceRecord:
    """Serialise to JSON and re-validate, mimicking the JsonlWriter / read_jsonl path."""
    return DecisionTraceRecord.model_validate_json(record.model_dump_json())


def test_v11_schema_version_string() -> None:
    """The driver advertises schema 1.1."""
    assert DECISION_TRACE_SCHEMA_VERSION == "1.1"


def test_v10_record_still_loads() -> None:
    """A record populating only v1.0 fields validates and round-trips; new fields are None."""
    record = _make_minimal_record()
    restored = _round_trip(record)

    assert restored.run_id == "E1_A0_seed42"
    assert restored.top1_state == "q_open"
    # All five v1.1 fields must default to None when not provided.
    assert restored.output_node_tuple is None
    assert restored.edge_traversed is None
    assert restored.mask_version_id is None
    assert restored.posterior_summary is None
    assert restored.axis_node_ids is None


def test_v11_optional_fields_default_to_none() -> None:
    """A freshly-constructed record exposes all five new fields with default None."""
    record = _make_minimal_record()
    assert record.output_node_tuple is None
    assert record.edge_traversed is None
    assert record.mask_version_id is None
    assert record.posterior_summary is None
    assert record.axis_node_ids is None


def test_v11_record_with_node_tuple() -> None:
    """output_node_tuple round-trips as an ordered list of axis node IDs."""
    tup = ["q_open", "latent_2", "sigma_low", "energy_mid"]
    record = _make_minimal_record(output_node_tuple=tup)
    restored = _round_trip(record)

    assert restored.output_node_tuple == tup
    # Ordering matters; be explicit.
    assert restored.output_node_tuple is not None
    assert restored.output_node_tuple[0] == "q_open"
    assert restored.output_node_tuple[3] == "energy_mid"


def test_v11_edge_traversed_round_trip() -> None:
    """edge_traversed = (src, dst); after JSON round-trip we still see src in [0] and dst in [1]."""
    src = ["q_init", "latent_0", "sigma_low", "energy_low"]
    dst = ["q_open", "latent_2", "sigma_low", "energy_mid"]
    record = _make_minimal_record(edge_traversed=(src, dst))
    restored = _round_trip(record)

    assert restored.edge_traversed is not None
    assert restored.edge_traversed[0] == src
    assert restored.edge_traversed[1] == dst
    # Pydantic coerces the JSON list-of-lists back into a tuple-of-lists.
    assert isinstance(restored.edge_traversed, tuple)
    assert len(restored.edge_traversed) == 2


def test_v11_mask_version_id_round_trip() -> None:
    """mask_version_id is a short opaque hash string; preserved verbatim through JSON."""
    mask_hash = "a3f1c08e4b29"
    record = _make_minimal_record(mask_version_id=mask_hash)
    restored = _round_trip(record)

    assert restored.mask_version_id == mask_hash


def test_v11_posterior_summary_round_trip() -> None:
    """posterior_summary is a free dict[str, float]; keys and values preserved."""
    summary = {"mean_alpha": 1.7, "mean_beta": 2.3, "mean_entropy": 0.42}
    record = _make_minimal_record(posterior_summary=summary)
    restored = _round_trip(record)

    assert restored.posterior_summary == summary
    assert restored.posterior_summary is not None
    assert set(restored.posterior_summary.keys()) == {
        "mean_alpha",
        "mean_beta",
        "mean_entropy",
    }
    assert restored.posterior_summary["mean_alpha"] == 1.7


def test_v11_axis_node_ids_round_trip() -> None:
    """axis_node_ids is dict[axis_name, axis_node_id]; preserved through JSON."""
    axis_map = {
        "fsm_state": "q_open",
        "latent": "latent_2",
        "sigma": "sigma_low",
        "energy": "energy_mid",
    }
    record = _make_minimal_record(axis_node_ids=axis_map)
    restored = _round_trip(record)

    assert restored.axis_node_ids == axis_map


def test_writer_writes_v11_records(tmp_path: Path) -> None:
    """A v1.1 record routed through the JsonlWriter survives the read_decision_trace round-trip."""
    path = tmp_path / "decision_trace.jsonl"
    src = ["q_init", "latent_0", "sigma_low", "energy_low"]
    dst = ["q_open", "latent_2", "sigma_low", "energy_mid"]
    record = _make_minimal_record(
        output_node_tuple=dst,
        edge_traversed=(src, dst),
        mask_version_id="a3f1c08e4b29",
        posterior_summary={"mean_alpha": 1.7, "mean_beta": 2.3, "mean_entropy": 0.42},
        axis_node_ids={
            "fsm_state": "q_open",
            "latent": "latent_2",
            "sigma": "sigma_low",
            "energy": "energy_mid",
        },
    )

    with open_decision_trace_writer(path) as writer:
        writer.append(record)
        assert writer.count == 1

    rows = read_decision_trace(path)
    assert len(rows) == 1
    restored = rows[0]

    assert restored.output_node_tuple == dst
    assert restored.edge_traversed is not None
    assert restored.edge_traversed[0] == src
    assert restored.edge_traversed[1] == dst
    assert restored.mask_version_id == "a3f1c08e4b29"
    assert restored.posterior_summary == {
        "mean_alpha": 1.7,
        "mean_beta": 2.3,
        "mean_entropy": 0.42,
    }
    assert restored.axis_node_ids == {
        "fsm_state": "q_open",
        "latent": "latent_2",
        "sigma": "sigma_low",
        "energy": "energy_mid",
    }
    # v1.0 fields untouched and still present.
    assert restored.run_id == "E1_A0_seed42"
    assert restored.top1_state == "q_open"
    assert restored.control.decision == "ROUTE_NORMAL"
