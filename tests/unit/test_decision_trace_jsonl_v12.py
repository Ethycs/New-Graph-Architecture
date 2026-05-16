"""Tests for decision_trace.jsonl schema 1.2 (labelled-hypergraph fields).

Schema 1.2 adds four additive optional fields that carry the per-step
labelled-hypergraph view:

    regime_named_label, regime_residual_features,
    regime_kl_signature_hash, feature_delta_at_transition

Acceptance bar A3 from ``docs/proposals/labelled-hypergraph.md``: every
existing consumer reads a v1.2 row with the new fields populated; a v1.1
row reads in v1.2 with new fields defaulted to None.
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
    base: dict[str, object] = {
        "run_id": "E30_A0_seed42",
        "experiment": "E30",
        "ablation": "A0",
        "seed": 42,
        "step": 0,
        "sample_id": "sample-0",
        "confidence": 0.9,
        "top1_state": "regime_7",
        "top2_state": "regime_8",
        "top1_prob": 0.9,
        "top2_prob": 0.1,
        "margin": 0.8,
        "mask": MaskAction(
            enabled=False,
            current_state="regime_7",
            illegal_indices_zeroed=[],
            pre_mask_argmax="regime_7",
            post_mask_argmax="regime_7",
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
    return DecisionTraceRecord.model_validate_json(record.model_dump_json())


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


def test_v12_schema_version_string() -> None:
    """The driver advertises schema 1.2."""
    assert DECISION_TRACE_SCHEMA_VERSION == "1.2"


# ---------------------------------------------------------------------------
# Backward compatibility: v1.0/v1.1 records still validate with new fields = None
# ---------------------------------------------------------------------------


def test_v11_record_loads_under_v12() -> None:
    """A record with no v1.2 fields validates and the new fields default to None."""
    record = _make_minimal_record()
    restored = _round_trip(record)

    assert restored.regime_named_label is None
    assert restored.regime_residual_features is None
    assert restored.regime_kl_signature_hash is None
    assert restored.feature_delta_at_transition is None


def test_v12_fields_default_to_none() -> None:
    """Defaults on a freshly-constructed record."""
    record = _make_minimal_record()
    assert record.regime_named_label is None
    assert record.regime_residual_features is None
    assert record.regime_kl_signature_hash is None
    assert record.feature_delta_at_transition is None


# ---------------------------------------------------------------------------
# v1.2 fields round-trip
# ---------------------------------------------------------------------------


def test_regime_named_label_round_trip() -> None:
    label = {"fsm_state": "q_open", "sae_dominant": "f_142_eiffel"}
    record = _make_minimal_record(regime_named_label=label)
    restored = _round_trip(record)
    assert restored.regime_named_label == label


def test_regime_residual_features_round_trip() -> None:
    residual = ["f_3017", "f_4422", "f_8801"]
    record = _make_minimal_record(regime_residual_features=residual)
    restored = _round_trip(record)
    assert restored.regime_residual_features == residual


def test_regime_kl_signature_hash_round_trip() -> None:
    h = "a3f1c08e4b29"
    record = _make_minimal_record(regime_kl_signature_hash=h)
    restored = _round_trip(record)
    assert restored.regime_kl_signature_hash == h


def test_feature_delta_at_transition_round_trip() -> None:
    delta = {"f_3017": 0.42, "f_142": -0.18, "f_8801": 0.05}
    record = _make_minimal_record(feature_delta_at_transition=delta)
    restored = _round_trip(record)
    assert restored.feature_delta_at_transition == delta


# ---------------------------------------------------------------------------
# Writer integration
# ---------------------------------------------------------------------------


def test_writer_writes_v12_records(tmp_path: Path) -> None:
    """A v1.2 record routed through the JsonlWriter survives read_decision_trace."""
    path = tmp_path / "decision_trace.jsonl"
    record = _make_minimal_record(
        regime_named_label={"fsm_state": "q_open", "sae_dominant": "f_142"},
        regime_residual_features=["f_3017", "f_4422"],
        regime_kl_signature_hash="a3f1c08e4b29",
        feature_delta_at_transition={"f_3017": 0.42, "f_142": -0.18},
    )

    with open_decision_trace_writer(path) as writer:
        writer.append(record)
        assert writer.count == 1

    rows = read_decision_trace(path)
    assert len(rows) == 1
    restored = rows[0]

    assert restored.regime_named_label == {
        "fsm_state": "q_open",
        "sae_dominant": "f_142",
    }
    assert restored.regime_residual_features == ["f_3017", "f_4422"]
    assert restored.regime_kl_signature_hash == "a3f1c08e4b29"
    assert restored.feature_delta_at_transition == {"f_3017": 0.42, "f_142": -0.18}
    # v1.0 + v1.1 fields untouched.
    assert restored.run_id == "E30_A0_seed42"
    assert restored.top1_state == "regime_7"
    assert restored.control.decision == "ROUTE_NORMAL"
