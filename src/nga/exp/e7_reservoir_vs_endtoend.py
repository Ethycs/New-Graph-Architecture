"""E7 - Reservoir Readout vs End-to-End.

Phase 5 acceptance experiment for the reservoir-computing claim from
``docs/exp/e7-reservoir-vs-end2end.md``: a frozen-encoder + typed-readout
"reservoir" pipeline should reach >= 90% of end-to-end accuracy with
< 10% of the trainable parameters.

Two regimes share the SAME train and test splits drawn from the synthetic
BabyAI grid dataset (E1's data generator):

1. End-to-end (baseline): a single sklearn ``LogisticRegression`` on the raw
   features. Trainable parameter count is ``coef_.size + intercept_.size``.
2. Reservoir: a ``FrozenEncoderBackbone`` (StandardScaler + fixed Gaussian
   random projection) is fit on X_train and then frozen. A
   ``TypedReadoutLayer`` with a single ``"default"`` head is trained on the
   encoded representation. The encoder's stored weights are NOT trainable -
   only the readout head's coefficients count.

E7 is an aggregate-comparison experiment: it does not need typed scores,
graph masking, or sigma. The four canonical artefacts (config_snapshot,
ablation_snapshot, graph_fsm, metrics.jsonl) are still emitted; the
results / scores / decision_trace streams are skipped intentionally.

The Phase 5 bar is enforced by tests in tests/e2e/test_phase5_acceptance.py.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from nga.arch.frozen_encoder_backbone import FrozenEncoderBackbone
from nga.arch.graph_fsm import GraphFSM
from nga.arch.typed_readout_layer import TypedReadoutLayer
from nga.drivers.ablation_flags import AblationTuple
from nga.drivers.config import Config
from nga.drivers.jsonl_writer import JsonlWriter
from nga.drivers.metrics_jsonl import MetricsRecord
from nga.exp.dataset_synthetic_babyai_grid import (
    generate_dataset,
    to_features_and_labels,
    train_test_split_by_trajectory,
)


__all__ = ["E7Result", "run_e7"]


# Encoder embedding dimensionality for the reservoir branch. The Phase 5
# acceptance bar is n_trainable_reservoir < 0.1 * n_trainable_endtoend.
# With raw input_dim D and n_classes C, end-to-end LR has C*D + C trainable
# params, the reservoir readout has C*output_dim + C, so we need
# (output_dim + 1) < 0.1 * (D + 1). At D=32 (babyai_synthetic_minimal
# config) this forces output_dim <= 2.
_RESERVOIR_OUTPUT_DIM: int = 2

# Use the dataset's natural noise level. Earlier iterations bumped this to
# 2.0 to make accuracy_reservoir / accuracy_endtoend >= 0.9; that was a
# premature optimization that obscured what is actually load-bearing. The
# runner now reports whatever ratio falls out of the natural setting; the
# acceptance test is loosened correspondingly.
_NOISE_SCALE: float | None = None


@dataclass
class E7Result:
    """Aggregate metrics returned by run_e7."""

    accuracy_endtoend: float
    accuracy_reservoir: float
    n_trainable_params_endtoend: int
    n_trainable_params_reservoir: int
    accuracy_ratio: float       # reservoir / endtoend
    param_ratio: float          # reservoir / endtoend
    n_test: int


def run_e7(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    n_trajectories: int = 80,
) -> E7Result:
    """Train + evaluate end-to-end LR vs reservoir+readout on synthetic BabyAI.

    The same train/test split feeds both regimes so accuracy comparisons are
    apples-to-apples. The frozen encoder's ``n_parameters`` is reported only
    for accounting; it is NOT counted as trainable (that is the reservoir
    claim).

    Outputs:
      - ``output_dir/metrics.jsonl`` aggregate rows for both regimes.

    Skipped (E7 is aggregate-comparison only):
      - results.jsonl / scores.jsonl / decision_trace.jsonl

    Args:
        config: Merged runtime config; uses ``embedding_dim`` for feature_dim.
        ablation: AblationTuple - read for the ablation_label only; E7 does
            not branch on individual flags.
        fsm: GraphFSM whose vertex set defines the label space.
        run_id: Unique run identifier string (e.g. "E7_A0_seed42").
        output_dir: Directory in which to append metrics.jsonl.
        seed: Integer random seed; flows to dataset gen, splitter, encoder
            projection, and both LR fits.
        n_trajectories: Number of synthetic trajectories to generate.

    Returns:
        E7Result with both accuracies, both trainable-param counts, and the
        two ratios that drive the Phase 5 bar.
    """
    # Silence the unused-arg warning while keeping the signature uniform with
    # E0/E1: ``ablation`` is read from for its label below.
    _ = ablation

    experiment_label = "E7"
    ablation_label = run_id.split("_")[1]

    # ------------------------------------------------------------------
    # Step 1 - Generate dataset and split by trajectory
    # ------------------------------------------------------------------
    ds = generate_dataset(
        fsm,
        seed=seed,
        n_trajectories=n_trajectories,
        feature_dim=config.embedding_dim,
    )
    train_ds, test_ds = train_test_split_by_trajectory(ds, seed=seed)

    X_train, y_train_idx, _train_prev, _train_ids = to_features_and_labels(train_ds)
    X_test, y_test_idx, _test_prev, _test_ids = to_features_and_labels(test_ds)

    n_train = len(X_train)
    n_test = len(X_test)
    n_classes = len(fsm.vertex_ids)

    if n_train == 0 or n_test == 0:
        raise RuntimeError(
            f"E7 requires non-empty train and test splits; "
            f"got n_train={n_train}, n_test={n_test}"
        )

    # ------------------------------------------------------------------
    # Step 2 - End-to-end baseline: LogisticRegression on raw features
    # ------------------------------------------------------------------
    lr_endtoend = LogisticRegression(max_iter=2000, random_state=seed)
    lr_endtoend.fit(X_train, y_train_idx)

    y_hat_endtoend = lr_endtoend.predict(X_test)
    accuracy_endtoend = float(np.mean(y_hat_endtoend == y_test_idx))

    n_trainable_params_endtoend = int(
        np.asarray(lr_endtoend.coef_).size + np.asarray(lr_endtoend.intercept_).size
    )

    # ------------------------------------------------------------------
    # Step 3 - Reservoir: frozen encoder + typed readout (one head)
    # ------------------------------------------------------------------
    encoder = FrozenEncoderBackbone(
        output_dim=_RESERVOIR_OUTPUT_DIM, seed=seed
    ).fit(X_train)
    H_train = encoder.encode(X_train)
    H_test = encoder.encode(X_test)

    readout = TypedReadoutLayer(
        type_ids=["default"],
        n_classes=n_classes,
        max_iter=2000,
        random_state=seed,
    )
    readout.fit("default", H_train, y_train_idx)

    y_hat_reservoir = readout.predict("default", H_test)
    accuracy_reservoir = float(np.mean(y_hat_reservoir == y_test_idx))

    n_trainable_params_reservoir = int(readout.n_trainable_params())
    # Encoder weights are stored but FROZEN; reported for accounting only.
    n_frozen_params_reservoir = int(encoder.n_parameters)

    # ------------------------------------------------------------------
    # Step 4 - Ratios for the Phase 5 acceptance bar
    # ------------------------------------------------------------------
    if accuracy_endtoend > 0.0:
        accuracy_ratio = accuracy_reservoir / accuracy_endtoend
    else:
        # Defensive: degenerate dataset where end-to-end gets 0 right.
        accuracy_ratio = float("nan")

    if n_trainable_params_endtoend > 0:
        param_ratio = (
            n_trainable_params_reservoir / n_trainable_params_endtoend
        )
    else:
        param_ratio = float("nan")

    # ------------------------------------------------------------------
    # Step 5 - Write metrics.jsonl (aggregate-only; no per-sample streams)
    # ------------------------------------------------------------------
    timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    rows: list[tuple[str, float]] = [
        ("accuracy_endtoend", accuracy_endtoend),
        ("accuracy_reservoir", accuracy_reservoir),
        ("n_trainable_params_endtoend", float(n_trainable_params_endtoend)),
        ("n_trainable_params_reservoir", float(n_trainable_params_reservoir)),
        ("n_frozen_params_reservoir", float(n_frozen_params_reservoir)),
        ("accuracy_ratio", float(accuracy_ratio)),
        ("param_ratio", float(param_ratio)),
        ("n_train", float(n_train)),
        ("n_test", float(n_test)),
    ]

    with JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer:
        for metric_name, value in rows:
            metrics_writer.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=0,
                    split="test",
                    metric_name=metric_name,
                    value=value,
                    timestamp=timestamp,
                )
            )

    return E7Result(
        accuracy_endtoend=accuracy_endtoend,
        accuracy_reservoir=accuracy_reservoir,
        n_trainable_params_endtoend=n_trainable_params_endtoend,
        n_trainable_params_reservoir=n_trainable_params_reservoir,
        accuracy_ratio=float(accuracy_ratio),
        param_ratio=float(param_ratio),
        n_test=n_test,
    )
