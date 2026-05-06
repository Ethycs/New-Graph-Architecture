"""E8 - Transfer / OOD Experiment.

The original E8 spec (docs/exp/e8-transfer-experiment.md) calls for cross-task
transfer (e.g. train on BabyAI Goto, test zero-shot on BabyAI Pickup). Real
cross-environment task families are not yet wired into this codebase, so this
runner implements a faithful in-distribution-shift surrogate using the
synthetic BabyAI grid generator:

  - TRAIN set: generate_dataset(seed=seed, ...) at the default noise_scale.
  - TEST set:  generate_dataset(seed=seed+1000, noise_scale=1.5*default, ...)
               i.e. a fresh trajectory population drawn under a wider class
               noise (more class overlap), simulating a related-but-distinct
               domain.

The classifier (sklearn LogisticRegression) is fit on the train split and
evaluated zero-shot on both splits. We report:

    accuracy_train   - in-distribution training-set accuracy (sanity)
    accuracy_test    - zero-shot accuracy on the shifted distribution
    transfer_gap     - accuracy_train - accuracy_test
    n_train, n_test  - sample counts on each split

This is REPORT-ONLY. No numerical bar is enforced on transfer_gap because on
small synthetic datasets it can swing wildly. The point of the experiment is
to wire up the transfer-evaluation harness end-to-end so later experiments
(or real BabyAI/MiniGrid sources) can plug in without further plumbing.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from nga.arch.graph_fsm import GraphFSM
from nga.drivers.ablation_flags import AblationTuple
from nga.drivers.config import Config
from nga.drivers.jsonl_writer import JsonlWriter
from nga.drivers.metrics_jsonl import MetricsRecord
from nga.drivers.results_jsonl import ResultsRecord
from nga.exp.dataset_synthetic_babyai_grid import (
    generate_dataset,
    to_features_and_labels,
)

__all__ = ["E8Result", "run_e8"]

# Same canonical 7-vertex set used by E1 / the synthetic BabyAI fixture.
_EXPECTED_VERTEX_IDS: list[str] = [
    "Parse",
    "Navigate",
    "ResolveDoor",
    "Pickup",
    "Deliver",
    "Interact",
    "Done",
]

# Default noise_scale matches dataset_synthetic_babyai_grid.generate_dataset.
_DEFAULT_NOISE_SCALE: float = 0.30
_TEST_NOISE_MULTIPLIER: float = 1.5
_TEST_SEED_OFFSET: int = 1000


@dataclass
class E8Result:
    """Aggregate metrics returned by run_e8."""

    accuracy_train: float
    accuracy_test: float
    transfer_gap: float    # accuracy_train - accuracy_test
    n_train: int
    n_test: int


def run_e8(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    n_trajectories: int = 80,
) -> E8Result:
    """Train on default synthetic BabyAI; eval zero-shot on a noise-shifted twin.

    Outputs (CLI has already created empty sinks; this function appends):
      - output_dir/metrics.jsonl  rows for accuracy_train, accuracy_test,
                                  transfer_gap, n_train, n_test.
      - output_dir/results.jsonl  per-test-sample predictions (zero-shot,
                                  on the shifted distribution).
      - output_dir/scores.jsonl   touched-empty (no typed score stream here).
      - output_dir/decision_trace.jsonl  not produced; E8 is a transfer
                                  benchmark, not a per-sample decision trace.

    Parameters
    ----------
    config:
        Merged runtime config; embedding_dim is used for feature_dim of both
        train and test datasets.
    ablation:
        AblationTuple. E8 itself does not consume ablation flags - the
        classifier is the same plain LogisticRegression in both regimes -
        but the parameter is accepted for CLI symmetry with E0/E1.
    fsm:
        GraphFSM matching the 7-vertex BabyAI synthetic graph.
    run_id:
        Unique run identifier string (e.g. "E8_A0_seed42").
    output_dir:
        Directory in which to append JSONL files.
    seed:
        Integer random seed. The train set uses seed; the test set uses
        seed + 1000 with 1.5x noise to enforce distribution shift.
    n_trajectories:
        Number of synthetic trajectories per split. Each split gets the same
        count so transfer_gap is not dominated by sample-size noise.

    Returns
    -------
    E8Result
        accuracy_train, accuracy_test, transfer_gap, n_train, n_test.

    Raises
    ------
    ValueError
        If fsm.vertex_ids does not match the expected 7-vertex set.
    """
    # ------------------------------------------------------------------
    # Step 1 - Validate FSM vertex set
    # ------------------------------------------------------------------
    if fsm.vertex_ids != _EXPECTED_VERTEX_IDS:
        raise ValueError(
            f"E8 requires fsm.vertex_ids == {_EXPECTED_VERTEX_IDS!r}; "
            f"got {fsm.vertex_ids!r}"
        )

    _ = ablation  # accepted for CLI symmetry; not consumed by E8.

    # ------------------------------------------------------------------
    # Step 2 - Generate the TRAIN dataset (default-noise, seed=seed)
    # ------------------------------------------------------------------
    train_ds = generate_dataset(
        fsm,
        seed=seed,
        n_trajectories=n_trajectories,
        feature_dim=config.embedding_dim,
        noise_scale=_DEFAULT_NOISE_SCALE,
    )

    # ------------------------------------------------------------------
    # Step 3 - Generate the TEST dataset (shifted: seed+1000, noise*1.5).
    # The generator's `noise_scale` knob is the cleanest available shift
    # axis: it widens per-class Gaussians without changing the FSM, so
    # train-time geometry (class means) survives but test-time samples
    # spill across class boundaries more often. This lets transfer_gap
    # measure robustness to a controlled, well-defined OOD shift.
    # ------------------------------------------------------------------
    test_ds = generate_dataset(
        fsm,
        seed=seed + _TEST_SEED_OFFSET,
        n_trajectories=n_trajectories,
        feature_dim=config.embedding_dim,
        noise_scale=_DEFAULT_NOISE_SCALE * _TEST_NOISE_MULTIPLIER,
    )

    # ------------------------------------------------------------------
    # Step 4 - Convert both splits to (X, y_str) arrays.
    # ------------------------------------------------------------------
    X_train, y_train_idx, _train_prev, train_sample_ids = to_features_and_labels(train_ds)
    X_test, y_test_idx, _test_prev, test_sample_ids = to_features_and_labels(test_ds)

    y_train_str: list[str] = [fsm.vertex_ids[int(i)] for i in y_train_idx]
    y_test_str: list[str] = [fsm.vertex_ids[int(i)] for i in y_test_idx]

    n_train = len(X_train)
    n_test = len(X_test)

    # ------------------------------------------------------------------
    # Step 5 - Fit LogReg on the train split. No mask, no FSM context;
    # E8's question is purely about feature-distribution transfer.
    # ------------------------------------------------------------------
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train_str)

    # ------------------------------------------------------------------
    # Step 6 - Predict on TRAIN and TEST.
    # ------------------------------------------------------------------
    y_hat_train: np.ndarray = clf.predict(X_train)
    y_hat_test: np.ndarray = clf.predict(X_test)

    y_train_arr = np.asarray(y_train_str)
    y_test_arr = np.asarray(y_test_str)

    accuracy_train = float(np.mean(y_hat_train == y_train_arr)) if n_train > 0 else 0.0
    accuracy_test = float(np.mean(y_hat_test == y_test_arr)) if n_test > 0 else 0.0
    transfer_gap = accuracy_train - accuracy_test

    # ------------------------------------------------------------------
    # Step 7 - Write artefacts.
    # ------------------------------------------------------------------
    experiment_label = "E8"
    ablation_label = run_id.split("_")[1]
    timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()

    with (
        JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer,
        JsonlWriter(output_dir / "results.jsonl", ResultsRecord) as results_writer,
    ):
        # --- Per-test-sample results: zero-shot predictions on the
        #     shifted distribution. We log the test split (not train)
        #     because that is the load-bearing zero-shot signal.
        for i in range(n_test):
            results_writer.append(
                ResultsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=i,
                    sample_id=test_sample_ids[i],
                    y_true=y_test_str[i],
                    y_hat=str(y_hat_test[i]),
                    margin=0.0,
                    singular_flag=False,
                    sigma_score=0.0,
                    behavioral_stratum="REGULAR",
                    stratum_bitmask=0,
                    transition_legal=None,
                )
            )

        for metric_name, value, split in [
            ("accuracy_train", accuracy_train, "train"),
            ("accuracy_test", accuracy_test, "test"),
            ("transfer_gap", transfer_gap, "all"),
            ("n_train", float(n_train), "train"),
            ("n_test", float(n_test), "test"),
        ]:
            metrics_writer.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=0,
                    split=split,  # type: ignore[arg-type]
                    metric_name=metric_name,
                    value=float(value),
                    timestamp=timestamp,
                )
            )

    # train_sample_ids accepted/ignored - kept here so future runners can
    # extend per-train-sample logging without rewiring to_features_and_labels.
    _ = train_sample_ids

    return E8Result(
        accuracy_train=accuracy_train,
        accuracy_test=accuracy_test,
        transfer_gap=transfer_gap,
        n_train=n_train,
        n_test=n_test,
    )
