"""E5 - IDF-weighting ablation experiment.

Trains a logistic-regression classifier on the synthetic-BabyAI grid dataset
(same data as E1) under two settings: with and without IDF feature reweighting.
The headline comparison is `accuracy_idf - accuracy_no_idf` (the IDF uplift).

This is an ABLATION runner: it focuses on aggregate accuracy rather than
per-step instrumentation, so it does NOT emit results.jsonl, scores.jsonl,
or decision_trace.jsonl. Only metrics.jsonl plus the four standard snapshot
artefacts (config_snapshot.yaml, ablation_snapshot.yaml, graph_fsm.yaml,
metrics.jsonl) are written by the surrounding CLI + this runner.

IDF formulation
---------------
For training feature matrix X of shape (N, D), the IDF weight per feature j
is::

    n_j         = number of training rows with X[:, j] active
    idf_weight_j = log(N / (1 + n_j))

A feature is "active" when it exceeds a small threshold (1e-6). The IDF-
reweighted feature matrix is::

    X_idf = X * (1 + alpha * idf_weight)

where `alpha = config.idf_alpha`. The plain (no-IDF) configuration trains on
X unchanged. Both classifiers are LogisticRegression with the same seed.

Reading docs/exp/e5-idf-ablation.md for the full spec.
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
from nga.exp.dataset_synthetic_babyai_grid import (
    generate_dataset,
    to_features_and_labels,
    train_test_split_by_trajectory,
)

__all__ = ["E5Result", "run_e5"]


# Canonical 7-vertex set for the synthetic BabyAI task graph (same as E1).
_EXPECTED_VERTEX_IDS: list[str] = [
    "Parse",
    "Navigate",
    "ResolveDoor",
    "Pickup",
    "Deliver",
    "Interact",
    "Done",
]


@dataclass
class E5Result:
    """Aggregate metrics returned by run_e5."""

    accuracy_idf: float
    accuracy_no_idf: float
    idf_uplift: float       # accuracy_idf - accuracy_no_idf
    n_samples: int          # n_test (held-out split)
    n_features: int


def _compute_idf_weights(X_train: np.ndarray) -> np.ndarray:
    """Return per-feature IDF weights for the training matrix.

    A feature is "active" on a row when |X[i, j]| > 1e-6. The IDF weight
    is::

        idf_j = log(N / (1 + n_j))

    where n_j is the count of rows on which feature j is active and N is the
    total number of training rows. The +1 in the denominator prevents
    divide-by-zero on dead features.

    Returns
    -------
    np.ndarray of shape (D,), dtype float64.
    """
    n_rows, n_features = X_train.shape
    active = np.abs(X_train) > 1e-6           # shape (N, D) bool
    n_samples_with_feature_active = active.sum(axis=0).astype(np.float64)
    idf = np.log(float(n_rows) / (1.0 + n_samples_with_feature_active))
    return idf.astype(np.float64)


def run_e5(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    n_trajectories: int = 80,
) -> E5Result:
    """Train + evaluate the synthetic-BabyAI classifier WITH and WITHOUT IDF.

    The classifier is sklearn LogisticRegression. Two models are fit on the
    same training split: one on plain features (X) and one on IDF-reweighted
    features (X * (1 + alpha * idf_weight)). Both are evaluated on the same
    held-out split. The headline metric is `idf_uplift`.

    Args:
        config: Merged runtime config (uses embedding_dim for feature_dim and
            idf_alpha as the reweighting strength).
        ablation: AblationTuple. The flag idf_weighting_enabled is recorded in
            ablation_snapshot.yaml by the CLI; this runner always trains both
            arms (with and without IDF) so the comparison is always available
            regardless of the ablation tuple.
        fsm: GraphFSM with vertex_ids matching the 7-vertex BabyAI set.
        run_id: Unique run identifier string (e.g. "E5_A0_seed42").
        output_dir: Directory in which to append metrics.jsonl.
        seed: Integer random seed for dataset generation, split, and classifier.
        n_trajectories: Number of synthetic trajectories to generate.

    Returns:
        E5Result with accuracy_idf, accuracy_no_idf, idf_uplift, n_samples,
        n_features.

    Raises:
        ValueError: If fsm.vertex_ids does not match the expected 7-vertex set.
    """
    del ablation  # ablation flags drive the CLI snapshot, not this runner

    # ------------------------------------------------------------------
    # Step 1 - Validate FSM vertex set
    # ------------------------------------------------------------------
    if fsm.vertex_ids != _EXPECTED_VERTEX_IDS:
        raise ValueError(
            f"E5 requires fsm.vertex_ids == {_EXPECTED_VERTEX_IDS!r}; "
            f"got {fsm.vertex_ids!r}"
        )

    # ------------------------------------------------------------------
    # Step 2 - Build the synthetic dataset (same generator as E1)
    # ------------------------------------------------------------------
    ds = generate_dataset(
        fsm,
        seed=seed,
        n_trajectories=n_trajectories,
        feature_dim=config.embedding_dim,
    )

    # ------------------------------------------------------------------
    # Step 3 - Split by trajectory (no temporal leakage)
    # ------------------------------------------------------------------
    train_ds, test_ds = train_test_split_by_trajectory(ds, seed=seed)

    # ------------------------------------------------------------------
    # Step 4 - Convert splits to arrays
    # ------------------------------------------------------------------
    X_train, y_train_idx, _train_prev, _train_ids = to_features_and_labels(train_ds)
    X_test, y_test_idx, _test_prev, _test_ids = to_features_and_labels(test_ds)

    y_train_str: list[str] = [fsm.vertex_ids[int(i)] for i in y_train_idx]
    y_test_str: list[str] = [fsm.vertex_ids[int(i)] for i in y_test_idx]

    n_test, n_features = X_test.shape

    # ------------------------------------------------------------------
    # Step 5 - Compute IDF weights from TRAINING data only (no test leakage)
    # ------------------------------------------------------------------
    idf_weights = _compute_idf_weights(X_train)             # shape (D,)
    alpha = float(config.idf_alpha)

    # IDF reweight is multiplicative on each feature column.
    feature_scaling = 1.0 + alpha * idf_weights              # shape (D,)
    X_train_idf = X_train * feature_scaling[np.newaxis, :]
    X_test_idf = X_test * feature_scaling[np.newaxis, :]

    # ------------------------------------------------------------------
    # Step 6 - Fit two classifiers (with / without IDF reweighting)
    # ------------------------------------------------------------------
    clf_idf = LogisticRegression(max_iter=2000, random_state=seed)
    clf_idf.fit(X_train_idf, y_train_str)

    clf_plain = LogisticRegression(max_iter=2000, random_state=seed)
    clf_plain.fit(X_train, y_train_str)

    # ------------------------------------------------------------------
    # Step 7 - Predict on held-out split
    # ------------------------------------------------------------------
    y_pred_idf = clf_idf.predict(X_test_idf)
    y_pred_plain = clf_plain.predict(X_test)

    y_test_arr = np.array(y_test_str)

    accuracy_idf = float(np.mean(y_test_arr == y_pred_idf))
    accuracy_no_idf = float(np.mean(y_test_arr == y_pred_plain))
    idf_uplift = accuracy_idf - accuracy_no_idf

    # ------------------------------------------------------------------
    # Step 8 - Write aggregate metric rows
    # ------------------------------------------------------------------
    experiment_label = "E5"
    ablation_label = run_id.split("_")[1]
    timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()

    with JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer:
        for metric_name, value in [
            ("accuracy_idf", accuracy_idf),
            ("accuracy_no_idf", accuracy_no_idf),
            ("idf_uplift", idf_uplift),
            ("n_samples", float(n_test)),
            ("n_features", float(n_features)),
            ("idf_alpha", alpha),
        ]:
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

    return E5Result(
        accuracy_idf=accuracy_idf,
        accuracy_no_idf=accuracy_no_idf,
        idf_uplift=idf_uplift,
        n_samples=n_test,
        n_features=n_features,
    )
