"""E3 - Hyperbolic vs Euclidean Embedding Sweep.

Sweeps over (dimension, embedding_kind) pairs, fits a nearest-prototype
classifier in that space on the synthetic-BabyAI dataset, and writes per-cell
accuracies. The Phase 3 acceptance criterion is

    accuracy_hyperbolic[d=8] >= accuracy_euclidean[d=16]

(resolves q01-hyperbolic-dim, informs q11-hyperbolic-vs-euclidean-tradeoff).

This runner intentionally uses the simpler "linear projection plus prototype"
pipeline rather than full neural-net training, because the load-bearing claim
is about the GEOMETRY (negative curvature buys hierarchy capacity), not about
any particular optimizer.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from nga.arch.gromov_hyperbolicity_diagnostic import estimate_gromov_delta
from nga.arch.graph_fsm import GraphFSM
from nga.arch.graph_prototype_vectors import (
    PrototypeBundle,
    fit_prototypes,
    init_prototypes_from_class_means,
)
from nga.arch.hyperbolic_distance_loss import predict_by_nearest_prototype
from nga.arch.hyperbolic_embedding import (
    embed_euclidean_to_poincare,
    poincare_distance,
)
from nga.drivers.ablation_flags import AblationTuple
from nga.drivers.config import Config
from nga.drivers.jsonl_writer import JsonlWriter
from nga.drivers.metrics_jsonl import MetricsRecord
from nga.drivers.results_jsonl import ResultsRecord
from nga.drivers.typed_score_record import TypedScoreRecord
from nga.exp.dataset_synthetic_babyai_grid import (
    generate_dataset,
    to_features_and_labels,
    train_test_split_by_trajectory,
)

__all__ = ["CellResult", "E3Result", "run_e3"]


@dataclass
class CellResult:
    dim: int
    kind: Literal["euclidean", "hyperbolic"]
    accuracy: float
    delta_max: float  # only meaningful for hyperbolic; 0.0 for euclidean
    n_test: int


@dataclass
class E3Result:
    cells: list[CellResult] = field(default_factory=list)
    hyperbolic_d8_acc: float = 0.0
    euclidean_d16_acc: float = 0.0
    hyperbolic_uplift: float = 0.0  # hyperbolic_d8_acc - euclidean_d16_acc
    n_test: int = 0


# Default sweep grid
DEFAULT_DIMS = (2, 4, 8, 16)
DEFAULT_KINDS: tuple[Literal["euclidean", "hyperbolic"], ...] = (
    "euclidean",
    "hyperbolic",
)

# Embedding scale used internally by init_prototypes_from_class_means.
# We use the same scale when embedding test features so prototypes and query
# points live at the same approximate radial depth in the Poincare ball.
_EMBED_SCALE = 0.5


def _init_prototypes_euclidean(
    *,
    fsm: GraphFSM,
    features: np.ndarray,  # shape (N, d), Euclidean d-space
    labels: list[str],     # length N; values from fsm.vertex_ids
    target_dim: int,
) -> np.ndarray:
    """Build per-vertex prototype centroids in Euclidean space (no Poincare projection).

    For any vertex with zero training samples the prototype is placed at the
    origin.  This is a local helper used only by the euclidean cell; hyperbolic
    cells use init_prototypes_from_class_means instead.

    Returns
    -------
    np.ndarray
        Shape (V, d) float array of per-vertex centroids in Euclidean d-space.
    """
    vertex_ids = fsm.vertex_ids
    vertex_index = fsm.vertex_index
    n_vertices = len(vertex_ids)

    sums = np.zeros((n_vertices, target_dim), dtype=float)
    counts = np.zeros(n_vertices, dtype=int)
    for feat, label in zip(features, labels):
        idx = vertex_index[label]
        sums[idx] += feat
        counts[idx] += 1

    prototypes = np.zeros((n_vertices, target_dim), dtype=float)
    for v_idx in range(n_vertices):
        if counts[v_idx] > 0:
            prototypes[v_idx] = sums[v_idx] / counts[v_idx]
        # else: leave at origin (zero vector)

    return prototypes


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax over a 2-D array.

    Parameters
    ----------
    x:
        Shape (N, V) array. Typically negative distances.
    temperature:
        Scaling factor.

    Returns
    -------
    np.ndarray
        Shape (N, V) array of row-normalised probabilities.
    """
    scaled = x / temperature
    # Subtract row max for numerical stability.
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_x = np.exp(scaled)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def run_e3(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    dims: tuple[int, ...] = DEFAULT_DIMS,
    kinds: tuple[Literal["euclidean", "hyperbolic"], ...] = DEFAULT_KINDS,
    n_trajectories: int = 60,
    n_fit_steps: int = 0,
    fit_lr: float = 0.05,
) -> E3Result:
    """Sweep (dim, kind) pairs; train a nearest-prototype classifier in each space.

    For each (dim, kind) pair:
      - euclidean: project to dim via orthonormal QR; classify by nearest Euclidean centroid.
      - hyperbolic: project to dim; embed in the Poincare ball via
        init_prototypes_from_class_means (which uses embed_euclidean_to_poincare
        internally at scale=0.5); optionally refine with Riemannian SGD;
        classify by nearest hyperbolic prototype.

    Note on n_fit_steps: the Riemannian SGD in fit_prototypes uses finite
    differences per sample per dimension, which is expensive.  The default of
    n_fit_steps=60 may exceed the 60-second runtime budget on large d; callers
    should set n_fit_steps=0 to skip fitting and use the (already good)
    initialisation from class means.  For this dataset the initialisation alone
    is competitive with the fitted result.

    Outputs:
      - output_dir/metrics.jsonl   per-cell accuracy rows, delta_max for hyperbolic
                                   cells, and three headline rows:
                                   hyperbolic_d8_acc, euclidean_d16_acc,
                                   hyperbolic_uplift.
      - output_dir/results.jsonl   per-sample results from the BEST cell only
                                   (highest accuracy across all cells).
      - output_dir/scores.jsonl    per-sample TypedScoreRecord from the best cell.

    sigma_score is set to 0.0 for all records because E3 does not run the
    singularity detector (that is E4's job).
    """
    max_dim = max(dims)
    experiment_label = "E3"
    ablation_label = run_id.split("_")[1]
    timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Step 1 - Build dataset once with the largest feature_dim so we can
    #          project down to each smaller dimension via a linear map.
    # ------------------------------------------------------------------
    ds = generate_dataset(
        fsm,
        seed=seed,
        n_trajectories=n_trajectories,
        feature_dim=max_dim,
    )

    # ------------------------------------------------------------------
    # Step 2 - Split by trajectory (no temporal leakage)
    # ------------------------------------------------------------------
    train_ds, test_ds = train_test_split_by_trajectory(ds, seed=seed)

    # ------------------------------------------------------------------
    # Step 3 - Convert splits to arrays
    # ------------------------------------------------------------------
    X_train_full, y_train_idx, _train_prev, _train_ids = to_features_and_labels(
        train_ds
    )
    X_test_full, y_test_idx, _test_prev, test_sample_ids = to_features_and_labels(
        test_ds
    )

    vertex_ids = fsm.vertex_ids
    y_train_labels: list[str] = [vertex_ids[int(i)] for i in y_train_idx]
    y_test_labels: list[str] = [vertex_ids[int(i)] for i in y_test_idx]
    n_test = len(y_test_idx)

    # ------------------------------------------------------------------
    # Step 4 - Compute per-class Euclidean centroids in the FULL feature space.
    # ------------------------------------------------------------------
    n_vertices = len(vertex_ids)
    vertex_index = fsm.vertex_index
    class_sums_full = np.zeros((n_vertices, max_dim), dtype=float)
    class_counts = np.zeros(n_vertices, dtype=int)
    for feat, label in zip(X_train_full, y_train_labels):
        idx = vertex_index[label]
        class_sums_full[idx] += feat
        class_counts[idx] += 1
    class_means_full = np.zeros((n_vertices, max_dim), dtype=float)
    for v_idx in range(n_vertices):
        if class_counts[v_idx] > 0:
            class_means_full[v_idx] = class_sums_full[v_idx] / class_counts[v_idx]

    # ------------------------------------------------------------------
    # Sweep over (dim, kind) pairs
    # ------------------------------------------------------------------
    cells: list[CellResult] = []

    # Track the best cell for writing per-sample records.
    best_accuracy = -1.0
    best_preds: np.ndarray | None = None
    best_dists: np.ndarray | None = None  # shape (n_test, n_vertices)
    best_kind: Literal["euclidean", "hyperbolic"] = "euclidean"
    best_dim: int = max_dim

    for dim in dims:
        # Build a deterministic orthonormal projection matrix P of shape (max_dim, dim).
        # Columns are orthonormal -> preserves more input geometry than a random matrix.
        rng_proj = np.random.default_rng(seed + 100 * dim)
        raw = rng_proj.normal(size=(max_dim, dim))
        # QR decomposition: Q has orthonormal columns, shape (max_dim, min(max_dim, dim)).
        Q, _ = np.linalg.qr(raw)
        P_d = Q[:, :dim]  # shape (max_dim, dim)

        # Project features and class means.
        X_train_d = X_train_full @ P_d      # shape (N_train, dim)
        X_test_d = X_test_full @ P_d        # shape (N_test, dim)
        class_means_d = class_means_full @ P_d  # shape (V, dim)

        for kind in kinds:
            if kind == "euclidean":
                # Euclidean cell: keep features as-is; prototypes are raw centroids.
                prototypes_e = _init_prototypes_euclidean(
                    fsm=fsm,
                    features=X_train_d,
                    labels=y_train_labels,
                    target_dim=dim,
                )  # shape (V, dim)

                # Distances shape: (n_test, n_vertices)
                diffs = X_test_d[:, None, :] - prototypes_e[None, :, :]  # (N, V, d)
                dists_e = np.linalg.norm(diffs, axis=-1)  # (N, V)
                preds_e = np.argmin(dists_e, axis=1).astype(int)  # (N,)

                accuracy = float(np.mean(preds_e == y_test_idx))
                delta_max = 0.0  # not computed for Euclidean

                cell = CellResult(
                    dim=dim,
                    kind="euclidean",
                    accuracy=accuracy,
                    delta_max=delta_max,
                    n_test=n_test,
                )
                cells.append(cell)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_preds = preds_e
                    best_dists = dists_e
                    best_kind = "euclidean"
                    best_dim = dim

            else:  # hyperbolic
                # Build prototypes using the Poincare-projected Euclidean class means.
                # init_prototypes_from_class_means expects raw (Euclidean) features; it
                # internally computes class centroids and projects them via
                # embed_euclidean_to_poincare(scale=0.5).  We pass X_train_d directly so
                # class centroid computation happens in projected Euclidean space before
                # the Poincare embedding step.
                bundle = init_prototypes_from_class_means(
                    fsm=fsm,
                    features=X_train_d,
                    labels=y_train_labels,
                    target_dim=dim,
                )

                # Optionally refine with Riemannian SGD.
                # NOTE: fit_prototypes uses per-sample finite differences which is
                # expensive (O(N * dim) distance evals per step).  With the default
                # n_fit_steps=60, fitting takes ~28s per d=8 cell - exceeding the
                # 60-second run budget.  For production use, set n_fit_steps=0.
                # When n_fit_steps > 0, train features must be in the Poincare ball.
                if n_fit_steps > 0:
                    X_train_h_for_fit = embed_euclidean_to_poincare(
                        X_train_d, scale=_EMBED_SCALE
                    )
                    bundle = fit_prototypes(
                        bundle,
                        features=X_train_h_for_fit,
                        labels=y_train_labels,
                        n_steps=n_fit_steps,
                        lr=fit_lr,
                        seed=seed,
                    )

                prototypes_h = bundle.prototypes  # shape (V, dim)

                # Embed test data at the same scale used by init_prototypes_from_class_means
                # internally so that test points and prototypes live at the same
                # approximate radial depth in the Poincare ball.
                X_test_h = embed_euclidean_to_poincare(X_test_d, scale=_EMBED_SCALE)

                # Predict by nearest Poincare-ball prototype.
                preds_h, _dists_nearest = predict_by_nearest_prototype(
                    X_test_h, prototypes_h
                )

                # Full distance matrix for softmax scoring: shape (N, V).
                dists_h_full = poincare_distance(X_test_h, prototypes_h)

                accuracy = float(np.mean(preds_h == y_test_idx))

                # Gromov delta on the prototype set (optional diagnostic).
                delta_est = estimate_gromov_delta(
                    prototypes_h, n_quadruples=200, seed=seed
                )
                delta_max = delta_est.delta_max

                cell = CellResult(
                    dim=dim,
                    kind="hyperbolic",
                    accuracy=accuracy,
                    delta_max=delta_max,
                    n_test=n_test,
                )
                cells.append(cell)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_preds = preds_h
                    best_dists = dists_h_full
                    best_kind = "hyperbolic"
                    best_dim = dim

    # ------------------------------------------------------------------
    # Step 6 - Aggregate headline numbers
    # ------------------------------------------------------------------
    # Locate the required cells: hyperbolic d=8 and euclidean d=16.
    hyp_d8_cells = [c for c in cells if c.kind == "hyperbolic" and c.dim == 8]
    euc_d16_cells = [c for c in cells if c.kind == "euclidean" and c.dim == 16]

    if not hyp_d8_cells:
        raise ValueError(
            "E3 requires a hyperbolic cell with dim=8 in the sweep grid, "
            f"but dims={dims} does not include 8."
        )
    if not euc_d16_cells:
        raise ValueError(
            "E3 requires a euclidean cell with dim=16 in the sweep grid, "
            f"but dims={dims} does not include 16."
        )

    hyperbolic_d8_acc = hyp_d8_cells[0].accuracy
    euclidean_d16_acc = euc_d16_cells[0].accuracy
    hyperbolic_uplift = hyperbolic_d8_acc - euclidean_d16_acc

    # ------------------------------------------------------------------
    # Steps 7+8 - Write outputs
    # ------------------------------------------------------------------
    assert best_preds is not None
    assert best_dists is not None

    # Softmax over negative distances to get per-class "probabilities".
    # Negating distances so smaller distance -> larger score.
    neg_dists = -best_dists  # (N, V)
    probs = _softmax(neg_dists, temperature=1.0)  # (N, V)

    with (
        JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer,
        JsonlWriter(output_dir / "results.jsonl", ResultsRecord) as results_writer,
        JsonlWriter(output_dir / "scores.jsonl", TypedScoreRecord) as scores_writer,
    ):
        # Per-cell accuracy metrics.
        for cell in cells:
            metric_name = f"accuracy_{cell.kind}_d{cell.dim}"
            metrics_writer.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=0,
                    split="test",
                    metric_name=metric_name,
                    value=cell.accuracy,
                    timestamp=timestamp,
                )
            )

        # Per-hyperbolic-cell Gromov delta.
        for cell in cells:
            if cell.kind == "hyperbolic":
                metrics_writer.append(
                    MetricsRecord(
                        run_id=run_id,
                        experiment=experiment_label,
                        ablation=ablation_label,
                        seed=seed,
                        step=0,
                        split="test",
                        metric_name=f"delta_max_hyperbolic_d{cell.dim}",
                        value=cell.delta_max,
                        timestamp=timestamp,
                    )
                )

        # Headline metrics.
        for metric_name, value in [
            ("hyperbolic_d8_acc", hyperbolic_d8_acc),
            ("euclidean_d16_acc", euclidean_d16_acc),
            ("hyperbolic_uplift", hyperbolic_uplift),
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

        # Per-sample records from the best cell only.
        for i in range(n_test):
            pred_idx = int(best_preds[i])
            y_true_i = y_test_labels[i]
            y_hat_i = vertex_ids[pred_idx]
            sample_id = test_sample_ids[i]

            # Confidence = softmax probability of predicted class.
            prob_row = probs[i]  # shape (V,)
            confidence = float(prob_row[pred_idx])

            # Margin = confidence - second-highest probability, clamped to [-1, 1].
            sorted_probs = np.sort(prob_row)[::-1]
            if len(sorted_probs) >= 2:
                margin = float(sorted_probs[0] - sorted_probs[1])
            else:
                margin = float(sorted_probs[0])
            margin = float(np.clip(margin, -1.0, 1.0))

            # sigma_score: E3 does not run the singularity detector; set to 0.0.
            sigma_score = 0.0

            results_writer.append(
                ResultsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=i,
                    sample_id=sample_id,
                    y_true=y_true_i,
                    y_hat=y_hat_i,
                    margin=margin,
                    singular_flag=False,
                    sigma_score=sigma_score,
                    behavioral_stratum=None,
                    stratum_bitmask=None,
                    transition_legal=None,
                    timestamp=timestamp,
                )
            )

            # TypedScoreRecord for the best cell.
            # dist must sum to 1.0 - re-normalise to guard against floating-point drift.
            dist_list = prob_row.tolist()
            dist_sum = sum(dist_list)
            dist_list = [v / dist_sum for v in dist_list]

            scores_writer.append(
                TypedScoreRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=i,
                    sample_id=sample_id,
                    predicted_state=y_hat_i,
                    stratum_label=y_hat_i,
                    confidence=confidence,
                    margin=margin,
                    dist=dist_list,
                    z_H=None,
                    embedding=None,
                    timestamp=timestamp,
                )
            )

    return E3Result(
        cells=cells,
        hyperbolic_d8_acc=hyperbolic_d8_acc,
        euclidean_d16_acc=euclidean_d16_acc,
        hyperbolic_uplift=hyperbolic_uplift,
        n_test=n_test,
    )
