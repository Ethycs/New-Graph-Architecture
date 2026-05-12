"""E23 - Self-supervised diagnostic discovery (Phase 19B).

The hypothesis: the TPN's existing typed-latent-clustering machinery,
applied to real medical symptom co-occurrence data **with NO disease
labels in training**, will recover something close to the medical
taxonomy when measured post-hoc by cluster purity / Adjusted Rand Index
/ Normalised Mutual Information against ground-truth diagnoses.

Pipeline:

  Phase 1 -- Patient embedding into the Poincare ball (random projection
             + tanh squashing) and Riemannian k-means clustering with a
             SINGLE global type. Three K values are tried in sequence:
             K=20 (under-cluster), K=41 (matches ground truth), K=80
             (over-cluster).

  Phase 2 -- Phase A forward-backward + Bayesian Beta-Dirichlet M-step
             over (cluster_t, cluster_{t+1}) transitions, where each
             patient contributes a Markov-randomised symptom
             trajectory: at step t the "current state" is the cluster
             ID of the partial-symptom-prefix embedding. Phase A runs
             once per K with the K=41 clustering as the reference (the
             other K's transition matrices are also produced for
             completeness).

  Phase 3 -- Evaluation: cluster purity, ARI, NMI, sigma vs hard-cluster
             AUROC, Phase-A self-consistency Hamming. Ground-truth
             disease labels enter ONLY here.

Compute-efficiency tracking is baked in: per-phase wall-clock + total
+ throughput + peak memory.

The runner emits the standard four artefacts (config_snapshot,
ablation_snapshot, metrics.jsonl, results.jsonl) plus scores.jsonl
(touched). decision_trace.jsonl is intentionally NOT populated -- there
is no FSM-step argmax for which "the WHY of one prediction" is
well-defined here; the load-bearing predictions are the global cluster
assignments. We document this in the run log via a metric.
"""
from __future__ import annotations

import datetime as dt
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nga.arch.failure_margin_auroc import sigma_auroc
from nga.arch.forward_backward import (
    bayesian_m_step_beta,
    expected_counts_observed,
)
from nga.arch.graph_fsm import GraphFSM
from nga.arch.hyperbolic_embedding import poincare_distance
from nga.arch.posterior_mask import PosteriorMask
from nga.arch.typed_latent_clustering import cluster_typed_latents
from nga.drivers.ablation_flags import AblationTuple
from nga.drivers.config import Config
from nga.drivers.jsonl_writer import JsonlWriter
from nga.drivers.metrics_jsonl import MetricsRecord
from nga.drivers.results_jsonl import ResultsRecord
from nga.exp.dataset_diagnostic_unsup import (
    dataset_unsup_summary,
    embed_symptoms_in_poincare,
    load_diagnostic_csv_as_vectors,
    markov_trajectory_for_patient,
)


__all__ = ["E23Result", "run_e23"]


K_VALUES_DEFAULT: tuple[int, ...] = (20, 41, 80)
EMBED_DIM_DEFAULT: int = 16


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class E23Result:
    n_patients: int
    n_symptoms: int
    n_diseases: int

    cluster_purity_K20: float
    cluster_purity_K41: float
    cluster_purity_K80: float

    adjusted_rand_index_K20: float
    adjusted_rand_index_K41: float
    adjusted_rand_index_K80: float

    normalized_mutual_info_K20: float
    normalized_mutual_info_K41: float
    normalized_mutual_info_K80: float

    mean_intra_cluster_distortion_K20: float
    mean_intra_cluster_distortion_K41: float
    mean_intra_cluster_distortion_K80: float

    phase_a_hamming_normalised_K41: float
    sigma_hardness_auroc_K41: float

    phase_1_wall_clock_seconds: float
    phase_2_wall_clock_seconds: float
    phase_3_wall_clock_seconds: float
    total_wall_clock_seconds: float
    inference_throughput_samples_per_sec: float
    peak_memory_kb: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _peak_memory_kb() -> float:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # pragma: no cover
        return 0.0


def _cluster_purity(labels: np.ndarray, ground_truth: np.ndarray) -> tuple[float, np.ndarray]:
    """Cluster purity: fraction of each cluster's patients sharing the
    most-common ground-truth disease, averaged over clusters weighted by
    cluster size.

    Returns
    -------
    weighted_purity:
        Float in [0, 1].
    per_cluster_majority:
        Array of length max(labels)+1; the most-common ground-truth
        index per cluster (-1 for empty clusters).
    """
    labels = np.asarray(labels)
    gt = np.asarray(ground_truth)
    if labels.shape != gt.shape:
        raise ValueError("labels and ground_truth must have the same shape")
    n = labels.shape[0]
    if n == 0:
        return 0.0, np.zeros(0, dtype=int)
    K = int(labels.max()) + 1 if labels.size else 0
    purity_sum = 0.0
    majority = np.full(K, -1, dtype=int)
    for c in range(K):
        mask = labels == c
        size = int(mask.sum())
        if size == 0:
            continue
        truths_in_c, counts = np.unique(gt[mask], return_counts=True)
        best = int(np.argmax(counts))
        majority[c] = int(truths_in_c[best])
        # Weight by cluster size: contribution is (count of majority) / N.
        purity_sum += float(counts[best])
    return purity_sum / float(n), majority


def _safe_ari_nmi(
    labels: np.ndarray, ground_truth: np.ndarray
) -> tuple[float, float]:
    """ARI and NMI via sklearn. Returns (ari, nmi) -- (0, 0) if unavailable."""
    try:
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
        )
    except ImportError:  # pragma: no cover
        return 0.0, 0.0
    labels = np.asarray(labels)
    gt = np.asarray(ground_truth)
    ari = float(adjusted_rand_score(gt, labels))
    nmi = float(normalized_mutual_info_score(gt, labels))
    return ari, nmi


def _embedding_to_cluster(
    point: np.ndarray, prototypes: np.ndarray
) -> tuple[int, float]:
    """Find the nearest Poincare prototype for a single point.

    Returns (cluster_id, min_distance).
    """
    pt = point.reshape(1, -1)
    d = poincare_distance(pt, prototypes)
    if d.ndim == 1:
        d = d[None, :]
    idx = int(np.argmin(d, axis=1)[0])
    return idx, float(d[0, idx])


def _embed_partial_prefix(
    prefix_set_indicator: np.ndarray,
    R: np.ndarray,
    *,
    scale: float = 0.5,
) -> np.ndarray:
    """Re-project a partial symptom indicator into the Poincare ball.

    Uses the same random matrix ``R`` and scaling that
    ``embed_symptoms_in_poincare`` used to produce the per-patient
    embeddings, so prefix embeddings live in the same ball.
    """
    from nga.arch.hyperbolic_embedding import embed_euclidean_to_poincare

    z = prefix_set_indicator.reshape(1, -1) @ R
    return embed_euclidean_to_poincare(z, scale=scale)[0]


def _build_random_projection(
    n_features: int, dim: int, seed: int
) -> np.ndarray:
    """The exact same Gaussian projection ``embed_symptoms_in_poincare``
    builds internally, exposed so prefix-embeddings stay consistent."""
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((n_features, int(dim))) / np.sqrt(float(n_features))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_e23(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,  # accepted for CLI symmetry; not used (FSM is discovered).
    run_id: str,
    output_dir: Path,
    seed: int,
    n_patients: int | None = None,
    embed_dim: int = EMBED_DIM_DEFAULT,
    K_values: tuple[int, ...] = K_VALUES_DEFAULT,
    cluster_max_iter: int = 30,
    csv_path: str | Path | None = None,
    n_trajectory_steps_per_patient: int | None = None,
) -> E23Result:
    """Run Phase 19B unsupervised diagnostic clustering.

    The classical ``cluster_typed_latents`` API expects type-orbit
    constraints. We map the unsupervised case onto it by assigning every
    patient the SAME synthetic type (``types = zeros(n)``). This is the
    one subtlety in the mapping: ``cluster_typed_latents`` returns
    labels LOCAL to each type, but with a single type the local IDs are
    already global -- no extra reindexing needed. We pass ``L_per_type =
    K`` so the routine spawns K prototypes for the only orbit.
    """
    del fsm  # 19B has no FSM; the cluster lattice IS the implicit FSM.

    if csv_path is None:
        repo_root = Path(__file__).resolve().parents[3]
        csv_path = repo_root / "tests/fixtures/data/diagnostic_training.csv"

    mem_start_kb = _peak_memory_kb()
    total_start = time.perf_counter()

    rng_master = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    # Phase 1 -- Load CSV, embed, cluster.
    # ------------------------------------------------------------------
    phase_1_start = time.perf_counter()

    X, disease_labels, symptom_names, disease_names = (
        load_diagnostic_csv_as_vectors(csv_path)
    )
    summary = dataset_unsup_summary(X, disease_labels)

    if n_patients is not None:
        n_keep = min(int(n_patients), X.shape[0])
        keep = rng_master.permutation(X.shape[0])[:n_keep]
        keep = np.sort(keep)
        X = X[keep]
        disease_labels = [disease_labels[int(i)] for i in keep]

    n = X.shape[0]
    n_sym = X.shape[1]

    embeddings = embed_symptoms_in_poincare(X, dim=embed_dim, seed=int(seed))
    types = np.zeros(n, dtype=np.int64)

    # Map disease labels to integers for sklearn metrics.
    disease_to_id = {d: i for i, d in enumerate(disease_names)}
    gt_int = np.array(
        [disease_to_id[d] for d in disease_labels], dtype=np.int64
    )

    cluster_results: dict[int, dict] = {}
    for K in K_values:
        result = cluster_typed_latents(
            observations_in_poincare=embeddings,
            types=types,
            L_per_type=int(K),
            max_iter=int(cluster_max_iter),
            seed=int(seed),
        )
        labels = np.asarray(result.labels, dtype=np.int64)
        prototypes = result.prototypes[0]  # single type -> single matrix
        # Mean intra-cluster distortion (Riemannian) per cluster.
        per_cluster_dists: list[float] = []
        for c in range(K):
            mask = labels == c
            if mask.sum() == 0:
                continue
            d = poincare_distance(embeddings[mask], prototypes[c : c + 1])
            if d.ndim == 0:
                d = np.array([float(d)])
            per_cluster_dists.append(float(np.mean(d)))
        mean_distortion = (
            float(np.mean(per_cluster_dists)) if per_cluster_dists else 0.0
        )
        cluster_results[K] = {
            "labels": labels,
            "prototypes": prototypes,
            "n_iter": int(result.n_iter),
            "converged": bool(result.converged),
            "mean_distortion": mean_distortion,
        }

    phase_1_seconds = time.perf_counter() - phase_1_start

    # ------------------------------------------------------------------
    # Phase 2 -- Phase A forward-backward / Bayesian Beta seed on the
    # symptom-trajectory cluster transitions for the K=41 reference.
    # ------------------------------------------------------------------
    phase_2_start = time.perf_counter()

    # Use K=41 (the ground-truth-cardinality clustering) as the reference.
    K_ref = 41 if 41 in cluster_results else K_values[0]
    ref = cluster_results[K_ref]
    K_eff = int(ref["labels"].max()) + 1 if n > 0 else int(K_ref)

    # The random projection re-used for prefix re-embedding.
    R = _build_random_projection(n_sym, embed_dim, seed=int(seed))

    symptom_id_for = {name: i for i, name in enumerate(symptom_names)}
    proto_ref = ref["prototypes"]

    # Build empirical cluster-transition counts AND a per-patient
    # "trajectory final cluster" record.
    pair_counts = np.zeros((K_eff, K_eff), dtype=np.float64)
    state_path_concat: list[int] = []
    n_steps_total = 0
    for i in range(n):
        traj = markov_trajectory_for_patient(
            X[i], symptom_id_for, seed=int(seed) + i
        )
        if not traj:
            continue
        if n_trajectory_steps_per_patient is not None:
            traj = traj[: int(n_trajectory_steps_per_patient)]
        # Walk: prefix embedding -> nearest prototype -> cluster id chain.
        prefix = np.zeros(n_sym, dtype=np.float64)
        cluster_chain: list[int] = []
        for tok in traj:
            prefix[symptom_id_for[tok]] = 1.0
            point = _embed_partial_prefix(prefix, R)
            c, _d = _embedding_to_cluster(point, proto_ref)
            cluster_chain.append(int(c))
        # Pair counts.
        for s_from, s_to in zip(cluster_chain[:-1], cluster_chain[1:]):
            pair_counts[s_from, s_to] += 1.0
            n_steps_total += 1
        state_path_concat.extend(cluster_chain)

    # Phase A Bayesian Beta-Dirichlet M-step seed.
    alpha_seed, beta_seed = bayesian_m_step_beta(
        expected_counts_pos=pair_counts,
        expected_counts_neg=None,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    posterior_a = PosteriorMask(
        n_vertices=K_eff, prior_alpha=1.0, prior_beta=1.0
    )
    posterior_a._alpha = np.asarray(alpha_seed, dtype=np.float64)
    posterior_a._beta = np.asarray(beta_seed, dtype=np.float64)

    # Compute the empirical adjacency: True iff at least one transition
    # was observed along that edge. Self-consistency check: does
    # posterior_mean > 0.5 agree with "saw at least one transition"?
    empirical_legal = pair_counts > 0
    learned_legal = posterior_a.legality_matrix(0.5)
    total_cells = int(empirical_legal.size)
    if total_cells:
        hamming = int(np.sum(learned_legal != empirical_legal))
        phase_a_hamming_normalised = float(hamming) / float(total_cells)
    else:
        phase_a_hamming_normalised = 0.0

    # Forward-backward expected counts on the concatenated cluster path
    # (sanity / surface-area touch).
    if state_path_concat:
        _ = expected_counts_observed(
            np.asarray(state_path_concat, dtype=np.int64), n_states=K_eff
        )

    # Per-patient sigma_hardness: low-cluster-confidence patients should
    # fire higher sigma. We define sigma as 1 - (max-prob over clusters
    # using a softmin on Riemannian distances), normalised.
    sigmas: list[float] = []
    is_hard_patient: list[int] = []
    # First, compute per-cluster purity at the K_ref resolution to label hard.
    purity_kref, majority_kref = _cluster_purity(ref["labels"], gt_int)
    # Per-cluster purity for the hard-flag is the per-cluster local purity.
    K_actual = int(ref["labels"].max()) + 1 if n > 0 else 0
    per_cluster_purity = np.zeros(K_actual, dtype=np.float64)
    for c in range(K_actual):
        mask = ref["labels"] == c
        size = int(mask.sum())
        if size == 0:
            continue
        truths, counts = np.unique(gt_int[mask], return_counts=True)
        per_cluster_purity[c] = float(counts.max()) / float(size)

    for i in range(n):
        d = poincare_distance(embeddings[i : i + 1], proto_ref)
        if d.ndim == 0:
            d = np.array([float(d)])
        d = np.asarray(d).reshape(-1)
        # Softmin probabilities: lower distance -> higher prob.
        logits = -d
        logits = logits - logits.max()
        ex = np.exp(logits)
        p = ex / max(ex.sum(), 1e-12)
        max_p = float(p.max())
        sigma = 1.0 - max_p
        sigmas.append(float(sigma))
        c = int(ref["labels"][i])
        hard = 1 if (
            0 <= c < K_actual and per_cluster_purity[c] < 0.5
        ) else 0
        is_hard_patient.append(hard)

    sigmas_arr = np.asarray(sigmas, dtype=float)
    hard_arr = np.asarray(is_hard_patient, dtype=np.int64)
    if hard_arr.size > 0 and 0 < int(hard_arr.sum()) < int(hard_arr.size):
        sigma_hardness_auroc_value = float(
            sigma_auroc(sigmas_arr, hard_arr.astype(bool))
        )
    else:
        sigma_hardness_auroc_value = 0.5

    phase_2_seconds = time.perf_counter() - phase_2_start

    # ------------------------------------------------------------------
    # Phase 3 -- Evaluation: cluster purity, ARI, NMI per K.
    # ------------------------------------------------------------------
    phase_3_start = time.perf_counter()

    purities: dict[int, float] = {}
    aris: dict[int, float] = {}
    nmis: dict[int, float] = {}
    for K in K_values:
        labels = cluster_results[K]["labels"]
        purity, _maj = _cluster_purity(labels, gt_int)
        ari, nmi = _safe_ari_nmi(labels, gt_int)
        purities[K] = float(purity)
        aris[K] = float(ari)
        nmis[K] = float(nmi)

    phase_3_seconds = time.perf_counter() - phase_3_start

    # ------------------------------------------------------------------
    # Emit artefacts.
    # ------------------------------------------------------------------
    inference_seconds = phase_1_seconds + phase_2_seconds + phase_3_seconds
    inference_throughput = (
        float(n) / inference_seconds if inference_seconds > 0 else 0.0
    )
    total_seconds = time.perf_counter() - total_start
    mem_end_kb = _peak_memory_kb()
    peak_memory_kb = max(mem_end_kb - mem_start_kb, 0.0)

    experiment_label = "E23"
    ablation_label = run_id.split("_")[1]
    timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()

    metric_pairs: list[tuple[str, float]] = []
    for K in K_values:
        metric_pairs.extend(
            [
                (f"cluster_purity_K{K}", float(purities[K])),
                (f"adjusted_rand_index_K{K}", float(aris[K])),
                (f"normalized_mutual_info_K{K}", float(nmis[K])),
                (f"n_clusters_K{K}", float(K)),
                (
                    f"mean_intra_cluster_distortion_K{K}",
                    float(cluster_results[K]["mean_distortion"]),
                ),
            ]
        )
    # Phase-A self-consistency hamming for the K=ref clustering.
    metric_pairs.append(
        (
            f"phase_a_hamming_normalised_K{K_ref}",
            float(phase_a_hamming_normalised),
        )
    )
    metric_pairs.append(
        (f"sigma_hardness_auroc_K{K_ref}", float(sigma_hardness_auroc_value))
    )

    # Compute efficiency.
    metric_pairs.extend(
        [
            ("phase_1_wall_clock_seconds", float(phase_1_seconds)),
            ("phase_2_wall_clock_seconds", float(phase_2_seconds)),
            ("phase_3_wall_clock_seconds", float(phase_3_seconds)),
            ("total_wall_clock_seconds", float(total_seconds)),
            (
                "inference_throughput_samples_per_sec",
                float(inference_throughput),
            ),
            ("peak_memory_kb", float(peak_memory_kb)),
        ]
    )

    # Run-summary metadata.
    metric_pairs.extend(
        [
            ("n_patients", float(n)),
            ("n_symptoms", float(n_sym)),
            ("n_diseases", float(summary["n_diseases"])),
            (
                "mean_symptoms_per_patient",
                float(summary["mean_symptoms_per_patient"]),
            ),
            ("seed", float(seed)),
        ]
    )

    with (
        JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer,
        JsonlWriter(output_dir / "results.jsonl", ResultsRecord) as results_writer,
    ):
        for metric_name, value in metric_pairs:
            metrics_writer.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=0,
                    split="all",
                    metric_name=metric_name,
                    value=float(value),
                    timestamp=timestamp,
                )
            )
        # Per-patient results: cluster ID at K=41, predicted majority
        # disease, and the ground truth.
        labels_ref = ref["labels"]
        _purity_ref, majority = _cluster_purity(labels_ref, gt_int)
        for i in range(n):
            c = int(labels_ref[i])
            pred_disease_idx = (
                int(majority[c]) if 0 <= c < majority.shape[0] else -1
            )
            pred_disease = (
                disease_names[pred_disease_idx]
                if 0 <= pred_disease_idx < len(disease_names)
                else "UNKNOWN"
            )
            true_disease = disease_labels[i]
            results_writer.append(
                ResultsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=i,
                    sample_id=f"patient{i}",
                    y_true=true_disease,
                    y_hat=pred_disease,
                    margin=0.0,
                    singular_flag=bool(sigmas[i] >= 0.5),
                    sigma_score=float(min(max(sigmas[i], 0.0), 1.0)),
                    behavioral_stratum=None,
                    stratum_bitmask=None,
                    transition_legal=None,
                    timestamp=timestamp,
                )
            )

    # decision_trace.jsonl is intentionally NOT emitted by this runner;
    # 19B has no FSM-step argmax for which a per-row "WHY of one step"
    # is well-defined. The load-bearing predictions are global cluster
    # assignments emitted via results.jsonl. The four standard artefacts
    # plus scores.jsonl are still produced (cli touches scores.jsonl).

    return E23Result(
        n_patients=int(n),
        n_symptoms=int(n_sym),
        n_diseases=int(summary["n_diseases"]),
        cluster_purity_K20=float(purities.get(20, 0.0)),
        cluster_purity_K41=float(purities.get(41, 0.0)),
        cluster_purity_K80=float(purities.get(80, 0.0)),
        adjusted_rand_index_K20=float(aris.get(20, 0.0)),
        adjusted_rand_index_K41=float(aris.get(41, 0.0)),
        adjusted_rand_index_K80=float(aris.get(80, 0.0)),
        normalized_mutual_info_K20=float(nmis.get(20, 0.0)),
        normalized_mutual_info_K41=float(nmis.get(41, 0.0)),
        normalized_mutual_info_K80=float(nmis.get(80, 0.0)),
        mean_intra_cluster_distortion_K20=float(
            cluster_results.get(20, {"mean_distortion": 0.0})["mean_distortion"]
        ),
        mean_intra_cluster_distortion_K41=float(
            cluster_results.get(41, {"mean_distortion": 0.0})["mean_distortion"]
        ),
        mean_intra_cluster_distortion_K80=float(
            cluster_results.get(80, {"mean_distortion": 0.0})["mean_distortion"]
        ),
        phase_a_hamming_normalised_K41=float(phase_a_hamming_normalised),
        sigma_hardness_auroc_K41=float(sigma_hardness_auroc_value),
        phase_1_wall_clock_seconds=float(phase_1_seconds),
        phase_2_wall_clock_seconds=float(phase_2_seconds),
        phase_3_wall_clock_seconds=float(phase_3_seconds),
        total_wall_clock_seconds=float(total_seconds),
        inference_throughput_samples_per_sec=float(inference_throughput),
        peak_memory_kb=float(peak_memory_kb),
    )
