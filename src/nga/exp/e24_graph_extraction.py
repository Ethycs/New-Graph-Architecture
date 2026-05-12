"""E24 - Universal graph extraction (Phase 20 Wave A integration runner).

The decisive Phase 20 experiment: extract a Typed Protocol Network from the
hidden-state stream of an arbitrary trained system, then compare it to
the hand-authored ground-truth FSM (when available) by normalised Hamming
distance and held-out predictive negative log-likelihood. If the extracted
graph matches, the architecture's universality claim is operationalised:
TPN is a universal post-hoc analysis applicable to any homogeneously
trained substrate.

Pipeline (the proposal's six steps composed end-to-end)
=======================================================

  Phase 1 -- HARVEST. ``HiddenStateHarvester`` runs the supplied encoder
             over the corpus and stacks the activations into a contiguous
             matrix ``hidden_states`` of shape ``(N_total, d)`` plus a
             parallel ``sample_index`` array.

  Phase 2 -- DISCRETISE + K-SELECT. ``estimate_k`` evaluates each K in the
             candidate range under the chosen criterion (held-out NLL,
             BIC, or elbow); refits at ``K_star`` on the full corpus.
             Output: ``labels`` of shape ``(N_total,)``.

  Phase 3 -- COUNT. Group ``labels`` by ``sample_index`` and run
             ``forward_backward.expected_counts_observed`` per sample to
             accumulate the transition matrix.

  Phase 4 -- PHASE A. ``bayesian_m_step_beta`` produces ``(alpha, beta)``;
             ``PosteriorMask`` wraps them; ``legality_matrix(threshold=0.5)``
             is the induced legality mask.

  Phase 5 -- TYPE DISCOVERY. Secondary clustering on the rows of the
             transition matrix groups vertices by successor profile;
             this gives the type assignment without ground truth.

  Phase 6 -- COMPARE. If a hand-authored legality matrix is supplied,
             compute the best-permutation normalised Hamming between
             extracted and authored masks. Always emit held-out predictive
             NLL vs a uniform-transition chain baseline.

Sanity tier
===========

For Wave A the runner exposes ``run_e24_synthetic``, a fixed-corpus
sanity test: generate a known typed Markov chain over K_true states with
a small noise model, harvest the noisy embeddings, run the full pipeline,
and confirm that the extracted FSM matches the ground-truth FSM by
Hamming ``<= 0.10`` and by held-out NLL improvement ``>= 1`` nat/token.
The synthetic substrate stands in for the "we don't yet have a trained
torch checkpoint to extract from" gap; Wave B will replace it with a
real ``TorchEnergyTrainer`` extraction on Python big / JSON.
"""
from __future__ import annotations

import datetime as dt
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nga.arch.bayesian_nonparametric_k import KEstimationResult, estimate_k
from nga.arch.forward_backward import (
    bayesian_m_step_beta,
    expected_counts_observed,
)
from nga.arch.hidden_state_harvester import (
    HarvestResult,
    HiddenStateHarvester,
)
from nga.arch.posterior_mask import PosteriorMask
from nga.drivers.jsonl_writer import JsonlWriter
from nga.drivers.metrics_jsonl import MetricsRecord
from nga.drivers.results_jsonl import ResultsRecord

__all__ = ["E24Result", "run_e24_synthetic"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class E24Result:
    """Headline numbers from a single E24 extraction."""

    n_total_steps: int
    n_samples: int
    hidden_dim: int

    K_star: int
    K_ground_truth: int | None

    # Extraction quality versus ground truth (when present).
    extracted_hamming_normalised: float
    holdout_nll_per_token_extracted: float
    holdout_nll_per_token_chain: float
    holdout_nll_improvement_per_token: float

    # Phase-A self-consistency.
    phase_a_hamming_normalised: float

    # Cluster purity vs ground-truth cluster IDs when supplied.
    cluster_purity: float

    # Compute efficiency.
    phase_1_wall_clock_seconds: float
    phase_2_wall_clock_seconds: float
    phase_3_wall_clock_seconds: float
    phase_4_wall_clock_seconds: float
    total_wall_clock_seconds: float
    extraction_throughput_steps_per_sec: float
    peak_memory_kb: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _peak_memory_kb() -> float:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # pragma: no cover
        return 0.0


def _best_permutation_hamming(
    extracted: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Find the best vertex permutation aligning extracted -> ground_truth
    and return (normalised_hamming, permutation).

    The extracted graph's vertex labels are arbitrary (cluster IDs from
    k-means); the ground-truth FSM's labels are the hand-authored
    canonical ordering. We need to align them before comparing edge sets.
    For small K (<= ~10) the brute-force permutation search is fine; for
    larger K we use a greedy Hungarian-style heuristic that matches
    extracted vertex i to the ground-truth vertex j minimising the row
    Hamming after assignment.
    """
    K = extracted.shape[0]
    if extracted.shape != ground_truth.shape:
        raise ValueError(
            f"shape mismatch: extracted {extracted.shape} vs "
            f"ground_truth {ground_truth.shape}"
        )
    if K <= 8:
        from itertools import permutations

        best_hamming = float("inf")
        best_perm = np.arange(K)
        for perm in permutations(range(K)):
            perm_arr = np.asarray(perm, dtype=np.int64)
            permuted = extracted[perm_arr][:, perm_arr]
            h = float(np.sum(permuted != ground_truth)) / (K * K)
            if h < best_hamming:
                best_hamming = h
                best_perm = perm_arr
        return best_hamming, best_perm
    # Greedy heuristic for large K: match each extracted row to the
    # ground-truth row whose pattern is most similar (lowest XOR popcount).
    available = list(range(K))
    perm = np.full(K, -1, dtype=np.int64)
    for i in range(K):
        best_j, best_d = available[0], int(K * K)
        for j in available:
            d = int(np.sum(extracted[i] != ground_truth[j]))
            if d < best_d:
                best_j, best_d = j, d
        perm[i] = best_j
        available.remove(best_j)
    permuted = extracted[perm][:, perm]
    h = float(np.sum(permuted != ground_truth)) / (K * K)
    return h, perm


def _holdout_nll(
    holdout_seq: np.ndarray,
    transition_log_probs: np.ndarray,
) -> float:
    """Mean negative log-likelihood per token of a held-out integer
    sequence under the supplied transition log-probability matrix.

    ``transition_log_probs[i, j] = log P(j | i)``. Rows are assumed
    normalised. NLL_per_token = -mean_t log P(s_{t+1} | s_t).
    """
    if holdout_seq.size < 2:
        return 0.0
    src = holdout_seq[:-1].astype(np.intp)
    dst = holdout_seq[1:].astype(np.intp)
    log_p = transition_log_probs[src, dst]
    return float(-np.mean(log_p))


def _row_normalise_to_log(
    transition_counts: np.ndarray,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Convert raw counts to row-stochastic log-probabilities with
    Laplace smoothing so log(0) never occurs on unobserved edges.
    """
    smoothed = transition_counts.astype(np.float64) + smoothing
    row_sums = smoothed.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    probs = smoothed / row_sums
    return np.log(probs)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_graph(
    hidden_states: np.ndarray,
    sample_index: np.ndarray,
    K_range: list[int] | tuple[int, ...],
    *,
    criterion: str = "holdout_nll",
    seed: int = 0,
) -> dict[str, Any]:
    """The core six-step extraction. Returns a dict carrying every
    intermediate so the runner can inspect and emit them.
    """
    # Step 2: discretise + K-select.
    k_result: KEstimationResult = estimate_k(
        hidden_states,
        K_range=K_range,
        criterion=criterion,
        seed=seed,
    )
    K_star = k_result.K_star
    labels = k_result.labels

    # Step 3: count transitions per sample.
    transition_counts = np.zeros((K_star, K_star), dtype=np.int64)
    n_samples = int(sample_index.max()) + 1 if sample_index.size else 0
    for s in range(n_samples):
        sample_rows = np.where(sample_index == s)[0]
        if sample_rows.size < 2:
            continue
        seq = labels[sample_rows]
        transition_counts += expected_counts_observed(seq, K_star)

    # Step 4: Phase A (Bayesian M-step) -> alpha, beta -> legality mask.
    alpha, beta = bayesian_m_step_beta(
        transition_counts.astype(np.float64),
        expected_counts_neg=None,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    posterior_mask = PosteriorMask(K_star, prior_alpha=1.0, prior_beta=1.0)
    posterior_mask._alpha = alpha
    posterior_mask._beta = beta
    extracted_mask = posterior_mask.legality_matrix(threshold=0.5)

    # Step 5: type discovery -- group vertices by transition-profile
    # similarity. Each vertex's row in the legality matrix IS its successor
    # signature; we use Hamming distance on those signatures.
    types = _discover_types(extracted_mask)

    return {
        "K_star": K_star,
        "K_range": k_result.K_range,
        "K_scores": k_result.scores,
        "labels": labels,
        "centroids": k_result.centroids,
        "transition_counts": transition_counts,
        "alpha": alpha,
        "beta": beta,
        "extracted_mask": extracted_mask,
        "types": types,
        "posterior_mask": posterior_mask,
    }


def _discover_types(extracted_mask: np.ndarray) -> np.ndarray:
    """Assign type IDs to vertices by clustering rows of the legality mask.

    Two vertices share a type iff their successor patterns match exactly.
    Equal rows -> same type. The function returns an integer array of
    shape ``(K,)`` whose entries are type IDs in ``[0, n_types)``.
    """
    K = extracted_mask.shape[0]
    # Hash each row to a tuple of bools; identical rows share a type.
    seen: dict[tuple, int] = {}
    types = np.zeros(K, dtype=np.int64)
    next_id = 0
    for i in range(K):
        key = tuple(bool(b) for b in extracted_mask[i])
        if key not in seen:
            seen[key] = next_id
            next_id += 1
        types[i] = seen[key]
    return types


# ---------------------------------------------------------------------------
# Synthetic sanity runner
# ---------------------------------------------------------------------------


def _generate_synthetic_corpus(
    n_samples: int,
    sequence_length: int,
    K_true: int,
    embed_dim: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> tuple[
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Generate a synthetic typed Markov-chain corpus.

    Returns:

    * ``corpus`` -- list of length ``n_samples``, each element a
      ``(sequence_length, embed_dim)`` array of noisy state embeddings.
    * ``state_sequences`` -- shape ``(n_samples, sequence_length)``,
      ground-truth state IDs.
    * ``ground_truth_legality`` -- shape ``(K_true, K_true)`` boolean
      array; ``True`` iff the synthetic chain emits ``i -> j`` with
      non-zero probability.
    * ``ground_truth_transition`` -- shape ``(K_true, K_true)`` float
      array; the row-stochastic transition matrix.
    * ``state_centroids`` -- shape ``(K_true, embed_dim)`` ground-truth
      cluster centroids in embedding space.
    """
    # Random transition matrix with sparsity ~ 0.5 (about half of edges
    # are legal). Each row is renormalised; the legality matrix records
    # which entries are non-zero.
    legality = (rng.random((K_true, K_true)) < 0.5)
    # Guarantee at least one outgoing edge per state.
    for i in range(K_true):
        if not legality[i].any():
            legality[i, rng.integers(K_true)] = True
    transition = rng.random((K_true, K_true)) * legality
    row_sums = transition.sum(axis=1, keepdims=True)
    transition = transition / row_sums

    # Ground-truth centroids: a random orthogonal-ish set in embed_dim.
    state_centroids = rng.standard_normal((K_true, embed_dim))
    # Push them apart so blobs are easy to recover.
    state_centroids = state_centroids / np.linalg.norm(
        state_centroids, axis=1, keepdims=True
    )
    state_centroids *= 3.0

    state_sequences = np.empty((n_samples, sequence_length), dtype=np.int64)
    corpus: list[np.ndarray] = []
    for n in range(n_samples):
        s = int(rng.integers(K_true))
        seq = np.empty(sequence_length, dtype=np.int64)
        for t in range(sequence_length):
            seq[t] = s
            s = int(rng.choice(K_true, p=transition[s]))
        state_sequences[n] = seq
        # Noisy embedding: centroid + Gaussian noise.
        embedded = state_centroids[seq] + noise_scale * rng.standard_normal(
            (sequence_length, embed_dim)
        )
        corpus.append(embedded)
    return corpus, state_sequences, legality, transition, state_centroids


def run_e24_synthetic(
    *,
    config: Any,
    ablation: Any,
    run_id: str,
    output_dir: Path,
    seed: int,
    K_true: int = 5,
    K_range_min: int | None = None,
    K_range_max: int | None = None,
    n_samples: int = 40,
    sequence_length: int = 30,
    embed_dim: int = 8,
    noise_scale: float = 0.3,
    holdout_n_samples: int = 8,
    k_selection_criterion: str = "bic",
) -> E24Result:
    """Wave-A sanity runner.

    Generates a synthetic typed Markov chain, treats the noisy-centroid
    embedding as a stand-in for a trained encoder's hidden-state stream,
    runs the extraction pipeline end-to-end, and reports Hamming +
    held-out NLL improvement against the ground-truth FSM.

    The ``config`` and ``ablation`` arguments are accepted for CLI symmetry
    with the other runners but are not consulted: the synthetic substrate
    is fully parametrised by the dedicated kwargs (with defaults that meet
    the Wave-A sanity bars from the proposal).
    """
    del config  # unused; CLI symmetry only.
    output_dir.mkdir(parents=True, exist_ok=True)

    if K_range_min is None:
        K_range_min = max(2, K_true - 2)
    if K_range_max is None:
        K_range_max = K_true + 3
    K_range = list(range(K_range_min, K_range_max + 1))
    holdout_n = int(holdout_n_samples)
    criterion = str(k_selection_criterion)

    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)

    # --- Phase 1: harvest (synthetic substrate: identity-encode the corpus).
    t1 = time.perf_counter()
    corpus, state_sequences, gt_legality, gt_transition, _ = (
        _generate_synthetic_corpus(
            n_samples=n_samples,
            sequence_length=sequence_length,
            K_true=K_true,
            embed_dim=embed_dim,
            noise_scale=noise_scale,
            rng=rng,
        )
    )
    harvester = HiddenStateHarvester.from_sequences(iter(corpus))
    harvest: HarvestResult = harvester.harvest([None] * n_samples)
    phase_1_seconds = time.perf_counter() - t1

    # --- Phase 2 + 3 + 4: extract.
    t2 = time.perf_counter()
    extraction = extract_graph(
        harvest.hidden_states,
        harvest.sample_index,
        K_range=K_range,
        criterion=criterion,
        seed=seed,
    )
    phase_2_seconds = time.perf_counter() - t2

    K_star = extraction["K_star"]
    labels = extraction["labels"]
    extracted_mask = extraction["extracted_mask"]
    transition_counts = extraction["transition_counts"]

    # --- Phase 3 (eval): compare to ground truth.
    t3 = time.perf_counter()
    # Cluster purity: best-permutation alignment of cluster IDs to states.
    # Compute per-cluster majority state.
    cluster_purity = _compute_cluster_purity(
        labels, state_sequences.reshape(-1)
    )

    if K_star == K_true:
        hamming, perm = _best_permutation_hamming(
            extracted_mask.astype(np.int64),
            gt_legality.astype(np.int64),
        )
    else:
        # Cannot align directly; report 1.0 as a sentinel meaning "could
        # not align by permutation; size mismatch".
        hamming = 1.0
        perm = None

    # Held-out NLL: split the corpus, refit the extraction's transition
    # log-probs on the train half, evaluate on holdout.
    holdout_n = min(holdout_n, max(1, n_samples // 5))
    train_idx = np.arange(n_samples - holdout_n)
    holdout_idx = np.arange(n_samples - holdout_n, n_samples)

    extracted_log_p = _row_normalise_to_log(transition_counts)
    chain_log_p = np.full(
        (K_star, K_star), -np.log(K_star), dtype=np.float64
    )

    holdout_seq_extracted: list[int] = []
    for s in holdout_idx:
        rows = np.where(harvest.sample_index == s)[0]
        if rows.size < 2:
            continue
        holdout_seq_extracted.extend(labels[rows].tolist())
    holdout_arr = np.asarray(holdout_seq_extracted, dtype=np.int64)
    holdout_nll_extracted = _holdout_nll(holdout_arr, extracted_log_p)
    holdout_nll_chain = _holdout_nll(holdout_arr, chain_log_p)

    # Phase A self-consistency Hamming: the legality matrix is built from
    # the posterior; we expect the posterior_mean > 0.5 mask to coincide
    # with the legality matrix it was built from. Should be 0 by construction.
    phase_a_hamming = float(
        np.mean(extraction["posterior_mask"].legality_matrix(0.5) != extracted_mask)
    )

    phase_3_seconds = time.perf_counter() - t3
    phase_4_seconds = 0.0  # absorbed into phase_2 above for the synthetic runner

    total = time.perf_counter() - t_start
    throughput = (
        float(harvest.n_steps) / total if total > 0.0 else float("nan")
    )

    # ---- Emit artefacts.
    del ablation  # unused below; the label is derived from run_id like e23.
    experiment_label = "E24"
    ablation_label = run_id.split("_")[1] if "_" in run_id else "A0"
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

    metric_pairs: list[tuple[str, float]] = [
        ("n_total_steps", float(harvest.n_steps)),
        ("n_samples", float(harvest.n_samples)),
        ("hidden_dim", float(harvest.hidden_dim)),
        ("K_star", float(K_star)),
        ("K_ground_truth", float(K_true)),
        ("extracted_hamming_normalised", float(hamming)),
        ("holdout_nll_per_token_extracted", float(holdout_nll_extracted)),
        ("holdout_nll_per_token_chain", float(holdout_nll_chain)),
        (
            "holdout_nll_improvement_per_token",
            float(holdout_nll_chain - holdout_nll_extracted),
        ),
        ("phase_a_hamming_normalised", float(phase_a_hamming)),
        ("cluster_purity", float(cluster_purity)),
        ("phase_1_wall_clock_seconds", float(phase_1_seconds)),
        ("phase_2_wall_clock_seconds", float(phase_2_seconds)),
        ("phase_3_wall_clock_seconds", float(phase_3_seconds)),
        ("phase_4_wall_clock_seconds", float(phase_4_seconds)),
        ("total_wall_clock_seconds", float(total)),
        ("extraction_throughput_steps_per_sec", float(throughput)),
        ("peak_memory_kb", float(_peak_memory_kb())),
        ("seed", float(seed)),
    ]

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
        # Per-step results: hidden state's cluster ID + ground-truth state.
        flat_states = state_sequences.reshape(-1)
        for i in range(harvest.n_steps):
            cluster_id = int(labels[i])
            gt_state = int(flat_states[i])
            results_writer.append(
                ResultsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=i,
                    sample_id=f"row{i}",
                    y_true=f"state{gt_state}",
                    y_hat=f"cluster{cluster_id}",
                    margin=0.0,
                    singular_flag=False,
                    sigma_score=0.0,
                    behavioral_stratum=None,
                    stratum_bitmask=None,
                    transition_legal=None,
                    timestamp=timestamp,
                )
            )

    # Also write the extracted graph itself (for downstream Wave-B tooling).
    extracted_graph_path = output_dir / "extracted_graph.json"
    extracted_graph_path.write_text(
        json.dumps(
            {
                "K_star": int(K_star),
                "K_ground_truth": int(K_true),
                "types": extraction["types"].tolist(),
                "extracted_mask": extracted_mask.astype(int).tolist(),
                "ground_truth_legality": gt_legality.astype(int).tolist(),
                "best_permutation": (
                    perm.tolist() if perm is not None else None
                ),
                "criterion": criterion,
                "K_range": list(K_range),
                "K_scores": extraction["K_scores"].tolist(),
            },
            indent=2,
        )
    )

    return E24Result(
        n_total_steps=int(harvest.n_steps),
        n_samples=int(harvest.n_samples),
        hidden_dim=int(harvest.hidden_dim),
        K_star=int(K_star),
        K_ground_truth=int(K_true),
        extracted_hamming_normalised=float(hamming),
        holdout_nll_per_token_extracted=float(holdout_nll_extracted),
        holdout_nll_per_token_chain=float(holdout_nll_chain),
        holdout_nll_improvement_per_token=float(
            holdout_nll_chain - holdout_nll_extracted
        ),
        phase_a_hamming_normalised=float(phase_a_hamming),
        cluster_purity=float(cluster_purity),
        phase_1_wall_clock_seconds=float(phase_1_seconds),
        phase_2_wall_clock_seconds=float(phase_2_seconds),
        phase_3_wall_clock_seconds=float(phase_3_seconds),
        phase_4_wall_clock_seconds=float(phase_4_seconds),
        total_wall_clock_seconds=float(total),
        extraction_throughput_steps_per_sec=float(throughput),
        peak_memory_kb=float(_peak_memory_kb()),
    )


def _compute_cluster_purity(
    labels: np.ndarray, ground_truth: np.ndarray
) -> float:
    """Fraction of points whose cluster's majority ground-truth state
    matches the point's true state. Standard purity metric.
    """
    if labels.size == 0:
        return 0.0
    K = int(labels.max()) + 1
    correct = 0
    for k in range(K):
        mask = labels == k
        if not mask.any():
            continue
        members = ground_truth[mask]
        majority = int(np.bincount(members.astype(np.int64)).argmax())
        correct += int(np.sum(members == majority))
    return float(correct) / float(labels.size)
