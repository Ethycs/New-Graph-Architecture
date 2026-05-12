"""E27 - Graph extraction with a partition-function probe (Phase 22a).

Where Wave C (E26) trained a state-conditioned MLP encoder on
next-state prediction alone -- and found that the encoder's natural
equivalence classes are ``(state, token-context)`` tuples rather than
FSM states -- E27 adds a **partition-function probe head** alongside
the next-state head. The probe predicts the gold partition function's
successor distribution given the encoder's hidden state, and KL loss on
the probe is added to the cross-entropy on next-state.

Architectural target
====================

The probe's target distribution depends only on ``current_state`` (the
partition function is a function of the FSM state, not of the observed
token); two steps with the same current_state but different tokens
share the same target. Gradient flow through the probe therefore
pushes the encoder to map ``(state, token_a)`` and ``(state, token_b)``
to hidden states that **agree on partition output** even when they
disagree on next-state prediction. This is the marginalisation the
Phase-21 bisimulation quotient could not perform empirically.

If the probe coupling works as designed, the encoder's natural
equivalence collapses from ``(state, token)`` toward ``state`` alone,
cluster purity at K = V approaches the oracle, and Hamming approaches
0.026 (the Phase 21 oracle floor).

Partition target (supervised, Phase 22a)
========================================

The cleanest first target is the uniform distribution over legal
successors under the gold legality matrix:

    P_Z(next = j | state = s) = 1[L[s, j]] / sum_k L[s, k]

This is the maximum-entropy distribution under the legality constraint
and uses only the FSM legality (no observed-transition statistics, no
encoder-learnable normaliser). Phase 22b will replace this with a
self-supervised target learned from trajectory consistency.

Comparisons
===========

The runner reports the standard E26 metrics plus a partition-probe
training-time fidelity score (probe KL on a held-out split) so we can
tell whether the probe actually learned the target. The sweep script
compares against:

  - Wave C free-K (no probe)
  - Wave C forced K = V (no probe)
  - oracle (ground-truth state labels)
  - **Phase 22a forced K = V (with probe)** -- the new headline
"""
from __future__ import annotations

import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only on torch-less envs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from nga.arch.graph_fsm import GraphFSM
from nga.exp.e24_graph_extraction import (
    _best_permutation_hamming,
    _compute_cluster_purity,
    _holdout_nll,
    _row_normalise_to_log,
    extract_graph,
)
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH

__all__ = [
    "E27Result",
    "run_e27",
    "_TrainableEncoderWithProbe",
    "_compute_partition_targets",
]


@dataclass
class E27Result:
    """Headline numbers from a single E27 extraction."""

    grammar: str
    n_total_steps: int
    n_samples: int
    hidden_dim: int

    V_ground_truth: int
    K_star: int

    extracted_hamming_normalised: float
    cluster_purity: float
    holdout_nll_per_token_extracted: float
    holdout_nll_per_token_chain: float
    holdout_nll_improvement_per_token: float

    encoder_train_accuracy: float
    encoder_train_final_loss: float
    encoder_train_final_probe_kl: float
    probe_weight: float

    phase_1_wall_clock_seconds: float
    phase_2_wall_clock_seconds: float
    phase_3_wall_clock_seconds: float
    total_wall_clock_seconds: float
    extraction_throughput_steps_per_sec: float
    peak_memory_kb: float


def _peak_memory_kb() -> float:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # pragma: no cover
        return 0.0


class _TrainableEncoderWithProbe(nn.Module if nn is not None else object):
    """Two-hidden-layer MLP + next-state head + partition-function probe."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_states: int,
        seed: int,
    ) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required")
        super().__init__()
        torch.manual_seed(int(seed))
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.next_state_head = nn.Linear(hidden_dim, n_states)
        # Partition probe: predict log P_Z(next | state) from h.
        self.partition_probe = nn.Linear(hidden_dim, n_states)

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        return h2

    def forward(
        self, x: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        h = self.encode(x)
        return self.next_state_head(h), self.partition_probe(h)


def _compute_partition_targets(
    current_states: np.ndarray,
    gold_legality: np.ndarray,
) -> np.ndarray:
    """Per-step partition-function target distribution.

    For each step ``i`` with current_state ``s``, the target is the
    uniform distribution over the legal successors of ``s`` under the
    gold legality matrix. Shape: ``(N, V)``.
    """
    V = gold_legality.shape[0]
    legal_per_state = gold_legality.astype(np.float64)  # (V, V)
    row_sums = legal_per_state.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    per_state_target = legal_per_state / row_sums  # (V, V), row-stochastic
    return per_state_target[current_states.astype(np.int64)]  # (N, V)


def _train_encoder_with_probe(
    encoder: _TrainableEncoderWithProbe,
    X: np.ndarray,
    y: np.ndarray,
    partition_targets: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    probe_weight: float,
    seed: int,
) -> tuple[float, float, float]:
    """Train the probe-augmented encoder. Returns (final_loss, train_acc,
    final_probe_kl)."""
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError("torch is required")
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    ce_fn = nn.CrossEntropyLoss()
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    pt_t = torch.from_numpy(partition_targets.astype(np.float32))
    n = X_t.shape[0]
    rng = np.random.default_rng(int(seed))
    final_loss = float("nan")
    final_probe_kl = float("nan")
    for _epoch in range(int(epochs)):
        perm = rng.permutation(n)
        idx = torch.from_numpy(perm.astype(np.int64))
        Xp = X_t[idx]
        yp = y_t[idx]
        ptp = pt_t[idx]
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            opt.zero_grad()
            next_logits, probe_logits = encoder(Xp[start:stop])
            loss_next = ce_fn(next_logits, yp[start:stop])
            log_probe = torch.log_softmax(probe_logits, dim=-1)
            # KL(target || log_probe): per-row sum(target * (log target - log probe))
            # Equivalent to -sum(target * log_probe) + const(target).
            target = ptp[start:stop]
            probe_kl = -(target * log_probe).sum(dim=-1).mean()
            loss = loss_next + probe_weight * probe_kl
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
            final_probe_kl = float(probe_kl.item())
    encoder.eval()
    with torch.no_grad():
        next_logits, _ = encoder(X_t)
        preds = next_logits.argmax(dim=1)
        acc = float((preds == y_t).float().mean().item())
    return final_loss, acc, final_probe_kl


def run_e27(
    *,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    grammar: str = "python_big",
    n_programs: int | None = None,
    hidden_dim: int = 32,
    n_train_epochs: int = 50,
    encoder_lr: float = 5e-3,
    batch_size: int = 64,
    probe_weight: float = 1.0,
    k_selection_criterion: str = "bic",
    K_range_pad: int = 5,
    holdout_fraction: float = 0.2,
    force_K_equals_V: bool = False,
) -> E27Result:
    """Train the probe-augmented encoder and extract from its mid-layer."""
    if torch is None:
        raise ImportError("torch is required for E27")
    if grammar not in GRAMMAR_DISPATCH:
        raise ValueError(
            f"unknown grammar {grammar!r}; expected one of "
            f"{sorted(GRAMMAR_DISPATCH.keys())}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dispatch = GRAMMAR_DISPATCH[grammar]
    V = fsm.vertex_count
    if force_K_equals_V:
        K_range = [V]
    else:
        K_min = max(2, V - K_range_pad)
        K_max = V + K_range_pad
        K_range = list(range(K_min, K_max + 1))

    t_start = time.perf_counter()
    t1 = time.perf_counter()
    n = int(n_programs) if n_programs is not None else int(dispatch["default_n"])
    train_ds = dispatch["loader"](fsm, n, seed)
    input_dim = int(train_ds.X.shape[1])
    seq_attr = dispatch["sequence_id_attr"]
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in train_ds.samples], dtype=np.int64
    )
    current_states = train_ds.current_states.astype(np.int64)

    # State-conditioned input.
    state_onehot = np.zeros((train_ds.X.shape[0], V), dtype=np.float64)
    state_onehot[np.arange(train_ds.X.shape[0]), current_states] = 1.0
    X_conditioned = np.concatenate(
        [train_ds.X.astype(np.float64), state_onehot], axis=1
    )

    # Partition targets (gold-legality, uniform over legal successors).
    partition_targets = _compute_partition_targets(
        current_states, fsm.legality_matrix
    )

    encoder = _TrainableEncoderWithProbe(
        input_dim=input_dim + V,
        hidden_dim=hidden_dim,
        n_states=V,
        seed=int(seed),
    )
    final_loss, train_acc, final_probe_kl = _train_encoder_with_probe(
        encoder,
        X_conditioned,
        train_ds.y_next,
        partition_targets,
        epochs=n_train_epochs,
        lr=encoder_lr,
        batch_size=batch_size,
        probe_weight=probe_weight,
        seed=int(seed),
    )

    # Harvest the trained mid-layer.
    X_t = torch.from_numpy(X_conditioned.astype(np.float32))
    with torch.no_grad():
        hidden = encoder.encode(X_t).cpu().numpy().astype(np.float64)
    phase_1_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    extraction = extract_graph(
        hidden, program_ids, K_range=K_range, criterion=k_selection_criterion,
        seed=seed,
    )
    phase_2_seconds = time.perf_counter() - t2

    K_star = int(extraction["K_star"])
    labels = extraction["labels"]
    extracted_mask = extraction["extracted_mask"]
    transition_counts = extraction["transition_counts"]
    gold_legality = fsm.legality_matrix

    t3 = time.perf_counter()
    if K_star == V:
        hamming, _perm = _best_permutation_hamming(
            extracted_mask.astype(np.int64),
            gold_legality.astype(np.int64),
        )
    else:
        hamming = 1.0
    cluster_purity = _compute_cluster_purity(labels, current_states)

    n_programs_total = int(program_ids.max()) + 1
    n_holdout = max(1, int(round(holdout_fraction * n_programs_total)))
    holdout_pids = set(range(n_programs_total - n_holdout, n_programs_total))
    train_counts = np.zeros((K_star, K_star), dtype=np.int64)
    holdout_seq: list[int] = []
    for p in range(n_programs_total):
        rows = np.where(program_ids == p)[0]
        if rows.size < 2:
            continue
        seq = labels[rows]
        if p in holdout_pids:
            holdout_seq.extend(seq.tolist())
        else:
            from nga.arch.forward_backward import expected_counts_observed

            train_counts += expected_counts_observed(seq, K_star)
    extracted_log_p = _row_normalise_to_log(train_counts)
    chain_log_p = np.full((K_star, K_star), -np.log(K_star), dtype=np.float64)
    holdout_arr = np.asarray(holdout_seq, dtype=np.int64)
    holdout_nll_extracted = _holdout_nll(holdout_arr, extracted_log_p)
    holdout_nll_chain = _holdout_nll(holdout_arr, chain_log_p)

    phase_3_seconds = time.perf_counter() - t3
    total = time.perf_counter() - t_start
    throughput = (
        float(hidden.shape[0]) / total if total > 0.0 else float("nan")
    )

    return E27Result(
        grammar=grammar,
        n_total_steps=int(hidden.shape[0]),
        n_samples=int(n_programs_total),
        hidden_dim=int(hidden_dim),
        V_ground_truth=int(V),
        K_star=int(K_star),
        extracted_hamming_normalised=float(hamming),
        cluster_purity=float(cluster_purity),
        holdout_nll_per_token_extracted=float(holdout_nll_extracted),
        holdout_nll_per_token_chain=float(holdout_nll_chain),
        holdout_nll_improvement_per_token=float(
            holdout_nll_chain - holdout_nll_extracted
        ),
        encoder_train_accuracy=float(train_acc),
        encoder_train_final_loss=float(final_loss),
        encoder_train_final_probe_kl=float(final_probe_kl),
        probe_weight=float(probe_weight),
        phase_1_wall_clock_seconds=float(phase_1_seconds),
        phase_2_wall_clock_seconds=float(phase_2_seconds),
        phase_3_wall_clock_seconds=float(phase_3_seconds),
        total_wall_clock_seconds=float(total),
        extraction_throughput_steps_per_sec=float(throughput),
        peak_memory_kb=float(_peak_memory_kb()),
    )
