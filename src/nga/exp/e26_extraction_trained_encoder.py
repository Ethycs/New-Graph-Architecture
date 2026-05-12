"""E26 - Universal graph extraction on a *trained-from-scratch* encoder
(Phase 20 Wave C, the decisive Tier-2 test).

Where Wave B (E25) extracted from a FrozenEncoderTorch substrate -- and
found that the substrate's clustering geometry is fixed at seed init
regardless of Phase-B training -- E26 lets the encoder itself adapt to
the corpus. A small MLP is trained from scratch on next-FSM-state
prediction; its trained mid-layer activations become the substrate for
extraction. If this brings normalised Hamming under 0.05 on any
grammar, the proposal's universality claim earns its keep at the
trained-substrate level: a homogeneously trained network's hidden states
*do* carry the typed graph, and the extraction pipeline can recover it.

Substrate
=========

A two-hidden-layer MLP (``Linear(input_dim, hidden_dim) -> ReLU ->
Linear(hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, n_states)``)
trained on the next-state prediction task via cross-entropy. The
encoder is fully trainable (no ``requires_grad=False`` anywhere). The
harvest is the **second hidden layer's pre-output ReLU**, which is the
trained representation immediately preceding the classifier head -- the
architecture's notion of "the model's internal state at this step."

State-conditioned input
=======================

A parser's next-state distribution is genuinely conditional on
(current_state, token); a per-step classifier without state context can
only do chance-level work on most grammars. We therefore concatenate a
one-hot encoding of ``current_state`` to the token feature vector before
passing it to the encoder. The encoder's input dimension is therefore
``input_dim = feature_dim + V`` where ``V = fsm.vertex_count``. This is
the same per-state-conditioning that TPN's ``TypedReadoutTorch``
implements with per-type heads -- here flattened into a single
state-conditioned MLP so the harvest is one consistent hidden
representation across all current_state values.

Comparison
==========

We rerun the same `extract_graph` + best-permutation Hamming pipeline
that E25 used. Each grammar's result is directly comparable to its
Wave-B counterpart:

  - K_star recovery (does training shift K_star toward V?)
  - cluster_purity (is the trained substrate more discriminative?)
  - extracted_hamming_normalised (does it cross the 0.05 strict bar?)
  - held-out NLL improvement (extracted vs uniform chain)

The same dispatch table as E25 covers all 5 grammars.
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

__all__ = ["E26Result", "run_e26"]


@dataclass
class E26Result:
    """Headline numbers from a single E26 extraction."""

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


class _TrainableEncoder(nn.Module if nn is not None else object):
    """Two-hidden-layer MLP trained on next-state prediction.

    Layers:
      Linear(input_dim, hidden_dim) -> ReLU
      Linear(hidden_dim, hidden_dim) -> ReLU  <-- harvested
      Linear(hidden_dim, n_states)
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_states: int, seed: int):
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for the trained encoder")
        super().__init__()
        torch.manual_seed(int(seed))
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_states)

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return the trained hidden-state representation."""
        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        return h2

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.head(self.encode(x))


def _train_encoder(
    encoder: _TrainableEncoder,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[float, float]:
    """Train ``encoder`` on next-state prediction via cross-entropy.

    Returns ``(final_loss, train_accuracy)``.
    """
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    n = X_t.shape[0]
    rng = np.random.default_rng(int(seed))
    final_loss = float("nan")
    for _epoch in range(int(epochs)):
        perm = rng.permutation(n)
        idx = torch.from_numpy(perm.astype(np.int64))
        Xp = X_t[idx]
        yp = y_t[idx]
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            opt.zero_grad()
            logits = encoder(Xp[start:stop])
            loss = loss_fn(logits, yp[start:stop])
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
    encoder.eval()
    with torch.no_grad():
        preds = encoder(X_t).argmax(dim=1)
        acc = float((preds == y_t).float().mean().item())
    return final_loss, acc


def run_e26(
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
    k_selection_criterion: str = "bic",
    K_range_pad: int = 5,
    holdout_fraction: float = 0.2,
    force_K_equals_V: bool = False,
) -> E26Result:
    """Train a small MLP on next-state prediction and run extraction on
    its trained mid-layer activations."""
    if torch is None:
        raise ImportError("torch is required for E26")
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

    # ---- Phase 1: dataset + train + harvest.
    t1 = time.perf_counter()
    n = int(n_programs) if n_programs is not None else int(dispatch["default_n"])
    train_ds = dispatch["loader"](fsm, n, seed)
    input_dim = int(train_ds.X.shape[1])
    seq_attr = dispatch["sequence_id_attr"]
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in train_ds.samples], dtype=np.int64
    )

    # State-condition the input: concatenate one-hot(current_state) so
    # the encoder can learn next-state prediction from (state, token).
    current_states = train_ds.current_states.astype(np.int64)
    state_onehot = np.zeros((train_ds.X.shape[0], V), dtype=np.float64)
    state_onehot[np.arange(train_ds.X.shape[0]), current_states] = 1.0
    X_conditioned = np.concatenate(
        [train_ds.X.astype(np.float64), state_onehot], axis=1
    )
    conditioned_input_dim = input_dim + V

    encoder = _TrainableEncoder(
        input_dim=conditioned_input_dim,
        hidden_dim=hidden_dim,
        n_states=V,
        seed=int(seed),
    )
    final_loss, train_acc = _train_encoder(
        encoder,
        X_conditioned,
        train_ds.y_next,
        epochs=n_train_epochs,
        lr=encoder_lr,
        batch_size=batch_size,
        seed=int(seed),
    )

    # Harvest the trained second hidden layer.
    X_t = torch.from_numpy(X_conditioned.astype(np.float32))
    with torch.no_grad():
        hidden = encoder.encode(X_t).cpu().numpy().astype(np.float64)
    phase_1_seconds = time.perf_counter() - t1

    # ---- Phase 2: extract.
    t2 = time.perf_counter()
    extraction = extract_graph(
        hidden,
        program_ids,
        K_range=K_range,
        criterion=k_selection_criterion,
        seed=seed,
    )
    phase_2_seconds = time.perf_counter() - t2

    K_star = int(extraction["K_star"])
    labels = extraction["labels"]
    extracted_mask = extraction["extracted_mask"]
    transition_counts = extraction["transition_counts"]
    gold_legality = fsm.legality_matrix

    # ---- Phase 3: compare.
    t3 = time.perf_counter()
    if K_star == V:
        hamming, _perm = _best_permutation_hamming(
            extracted_mask.astype(np.int64),
            gold_legality.astype(np.int64),
        )
    else:
        hamming = 1.0

    vidx = {v: i for i, v in enumerate(fsm.vertex_ids)}
    gt_state = np.asarray(
        [vidx[s.current_state] for s in train_ds.samples], dtype=np.int64
    )
    cluster_purity = _compute_cluster_purity(labels, gt_state)

    # Held-out NLL split by program.
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

    return E26Result(
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
        phase_1_wall_clock_seconds=float(phase_1_seconds),
        phase_2_wall_clock_seconds=float(phase_2_seconds),
        phase_3_wall_clock_seconds=float(phase_3_seconds),
        total_wall_clock_seconds=float(total),
        extraction_throughput_steps_per_sec=float(throughput),
        peak_memory_kb=float(_peak_memory_kb()),
    )
