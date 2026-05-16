"""E30 - PCG-X over a frozen pretrained causal LM (Phase 24 / deferred Tier-3).

The original `docs/proposals/graph-extraction.md` Tier-3 test, finally
shipped: the substrate is now a **frozen pretrained** causal LM —
GPT-2 small (124M, 12 layers, hidden 768) loaded from the DVC-tracked
`~/models/hf` cache. The model is not trained on the grammar; it ships
with whatever priors HuggingFace's GPT-2 already learned on WebText.
PCG-X then asks whether useful grammar-shaped regimes emerge from
activations of a substrate that was never told this grammar exists.

Contrast with prior substrates:

  - E26 (Wave-C MLP, state-conditioned): trained on next-state CE with
    state injected; mean purity 0.79.
  - E29 (small transformer from scratch): trained on the grammar's
    causal-LM objective; mean purity 0.66 with substrate next-token acc
    0.63.
  - E30 (this runner; **frozen pretrained**): no grammar-specific
    training of the substrate at all; the substrate sees the grammar
    only through its tokens. Open empirical question.

Pipeline
========

1. Load GPT-2 from ``~/models/hf`` (HF_HOME pinned, ``TRANSFORMERS_OFFLINE=1``
   so the loader never reaches out to the hub).
2. Generate the grammar's dataset. Group samples by program_id.
3. For each program: build a text by joining observed tokens with a
   space, tokenize once, align each grammar step to the index of its
   final BPE position via the fast tokenizer's offset_mapping, run a
   single forward pass with ``output_hidden_states=True``, and harvest
   the mid-layer hidden state at each step's BPE position.
4. Run the standard PCG-X pipeline on the harvested ``h``: predictive
   projection h -> z (next-state CE + entropy regression + failure
   BCE), argmax-of-next-state partition, bisimulation quotient to
   ``target_K = V`` as a post-hoc consolidator.
5. Emit ``control_graph.json`` and ``decision_trace.jsonl`` exactly as
   E28 does (the σ + control bridge is substrate-agnostic).
"""
from __future__ import annotations

import json
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from nga.arch.bisimulation_quotient import quotient_by_bisimulation
from nga.arch.forward_backward import bayesian_m_step_beta
from nga.arch.graph_fsm import GraphFSM
from nga.arch.posterior_mask import PosteriorMask
from nga.arch.predictive_projection import (
    PredictiveProjection,
    PredictiveProjectionConfig,
    train_predictive_projection,
)
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH
from nga.exp.e28_pcg_extractor import (
    RegimeEdge,
    RegimeNode,
    _emit_regime_decision_trace,
    _failure_targets,
    _next_state_entropy_targets,
    _peak_memory_kb,
    _UNKNOWN_REGIME,
)

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

__all__ = ["E30Result", "run_e30"]


# Pin the HF resolution to the DVC-tracked area before any transformers
# import happens (callers should also set this; we set it defensively).
_HF_HOME = str(Path.home() / "models" / "hf")
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# GPT-2 small: 12 layers, hidden_states tuple has 13 entries
# (embedding + 12 transformer blocks). Default is mid-layer (block 6).
# Phase 25 (scripts/phase25_layer_ablation_sweep.py) swept layers and
# established a robust shape: L0 (embedding) ≈ 0.56, L6 ≈ 0.75, L10 ≈
# 0.78, L12 (final block) ≈ 0.67 mean purity across 5 grammars. The
# mid-to-late (L6-L10) region is the sweet spot; the specific best
# layer between L6 and L10 is within CUDA-nondeterminism seed variance
# (~±0.02-0.03), so multi-seed bootstrap is needed to claim L10 > L6.
# Until then, L6 stays the default because it matches Phase 24's
# published numbers and the L6-vs-L10 gap is in noise.
_DEFAULT_MODEL_ID = "gpt2"
_DEFAULT_HARVEST_LAYER = 6
_GPT2_MAX_CONTEXT = 1024


@dataclass
class E30Result:
    """Headline numbers for one E30 run."""

    grammar: str
    model_id: str
    n_total_steps: int
    n_samples: int

    V_ground_truth: int
    n_argmax_cells: int
    n_regimes_after_merge: int

    mean_failure_rate: float
    mean_entropy: float
    mean_margin_to_tie: float
    mean_purity_against_current_state: float

    aligned_hamming_at_target_V: float

    projection_next_state_accuracy: float
    projection_failure_accuracy: float
    projection_entropy_mse: float
    projection_token_accuracy: float
    adversarial_token_weight: float
    harvest_layer: int
    n_programs_truncated_to_context: int

    phase_1_wall_clock_seconds: float
    phase_2_wall_clock_seconds: float
    phase_3_wall_clock_seconds: float
    phase_4_wall_clock_seconds: float
    total_wall_clock_seconds: float
    peak_memory_kb: float


# ---------------------------------------------------------------------------
# Pretrained substrate harvester
# ---------------------------------------------------------------------------


class _PretrainedSubstrate:
    """Loads a frozen pretrained causal LM and harvests per-step hidden states.

    Built around the offset-mapping path so each grammar step (one
    ``observed_token`` from the dataset) maps deterministically to a
    single BPE position — the LAST BPE that overlaps the step's
    character range — and we take one mid-layer vector per step.

    Memory note: GPT-2 small at fp32 is ~500 MB on GPU. With max
    sequence length capped at 1024 (GPT-2's context) and batch size 1,
    the activations fit comfortably on a 6 GB card.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        harvest_layer: int = _DEFAULT_HARVEST_LAYER,
        device: str | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for E30")
        # Lazy-import transformers so the module is import-safe when the
        # transformers dep isn't present (e.g., the import-only CI gate).
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model_id = model_id
        self.harvest_layer = int(harvest_layer)

        # Fast tokenizer so offset_mapping works.
        self._tokenizer: "PreTrainedTokenizerFast" = AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )
        self._model: "PreTrainedModel" = AutoModelForCausalLM.from_pretrained(
            model_id
        )
        self._model.eval()
        self._model.to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)

        self.hidden_size: int = int(self._model.config.hidden_size)
        # GPT-2 hidden_states tuple length = n_layers + 1 (embedding + each block).
        self._n_layers: int = int(self._model.config.num_hidden_layers)
        if not (0 <= self.harvest_layer <= self._n_layers):
            raise ValueError(
                f"harvest_layer {self.harvest_layer} out of range; model has "
                f"{self._n_layers} transformer blocks (valid: 0..{self._n_layers})"
            )

        self.max_context: int = int(
            getattr(self._model.config, "n_positions", _GPT2_MAX_CONTEXT)
        )

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self._model.parameters()))

    def harvest_program(
        self, observed_tokens: list[str]
    ) -> tuple[np.ndarray, bool]:
        """Return ``(per_step_hidden, was_truncated)``.

        ``per_step_hidden`` has shape ``(len(observed_tokens), hidden_size)``
        in fp32 numpy. ``was_truncated`` is True iff the tokenized
        sequence exceeded the model's context and we dropped grammar
        steps that fell beyond the cap.
        """
        if not observed_tokens:
            return np.zeros((0, self.hidden_size), dtype=np.float64), False

        # Build text + per-step char ranges. Single leading space before
        # every step except the first so each BPE token gets the standard
        # word-boundary treatment GPT-2 expects.
        text_parts: list[str] = []
        char_ranges: list[tuple[int, int]] = []
        cursor = 0
        for i, tok in enumerate(observed_tokens):
            if i > 0:
                text_parts.append(" ")
                cursor += 1
            start = cursor
            text_parts.append(tok)
            cursor += len(tok)
            char_ranges.append((start, cursor))
        text = "".join(text_parts)

        # Fast-tokenize once. ``offset_mapping`` is (n_bpe, 2) of char ranges.
        enc = self._tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context,
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"][0].tolist()
        n_bpe = len(offsets)

        # For each grammar step, find the LAST BPE token whose start char
        # is strictly less than the step's end char. That's the BPE position
        # whose hidden state encodes "up to and including this grammar step."
        bpe_indices: list[int] = []
        last_seen = 0
        for _cs, ce in char_ranges:
            last_idx = -1
            for k in range(last_seen, n_bpe):
                bs = offsets[k][0]
                if bs < ce:
                    last_idx = k
                else:
                    break
            if last_idx < 0:
                # Step fell past the truncation boundary; we'll record it as
                # missing and the runner skips it from harvested arrays.
                bpe_indices.append(-1)
            else:
                bpe_indices.append(last_idx)
                last_seen = last_idx

        # Forward.
        input_ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        # hidden_states is a tuple of (n_layers + 1) tensors of shape
        # (1, seq_len, hidden_size). Index 0 is the embedding output;
        # index k for k in [1, n_layers] is block k's output.
        layer_out = out.hidden_states[self.harvest_layer]  # (1, seq_len, d)

        # Gather one vector per step.
        per_step = np.zeros((len(observed_tokens), self.hidden_size), dtype=np.float64)
        was_truncated = False
        layer_np = layer_out[0].detach().cpu().numpy()
        for step_i, bpe_i in enumerate(bpe_indices):
            if bpe_i < 0:
                was_truncated = True
                # Reuse the last valid BPE position for the missing tail.
                # Out-of-distribution but defensible: the alternative is
                # to drop those steps, which complicates downstream
                # alignment with current_states / y_next.
                per_step[step_i] = layer_np[-1]
            else:
                per_step[step_i] = layer_np[bpe_i]

        return per_step, was_truncated


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_e30(
    *,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    grammar: str = "python_expr",
    n_programs: int | None = None,
    model_id: str = _DEFAULT_MODEL_ID,
    harvest_layer: int = _DEFAULT_HARVEST_LAYER,
    device: str | None = None,
    projection_z_dim: int = 32,
    projection_hidden_dim: int = 64,
    n_projection_train_epochs: int = 30,
    target_n_regimes: int | None = None,
    adversarial_token_weight: float = 0.0,
    eval_n_programs: int | None = None,
    eval_seed: int | None = None,
) -> E30Result:
    """PCG-X on a frozen pretrained causal LM."""
    if torch is None:
        raise ImportError("torch is required for E30")
    if grammar not in GRAMMAR_DISPATCH:
        raise ValueError(
            f"unknown grammar {grammar!r}; expected one of "
            f"{sorted(GRAMMAR_DISPATCH.keys())}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dispatch = GRAMMAR_DISPATCH[grammar]
    V = fsm.vertex_count
    target_K = int(target_n_regimes) if target_n_regimes is not None else V

    t_start = time.perf_counter()

    # ---- Phase 1: load pretrained, dataset, harvest. -----------------------
    t1 = time.perf_counter()
    substrate = _PretrainedSubstrate(
        model_id=model_id, harvest_layer=harvest_layer, device=device
    )

    n = int(n_programs) if n_programs is not None else int(dispatch["default_n"])
    train_ds = dispatch["loader"](fsm, n, seed)
    seq_attr = dispatch["sequence_id_attr"]
    n_total = int(train_ds.X.shape[0])
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in train_ds.samples], dtype=np.int64
    )
    current_states = train_ds.current_states.astype(np.int64)

    # Group samples by program in their original order.
    program_to_indices: dict[int, list[int]] = {}
    for i, pid in enumerate(program_ids):
        program_to_indices.setdefault(int(pid), []).append(i)

    h_all = np.zeros((n_total, substrate.hidden_size), dtype=np.float64)
    n_truncated = 0
    for pid, indices in program_to_indices.items():
        ordered = sorted(indices)
        observed_tokens = [train_ds.samples[i].observed_token for i in ordered]
        per_step, was_truncated = substrate.harvest_program(observed_tokens)
        if was_truncated:
            n_truncated += 1
        for k, original_idx in enumerate(ordered):
            h_all[original_idx] = per_step[k]
    phase_1_seconds = time.perf_counter() - t1

    # ---- Phase 2: predictive projection h -> z. ---------------------------
    t2 = time.perf_counter()
    # Build token vocab for the adversarial head. Even when its weight is
    # 0.0 we compute it cheaply so the diagnostics column is comparable
    # across runs / phases.
    observed_tokens = [s.observed_token for s in train_ds.samples]
    token_vocab = sorted(set(observed_tokens))
    token_to_id = {t: i for i, t in enumerate(token_vocab)}
    token_ids = np.asarray(
        [token_to_id[t] for t in observed_tokens], dtype=np.int64
    )
    n_tokens = len(token_vocab)

    proj_cfg = PredictiveProjectionConfig(
        z_dim=projection_z_dim,
        hidden_dim=projection_hidden_dim,
        n_states=V,
        entropy_weight=1.0,
        failure_weight=1.0,
        adversarial_token_weight=float(adversarial_token_weight),
        n_tokens=n_tokens if adversarial_token_weight > 0.0 else 0,
    )
    projection = PredictiveProjection(
        input_dim=substrate.hidden_size, config=proj_cfg
    )
    entropy_targets = _next_state_entropy_targets(fsm, current_states)
    failure_targets = _failure_targets(train_ds.y_next, fsm, current_states)
    proj_diag = train_predictive_projection(
        projection,
        h_all,
        next_states=train_ds.y_next,
        entropy_targets=entropy_targets,
        failure_targets=failure_targets,
        token_ids=(
            token_ids if adversarial_token_weight > 0.0 else None
        ),
        epochs=n_projection_train_epochs,
        lr=1e-3,
        batch_size=64,
        seed=int(seed),
    )
    with torch.no_grad():
        out = projection(torch.from_numpy(h_all.astype(np.float32)))
    next_logits = out["next_state_logits"].cpu().numpy().astype(np.float64)
    entropy_pred = out["entropy_pred"].cpu().numpy().astype(np.float64)
    failure_pred = out["failure_logit"].cpu().numpy().astype(np.float64)
    phase_2_seconds = time.perf_counter() - t2

    # ---- Phase 3: argmax partition. ---------------------------------------
    t3 = time.perf_counter()
    argmax_cells = next_logits.argmax(axis=1).astype(np.int64)
    sorted_logits = np.sort(next_logits, axis=1)
    margin_to_tie = (sorted_logits[:, -1] - sorted_logits[:, -2]).astype(np.float64)
    cells = np.unique(argmax_cells)
    K_argmax = int(cells.size)
    cell_to_idx = {int(c): i for i, c in enumerate(cells)}
    compact = np.asarray(
        [cell_to_idx[int(c)] for c in argmax_cells], dtype=np.int64
    )
    from nga.arch.forward_backward import expected_counts_observed

    transition_counts = np.zeros((K_argmax, K_argmax), dtype=np.int64)
    for p in range(int(program_ids.max()) + 1):
        rows = np.where(program_ids == p)[0]
        if rows.size < 2:
            continue
        seq = compact[rows]
        transition_counts += expected_counts_observed(seq, K_argmax)

    per_cell = []
    for c_idx in range(K_argmax):
        members = np.where(compact == c_idx)[0]
        if members.size == 0:
            continue
        f_logit = failure_pred[members]
        f_prob = 1.0 / (1.0 + np.exp(-f_logit))
        per_cell.append(
            {
                "support": int(members.size),
                "failure_rate": float(f_prob.mean()),
                "entropy_mean": float(entropy_pred[members].mean()),
                "mean_margin_to_tie": float(margin_to_tie[members].mean()),
                "current_state_distribution": np.bincount(
                    current_states[members], minlength=V
                ).astype(np.int64),
            }
        )
    phase_3_seconds = time.perf_counter() - t3

    # ---- Phase 4: bisimulation merge. -------------------------------------
    t4 = time.perf_counter()
    emission_counts = np.stack(
        [c["current_state_distribution"] for c in per_cell], axis=0
    ).astype(np.int64)
    quotient = quotient_by_bisimulation(
        compact,
        transition_counts,
        target_K=min(target_K, K_argmax),
        criterion="full",
        emission_counts=emission_counts,
        seed=int(seed),
    )
    merged_labels = quotient.labels
    merged_counts = quotient.transition_counts
    n_regimes = int(merged_counts.shape[0])
    phase_4_seconds = time.perf_counter() - t4

    # Per-regime stats.
    regimes: list[RegimeNode] = []
    for r_id in range(n_regimes):
        members = np.where(merged_labels == r_id)[0]
        if members.size == 0:
            regimes.append(
                RegimeNode(
                    regime_id=r_id,
                    support=0,
                    failure_rate=float("nan"),
                    entropy_mean=float("nan"),
                    mean_margin_to_tie=float("nan"),
                    dominant_current_state="<empty>",
                    purity_against_current_state=0.0,
                )
            )
            continue
        f_prob = 1.0 / (1.0 + np.exp(-failure_pred[members]))
        cs_counts = np.bincount(current_states[members], minlength=V)
        dom = int(cs_counts.argmax())
        purity = float(cs_counts[dom]) / float(members.size)
        regimes.append(
            RegimeNode(
                regime_id=r_id,
                support=int(members.size),
                failure_rate=float(f_prob.mean()),
                entropy_mean=float(entropy_pred[members].mean()),
                mean_margin_to_tie=float(margin_to_tie[members].mean()),
                dominant_current_state=str(fsm.vertex_ids[dom]),
                purity_against_current_state=purity,
            )
        )

    alpha, beta = bayesian_m_step_beta(
        merged_counts.astype(np.float64),
        expected_counts_neg=None,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    edges: list[RegimeEdge] = []
    row_sums = merged_counts.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    probs = merged_counts.astype(np.float64) / row_sums
    for i in range(n_regimes):
        for j in range(n_regimes):
            count = int(merged_counts[i, j])
            if count == 0:
                continue
            edges.append(
                RegimeEdge(
                    src=i,
                    dst=j,
                    probability=float(probs[i, j]),
                    confidence_alpha=float(alpha[i, j]),
                    confidence_beta=float(beta[i, j]),
                    count=count,
                )
            )

    aligned_hamming = float("nan")
    if n_regimes == V:
        from nga.exp.e24_graph_extraction import _best_permutation_hamming

        posterior_mask = PosteriorMask(V, prior_alpha=1.0, prior_beta=1.0)
        posterior_mask._alpha = alpha
        posterior_mask._beta = beta
        extracted = posterior_mask.legality_matrix(0.5)
        h_norm, _perm = _best_permutation_hamming(
            extracted.astype(np.int64),
            fsm.legality_matrix.astype(np.int64),
        )
        aligned_hamming = float(h_norm)

    total = time.perf_counter() - t_start

    # Emit control_graph.json (training-set extraction).
    control_graph = {
        "grammar": grammar,
        "substrate": f"frozen_pretrained({model_id}, layer={harvest_layer})",
        "n_total_steps": int(h_all.shape[0]),
        "n_argmax_cells_before_merge": int(K_argmax),
        "n_regimes_after_merge": int(n_regimes),
        "regimes": [
            {
                "regime_id": r.regime_id,
                "support": r.support,
                "failure_rate": r.failure_rate,
                "entropy_mean": r.entropy_mean,
                "mean_margin_to_tie": r.mean_margin_to_tie,
                "dominant_current_state": r.dominant_current_state,
                "purity_against_current_state": r.purity_against_current_state,
            }
            for r in regimes
        ],
        "edges": [
            {
                "src": e.src,
                "dst": e.dst,
                "probability": e.probability,
                "confidence_alpha": e.confidence_alpha,
                "confidence_beta": e.confidence_beta,
                "count": e.count,
            }
            for e in edges
        ],
        "diagnostics": {
            "aligned_hamming_at_target_V": aligned_hamming,
            "model_id": model_id,
            "harvest_layer": int(harvest_layer),
            "hidden_size": int(substrate.hidden_size),
            "device": str(substrate.device),
            "projection": proj_diag,
            "adversarial_token_weight": float(adversarial_token_weight),
            "n_tokens": int(n_tokens),
            "n_programs_truncated_to_context": int(n_truncated),
        },
    }
    (output_dir / "control_graph.json").write_text(
        json.dumps(control_graph, indent=2)
    )

    # ---- σ + control trace (substrate-agnostic helper from E28). ---------
    fsm_to_regime = np.full(int(V), _UNKNOWN_REGIME, dtype=np.int64)
    for fsm_state, compact_idx in cell_to_idx.items():
        fsm_to_regime[int(fsm_state)] = int(quotient.cluster_map[compact_idx])

    if eval_n_programs is not None:
        eval_logits, eval_predicted_regimes, eval_program_ids = (
            _harvest_eval_slice_pretrained(
                fsm=fsm,
                dispatch=dispatch,
                substrate=substrate,
                projection=projection,
                fsm_to_regime=fsm_to_regime,
                eval_n_programs=int(eval_n_programs),
                eval_seed=(
                    int(eval_seed) if eval_seed is not None else int(seed) + 1
                ),
                n_fsm_states=int(V),
            )
        )
        trace_logits = eval_logits
        trace_predicted_regimes = eval_predicted_regimes
        trace_program_ids = eval_program_ids
        trace_label = "eval"
    else:
        trace_logits = next_logits
        trace_predicted_regimes = merged_labels.astype(np.int64)
        trace_program_ids = program_ids
        trace_label = "train"

    _emit_regime_decision_trace(
        output_dir=output_dir,
        run_id=run_id,
        seed=int(seed),
        next_logits=trace_logits,
        predicted_regimes=trace_predicted_regimes,
        program_ids=trace_program_ids,
        fsm_to_regime=fsm_to_regime,
        regimes=regimes,
        regime_edges=edges,
        n_regimes=n_regimes,
        trace_label=trace_label,
    )

    # Aggregate.
    mean_failure = float(
        np.mean([r.failure_rate for r in regimes if r.support > 0])
    )
    mean_entropy = float(
        np.mean([r.entropy_mean for r in regimes if r.support > 0])
    )
    mean_margin = float(
        np.mean([r.mean_margin_to_tie for r in regimes if r.support > 0])
    )
    mean_purity = float(
        np.mean(
            [r.purity_against_current_state for r in regimes if r.support > 0]
        )
    )

    return E30Result(
        grammar=grammar,
        model_id=model_id,
        n_total_steps=int(h_all.shape[0]),
        n_samples=int(program_ids.max()) + 1,
        V_ground_truth=int(V),
        n_argmax_cells=int(K_argmax),
        n_regimes_after_merge=int(n_regimes),
        mean_failure_rate=mean_failure,
        mean_entropy=mean_entropy,
        mean_margin_to_tie=mean_margin,
        mean_purity_against_current_state=mean_purity,
        aligned_hamming_at_target_V=float(aligned_hamming),
        projection_next_state_accuracy=float(proj_diag.get("next_state_acc", 0.0)),
        projection_failure_accuracy=float(proj_diag.get("failure_acc", 0.0)),
        projection_entropy_mse=float(proj_diag.get("entropy_mse", 0.0)),
        projection_token_accuracy=float(
            proj_diag.get("token_acc", float("nan"))
        ),
        adversarial_token_weight=float(adversarial_token_weight),
        harvest_layer=int(harvest_layer),
        n_programs_truncated_to_context=int(n_truncated),
        phase_1_wall_clock_seconds=float(phase_1_seconds),
        phase_2_wall_clock_seconds=float(phase_2_seconds),
        phase_3_wall_clock_seconds=float(phase_3_seconds),
        phase_4_wall_clock_seconds=float(phase_4_seconds),
        total_wall_clock_seconds=float(total),
        peak_memory_kb=float(_peak_memory_kb()),
    )


def _harvest_eval_slice_pretrained(
    *,
    fsm: GraphFSM,
    dispatch: dict,
    substrate: _PretrainedSubstrate,
    projection: PredictiveProjection,
    fsm_to_regime: np.ndarray,
    eval_n_programs: int,
    eval_seed: int,
    n_fsm_states: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the frozen substrate + trained projection over a held-out slice."""
    eval_ds = dispatch["loader"](fsm, eval_n_programs, eval_seed)
    seq_attr = dispatch["sequence_id_attr"]
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in eval_ds.samples], dtype=np.int64
    )

    program_to_indices: dict[int, list[int]] = {}
    for i, pid in enumerate(program_ids):
        program_to_indices.setdefault(int(pid), []).append(i)

    h_eval = np.zeros(
        (eval_ds.X.shape[0], substrate.hidden_size), dtype=np.float64
    )
    for pid, indices in program_to_indices.items():
        ordered = sorted(indices)
        observed_tokens = [eval_ds.samples[i].observed_token for i in ordered]
        per_step, _was_truncated = substrate.harvest_program(observed_tokens)
        for k, original_idx in enumerate(ordered):
            h_eval[original_idx] = per_step[k]

    with torch.no_grad():
        out = projection(torch.from_numpy(h_eval.astype(np.float32)))
    eval_logits = out["next_state_logits"].cpu().numpy().astype(np.float64)
    argmax_fsm = eval_logits.argmax(axis=1).astype(np.int64)
    predicted_regimes = fsm_to_regime[argmax_fsm]
    return eval_logits, predicted_regimes, program_ids
