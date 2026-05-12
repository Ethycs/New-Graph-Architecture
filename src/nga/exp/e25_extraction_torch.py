"""E25 - Universal graph extraction on the TorchEnergyTrainer substrate
(Phase 20 Wave B, Tier 1 sanity).

Where E24 validated the extraction pipeline on a synthetic typed Markov
chain, E25 plugs in our **own** trained substrate: a
``FrozenEncoderTorch`` over the python_big dataset, the same encoder
E18 uses for end-to-end TPN training. The decisive question is whether
the extraction pipeline -- cluster + K-select + Phase A + type-discover
-- recovers the hand-authored python_big legality matrix from the
encoder's hidden-state stream alone, without any direct access to FSM
labels during extraction.

Substrate
=========

We use the encoder substrate that E18 uses as Phase B's feature ground
truth: ``FrozenEncoderTorch.fit(train_ds.X)`` is a no-op flag flip; the
encoder's output ``Z = encoder.encode(train_ds.X)`` is a deterministic
(seeded) non-linear projection from the per-token raw features into the
trainer's embedding space. The harvest's ``hidden_states`` is exactly
``Z``; the harvest's ``sample_index`` is the ``program_id`` field on
the python_big samples, so each program's steps are grouped together
for transition counting.

This is the **honest** test: a frozen-random encoder. If the pipeline
recovers the FSM, it is because token-level features alone (the
encoder's input) already cluster informatively. If it does not, that is
the empirical signal that a *trained* encoder is required, and Wave C /
D will deliver one.

Comparison metric
=================

Best-permutation normalised Hamming between the extracted Beta-mean
legality matrix and the hand-authored ``GraphFSM.legality_matrix``,
*conditioned on* ``K_star == V`` (the FSM's vertex count). If
K-selection picks a different K, the comparison is undefined and we
emit 1.0 as a sentinel.

We also emit:

  - ``cluster_purity`` -- fraction of (step, ground_truth_state) pairs
    whose cluster's majority ground-truth state matches the step's
    state. The standard cluster-quality metric.
  - ``holdout_nll_improvement_per_token`` -- the extracted transition
    matrix's held-out predictive lift over a uniform-transition chain.
  - All five compute-efficiency metrics.

The Tier 1 acceptance bar (from ``docs/proposals/graph-extraction.md``)
is Hamming <= 0.05 on >= 3/5 grammars; this runner exercises one
grammar (python_big), so it is one observation among five. The full
multi-grammar Tier 1 result lives in research_log.md once Wave B runs
on all five grammars.
"""
from __future__ import annotations

import datetime as dt
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nga.arch.frozen_encoder_torch import FrozenEncoderTorch
from nga.arch.graph_fsm import GraphFSM
from nga.arch.hidden_state_harvester import HiddenStateHarvester
from nga.drivers.ablation_flags import AblationTuple
from nga.drivers.config import Config
from nga.drivers.jsonl_writer import JsonlWriter
from nga.drivers.metrics_jsonl import MetricsRecord
from nga.drivers.results_jsonl import ResultsRecord
from nga.exp.dataset_json import generate_json_dataset
from nga.exp.dataset_listops import generate_listops_dataset
from nga.exp.dataset_python_big import generate_python_big_dataset
from nga.exp.dataset_python_control import generate_python_control_dataset
from nga.exp.dataset_python_expr import generate_python_expr_dataset
from nga.exp.e24_graph_extraction import (
    _best_permutation_hamming,
    _compute_cluster_purity,
    _holdout_nll,
    _row_normalise_to_log,
    extract_graph,
)

__all__ = ["E25Result", "run_e25", "GRAMMAR_DISPATCH"]


# ---------------------------------------------------------------------------
# Per-grammar dataset dispatch
# ---------------------------------------------------------------------------
#
# Each grammar's dataset module has a *similar* but not identical
# constructor and sample shape. The dispatch table normalises them so the
# extraction runner can stay grammar-agnostic. Each entry exposes:
#
#   - ``loader(fsm, n, seed) -> dataset`` -- thin closure over the per-grammar
#     ``generate_*_dataset`` function, with the grammar's natural "sample
#     count" argument (n_programs / n_documents) mapped to ``n``.
#   - ``sequence_id_attr`` -- the name of the per-sample integer field
#     that groups consecutive transitions into one trajectory. python_*
#     grammars use ``program_id``, listops uses ``sequence_id``, json uses
#     ``document_id``.
#   - ``fsm_yaml_path`` -- the canonical FSM fixture for the grammar.
#   - ``default_n`` -- a Wave-B-friendly default sample count.

_GRAMMAR_DISPATCH = {
    "listops": {
        "loader": lambda fsm, n, seed: generate_listops_dataset(
            fsm=fsm, n_sequences=n, seed=seed
        ),
        "sequence_id_attr": "sequence_id",
        "fsm_yaml_path": "tests/fixtures/graphs/listops.fsm.yaml",
        "default_n": 80,
    },
    "python_expr": {
        "loader": lambda fsm, n, seed: generate_python_expr_dataset(
            fsm=fsm, n_programs=n, seed=seed
        ),
        "sequence_id_attr": "program_id",
        "fsm_yaml_path": "tests/fixtures/graphs/python_expr.fsm.yaml",
        "default_n": 80,
    },
    "python_big": {
        "loader": lambda fsm, n, seed: generate_python_big_dataset(
            fsm=fsm, n_programs=n, seed=seed
        ),
        "sequence_id_attr": "program_id",
        "fsm_yaml_path": "tests/fixtures/graphs/python_big.fsm.yaml",
        "default_n": 80,
    },
    "json": {
        "loader": lambda fsm, n, seed: generate_json_dataset(
            fsm=fsm, n_documents=n, seed=seed
        ),
        "sequence_id_attr": "document_id",
        "fsm_yaml_path": "tests/fixtures/graphs/json.fsm.yaml",
        "default_n": 80,
    },
    "python_control": {
        "loader": lambda fsm, n, seed: generate_python_control_dataset(
            fsm=fsm, n_programs=n, seed=seed
        ),
        "sequence_id_attr": "program_id",
        "fsm_yaml_path": "tests/fixtures/graphs/python_control.fsm.yaml",
        "default_n": 80,
    },
}

GRAMMAR_DISPATCH: dict[str, dict] = _GRAMMAR_DISPATCH


@dataclass
class E25Result:
    """Headline numbers from a single E25 extraction on python_big."""

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


def run_e25(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    grammar: str = "python_big",
    n_programs: int | None = None,
    encoder_hidden_dim: int = 32,
    k_selection_criterion: str = "bic",
    K_range_pad: int = 5,
    holdout_fraction: float = 0.2,
) -> E25Result:
    """Tier 1 sanity extraction on one of five grammars.

    Pipeline:
      1. Generate the dataset for the chosen grammar via the dispatch
         table (``GRAMMAR_DISPATCH``).
      2. Encode every step's feature vector via ``FrozenEncoderTorch``;
         harvest into a ``(N_total_steps, embedding_dim)`` matrix.
      3. Run ``extract_graph`` with ``K_range`` centred on
         ``V = fsm.vertex_count``.
      4. Permute the extracted legality matrix to align with the
         ground-truth FSM and report normalised Hamming.

    The ``config`` and ``ablation`` arguments are CLI-symmetry only.
    Pass ``grammar`` to switch grammars; defaults to ``python_big`` for
    backwards compatibility with the original Wave-B single-grammar run.
    """
    del config, ablation
    output_dir.mkdir(parents=True, exist_ok=True)

    if grammar not in _GRAMMAR_DISPATCH:
        raise ValueError(
            f"unknown grammar {grammar!r}; expected one of "
            f"{sorted(_GRAMMAR_DISPATCH.keys())}"
        )
    dispatch = _GRAMMAR_DISPATCH[grammar]

    grammar_label = grammar
    V = fsm.vertex_count
    K_min = max(2, V - K_range_pad)
    K_max = V + K_range_pad
    K_range = list(range(K_min, K_max + 1))

    t_start = time.perf_counter()

    # ---- Phase 1: dataset + harvest.
    t1 = time.perf_counter()
    n = int(n_programs) if n_programs is not None else int(dispatch["default_n"])
    train_ds = dispatch["loader"](fsm, n, seed)
    embedding_dim = int(train_ds.X.shape[1])
    encoder = FrozenEncoderTorch(
        input_dim=embedding_dim,
        output_dim=embedding_dim,
        hidden_dim=encoder_hidden_dim,
        seed=int(seed),
    )
    encoder.fit(train_ds.X)

    # Group rows by their per-sample sequence id so transition counting
    # respects program boundaries; rows arrive in time order within each
    # sequence.
    seq_attr = dispatch["sequence_id_attr"]
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in train_ds.samples], dtype=np.int64
    )

    Z = encoder.encode(train_ds.X).astype(np.float64)
    harvester = HiddenStateHarvester.from_callable(
        lambda i: Z[int(i)], per_sample_emits_sequence=False
    )
    # Use the harvester's container shape but feed directly: we already
    # have the encoded matrix in memory, so loop trivially.
    harvest = harvester.harvest(range(Z.shape[0]))
    # Re-label sample_index by program_id so transition counting groups
    # all rows of a single program together.
    harvest = type(harvest)(
        hidden_states=harvest.hidden_states,
        sample_index=program_ids,
    )
    phase_1_seconds = time.perf_counter() - t1

    # ---- Phase 2: extract.
    t2 = time.perf_counter()
    extraction = extract_graph(
        harvest.hidden_states,
        harvest.sample_index,
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

    # ---- Phase 3: compare to gold.
    t3 = time.perf_counter()
    if K_star == V:
        hamming, perm = _best_permutation_hamming(
            extracted_mask.astype(np.int64),
            gold_legality.astype(np.int64),
        )
    else:
        hamming = 1.0
        perm = None

    # Ground-truth state per step (FSM vertex ID -> integer).
    vidx = {v: i for i, v in enumerate(fsm.vertex_ids)}
    gt_state = np.asarray(
        [vidx[s.current_state] for s in train_ds.samples], dtype=np.int64
    )
    cluster_purity = _compute_cluster_purity(labels, gt_state)

    # Held-out NLL: split programs into train / holdout. Build extracted
    # transition log-probs from the train programs only; evaluate on the
    # holdout programs.
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
        float(harvest.n_steps) / total if total > 0.0 else float("nan")
    )

    # ---- Emit artefacts.
    experiment_label = "E25"
    ablation_label = run_id.split("_")[1] if "_" in run_id else "A0"
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

    metric_pairs: list[tuple[str, float]] = [
        ("n_total_steps", float(harvest.n_steps)),
        ("n_samples", float(n_programs_total)),
        ("hidden_dim", float(harvest.hidden_dim)),
        ("V_ground_truth", float(V)),
        ("K_star", float(K_star)),
        ("extracted_hamming_normalised", float(hamming)),
        ("cluster_purity", float(cluster_purity)),
        ("holdout_nll_per_token_extracted", float(holdout_nll_extracted)),
        ("holdout_nll_per_token_chain", float(holdout_nll_chain)),
        (
            "holdout_nll_improvement_per_token",
            float(holdout_nll_chain - holdout_nll_extracted),
        ),
        ("phase_1_wall_clock_seconds", float(phase_1_seconds)),
        ("phase_2_wall_clock_seconds", float(phase_2_seconds)),
        ("phase_3_wall_clock_seconds", float(phase_3_seconds)),
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
        # Per-step results: cluster ID + ground-truth state name.
        for i in range(harvest.n_steps):
            results_writer.append(
                ResultsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=i,
                    sample_id=f"prog{int(program_ids[i])}_step{i}",
                    y_true=str(train_ds.samples[i].current_state),
                    y_hat=f"cluster{int(labels[i])}",
                    margin=0.0,
                    singular_flag=False,
                    sigma_score=0.0,
                    behavioral_stratum=None,
                    stratum_bitmask=None,
                    transition_legal=None,
                    timestamp=timestamp,
                )
            )

    extracted_graph_path = output_dir / "extracted_graph.json"
    extracted_graph_path.write_text(
        json.dumps(
            {
                "grammar": grammar_label,
                "V_ground_truth": int(V),
                "K_star": int(K_star),
                "K_range": list(K_range),
                "K_scores": extraction["K_scores"].tolist(),
                "extracted_mask": extracted_mask.astype(int).tolist(),
                "gold_legality": gold_legality.astype(int).tolist(),
                "best_permutation": perm.tolist() if perm is not None else None,
                "fsm_vertex_ids": list(fsm.vertex_ids),
                "criterion": k_selection_criterion,
            },
            indent=2,
        )
    )

    return E25Result(
        grammar=grammar_label,
        n_total_steps=int(harvest.n_steps),
        n_samples=int(n_programs_total),
        hidden_dim=int(harvest.hidden_dim),
        V_ground_truth=int(V),
        K_star=int(K_star),
        extracted_hamming_normalised=float(hamming),
        cluster_purity=float(cluster_purity),
        holdout_nll_per_token_extracted=float(holdout_nll_extracted),
        holdout_nll_per_token_chain=float(holdout_nll_chain),
        holdout_nll_improvement_per_token=float(
            holdout_nll_chain - holdout_nll_extracted
        ),
        phase_1_wall_clock_seconds=float(phase_1_seconds),
        phase_2_wall_clock_seconds=float(phase_2_seconds),
        phase_3_wall_clock_seconds=float(phase_3_seconds),
        total_wall_clock_seconds=float(total),
        extraction_throughput_steps_per_sec=float(throughput),
        peak_memory_kb=float(_peak_memory_kb()),
    )
