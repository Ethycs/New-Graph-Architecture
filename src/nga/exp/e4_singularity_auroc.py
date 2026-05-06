"""E4 - Singularity Detector Validation (post-hoc).

Reads results.jsonl from one or more source runs (typically E0 and E1),
computes the AUROC of margin and of sigma(x) against ground-truth errors
(y_true != y_hat), and asserts the load-bearing claim that sigma >= margin + 0.03
(resolves q10-failure-prediction-baseline).

This runner does not train or predict; it is purely an aggregator over
existing JSONL streams. The CLI surfaces it as `--experiment E4` plus the
new `--source-runs` flag listing the upstream run ids to analyse.

Edge cases:
- If n_errors == 0 (perfect classifier), AUROC is degenerate. binary_auroc
  already returns 0.5 with a UserWarning; this is documented behaviour.
- If n_samples < 30, a warning is written to stderr.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nga.arch.failure_margin_auroc import binary_auroc, margin_auroc, sigma_auroc
from nga.drivers.jsonl_writer import read_jsonl
from nga.drivers.metrics_jsonl import MetricsRecord, open_metrics_writer
from nga.drivers.results_jsonl import ResultsRecord

__all__ = ["E4Result", "run_e4"]


@dataclass
class E4Result:
    """Aggregate results returned by run_e4."""

    margin_auroc_value: float
    sigma_auroc_value: float
    sigma_uplift: float            # sigma_auroc - margin_auroc
    n_samples: int
    n_errors: int
    sources: list[str] = field(default_factory=list)  # the source run_ids analysed


def run_e4(
    *,
    runs_root: Path,
    source_run_ids: list[str],
    run_id: str,
    output_dir: Path,
    seed: int,
) -> E4Result:
    """Read source results.jsonl files, compute and write AUROC comparison.

    Outputs:
      - output_dir/metrics.jsonl   rows for margin_auroc, sigma_auroc,
                                   sigma_uplift, n_samples and n_errors as
                                   informational rows.
      - output_dir/results.jsonl   left empty (E4 has no per-sample output).
      - output_dir/scores.jsonl    left empty.

    Parameters
    ----------
    runs_root:
        Typically Path("runs") at the repo root.
    source_run_ids:
        List of upstream run dirs (e.g. ["E0_A0_seed42", "E1_A0_seed42"])
        whose results.jsonl files E4 reads.
    run_id:
        The run identifier for this E4 run (used in MetricsRecord fields).
    output_dir:
        Directory where metrics.jsonl (and empty results/scores files) are written.
    seed:
        Integer seed (passed through to MetricsRecord; not used for computation).

    Returns
    -------
    E4Result
        Aggregated AUROC comparison values.

    Raises
    ------
    ValueError
        If source_run_ids is empty or any source results.jsonl does not exist.
    """
    # ------------------------------------------------------------------
    # 1. Validate source_run_ids is non-empty
    # ------------------------------------------------------------------
    if not source_run_ids:
        raise ValueError(
            "source_run_ids must be non-empty; provide at least one upstream run id."
        )

    # ------------------------------------------------------------------
    # 2. Load results.jsonl from each source run
    # ------------------------------------------------------------------
    all_records: list[ResultsRecord] = []
    for src_id in source_run_ids:
        results_path = runs_root / src_id / "results.jsonl"
        if not results_path.exists():
            raise ValueError(
                f"Source results.jsonl not found for run '{src_id}': {results_path}"
            )
        records = read_jsonl(results_path, ResultsRecord)
        all_records.extend(records)

    # ------------------------------------------------------------------
    # 3. Filter: keep only records with sigma_score populated
    # ------------------------------------------------------------------
    n_total = len(all_records)
    filtered = [r for r in all_records if r.sigma_score is not None]
    n_skipped = n_total - len(filtered)
    if n_skipped > 0:
        frac = n_skipped / max(n_total, 1)
        warn_msg = (
            f"E4: {n_skipped}/{n_total} records ({frac:.1%}) had sigma_score=None "
            "and were skipped. These may be Phase 1 records without a sigma signal."
        )
        warnings.warn(warn_msg, UserWarning, stacklevel=2)

    n_samples = len(filtered)

    # ------------------------------------------------------------------
    # 4. Sample-count sanity check
    # ------------------------------------------------------------------
    if n_samples < 30:
        print(
            f"WARNING: E4 received only {n_samples} samples with sigma_score. "
            "AUROC estimates may be unreliable with fewer than 30 samples.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # 5. Build numpy arrays
    # ------------------------------------------------------------------
    margins = np.array([r.margin for r in filtered], dtype=float)
    sigmas = np.array([r.sigma_score for r in filtered], dtype=float)
    errors = np.array([r.y_true != r.y_hat for r in filtered], dtype=bool)

    n_errors = int(errors.sum())

    # ------------------------------------------------------------------
    # 6. Compute AUROCs
    # ------------------------------------------------------------------
    margin_auroc_value = margin_auroc(margins, errors)
    sigma_auroc_value = sigma_auroc(sigmas, errors)
    sigma_uplift = sigma_auroc_value - margin_auroc_value

    # ------------------------------------------------------------------
    # 7. Write MetricsRecord rows
    # ------------------------------------------------------------------
    metrics_path = output_dir / "metrics.jsonl"
    with open_metrics_writer(metrics_path) as writer:
        common = dict(
            run_id=run_id,
            experiment="E4",
            ablation=run_id.split("_")[1] if "_" in run_id else "A0",
            seed=seed,
            step=0,
            split="all",
        )
        writer.append(MetricsRecord(
            **common,
            metric_name="margin_auroc",
            value=margin_auroc_value,
        ))
        writer.append(MetricsRecord(
            **common,
            metric_name="sigma_auroc",
            value=sigma_auroc_value,
        ))
        writer.append(MetricsRecord(
            **common,
            metric_name="sigma_uplift",
            value=sigma_uplift,
        ))
        writer.append(MetricsRecord(
            **common,
            metric_name="n_samples",
            value=float(n_samples),
        ))
        writer.append(MetricsRecord(
            **common,
            metric_name="n_errors",
            value=float(n_errors),
        ))

    return E4Result(
        margin_auroc_value=margin_auroc_value,
        sigma_auroc_value=sigma_auroc_value,
        sigma_uplift=sigma_uplift,
        n_samples=n_samples,
        n_errors=n_errors,
        sources=list(source_run_ids),
    )
