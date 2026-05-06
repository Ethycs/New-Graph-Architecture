#!/usr/bin/env python3
"""Walk runs/, gather every metrics.jsonl, present one consolidated table.

This is the closing-the-loop tool the spec calls the "evidence-level tracker"
in lightweight form: read every JSONL row, group by (experiment, ablation,
seed), surface the headline numbers per run.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

D = Path(__file__).parent
RUNS = D / "runs"


def main() -> None:
    by_run: dict[str, dict[str, float]] = defaultdict(dict)
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir() or run_dir.name == ".gitkeep":
            continue
        m = run_dir / "metrics.jsonl"
        if not m.exists():
            continue
        for line in m.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_run[run_dir.name][r["metric_name"]] = r["value"]

    # ---- per-run table ------------------------------------------------------
    for run_id, metrics in by_run.items():
        print(f"\n=== {run_id} ===")
        for name, value in metrics.items():
            print(f"  {name:<40s} {value:>10.4f}")

    # ---- cross-run synthesis ------------------------------------------------
    print("\n=== Synthesis (cross-run) ===")
    e0 = by_run.get("E0_A0_seed42", {})
    e1 = by_run.get("E1_A0_seed42", {})
    e3 = by_run.get("E3_A0_seed42", {})
    e4 = by_run.get("E4_A0_seed42", {})
    e5 = by_run.get("E5_A0_seed42", {})
    e6 = by_run.get("E6_A0_seed42", {})
    e7 = by_run.get("E7_A0_seed42", {})
    e8 = by_run.get("E8_A0_seed42", {})

    rows = [
        ("Phase 1 - E0 accuracy", e0.get("accuracy")),
        ("Phase 1 - low-margin accuracy (E0)", e0.get("low_margin_accuracy")),
        ("Phase 2 - E1 accuracy with mask", e1.get("accuracy")),
        ("Phase 2 - E1 accuracy no mask", e1.get("accuracy_no_mask")),
        ("Phase 2 - mask uplift (E1)", e1.get("mask_accuracy_uplift")),
        ("Phase 2 - illegal rate with mask (E1)", e1.get("illegal_transition_rate")),
        ("Phase 2 - illegal rate no mask (E1)", e1.get("illegal_transition_rate_no_mask")),
        ("Phase 2 - margin AUROC (E4)", e4.get("margin_auroc")),
        ("Phase 2 - sigma AUROC (E4)", e4.get("sigma_auroc")),
        ("Phase 2 - sigma uplift (E4)", e4.get("sigma_uplift")),
        ("Phase 3 - hyperbolic d=8 (E3)", e3.get("hyperbolic_d8_acc")),
        ("Phase 3 - euclidean d=16 (E3)", e3.get("euclidean_d16_acc")),
        ("Phase 3 - hyperbolic uplift (E3)", e3.get("hyperbolic_uplift")),
    ]
    for name, value in rows:
        if value is None:
            print(f"  {name:<45s} (missing)")
        else:
            print(f"  {name:<45s} {value:>+10.4f}")

    # ---- E3 dim sweep table -------------------------------------------------
    if e3:
        print("\n=== E3 dim sweep (accuracy by dim, kind) ===")
        kinds = ["euclidean", "hyperbolic"]
        dims = sorted({int(k.rsplit("_d", 1)[1])
                       for k in e3 if "_d" in k and k.startswith("accuracy_")})
        header = f"  {'dim':>4s}  " + "  ".join(f"{k:>11s}" for k in kinds)
        print(header)
        for d in dims:
            cells = [e3.get(f"accuracy_{k}_d{d}") for k in kinds]
            print(f"  {d:>4d}  " + "  ".join(
                f"{c:>11.4f}" if c is not None else f"{'-':>11s}" for c in cells
            ))

    # ---- claim verdicts -----------------------------------------------------
    print("\n=== Load-bearing claim verdicts ===")
    claims = [
        (
            "Phase 1 - typed pipeline runs end-to-end",
            (e0.get("accuracy", 0) >= 0.90),
            f"E0 accuracy {e0.get('accuracy', 0):.3f} >= 0.90",
        ),
        (
            "Phase 2 - graph mask is load-bearing",
            (e1.get("mask_accuracy_uplift", 0) > 0.0
             and e1.get("illegal_transition_rate", 1) == 0.0),
            f"E1 mask uplift {e1.get('mask_accuracy_uplift', 0):+.3f}, "
            f"illegal rate {e1.get('illegal_transition_rate', 1):.3f}",
        ),
        (
            "Phase 2 - sigma does not degrade vs margin",
            (e4.get("sigma_uplift", -1.0) >= -0.05),
            f"sigma uplift {e4.get('sigma_uplift', -1.0):+.4f} (lenient bar -0.05)",
        ),
        (
            "Phase 2 - sigma beats margin by >=0.03 (q10 strict)",
            (e4.get("sigma_uplift", -1.0) >= 0.03),
            f"sigma uplift {e4.get('sigma_uplift', -1.0):+.4f} (strict bar +0.03)",
        ),
        (
            "Phase 3 - hyperbolic d=8 >= euclidean d=16 (q01 strict)",
            (e3.get("hyperbolic_uplift", -1.0) >= 0.0),
            f"hyperbolic uplift {e3.get('hyperbolic_uplift', -1.0):+.4f}",
        ),
        (
            "Phase 3 - hyperbolic >= euclidean at matched dim",
            (e3.get("accuracy_hyperbolic_d8", 0)
             >= e3.get("accuracy_euclidean_d8", 1)),
            f"d=8: hyp {e3.get('accuracy_hyperbolic_d8', 0):.3f} "
            f"vs euc {e3.get('accuracy_euclidean_d8', 0):.3f}",
        ),
        (
            "Phase 4 - orbit-pair multi-orbit parity",
            (e6.get("output_parity_l2_multi_orbit", 1.0) < 1e-6),
            f"multi-orbit parity_l2 "
            f"{e6.get('output_parity_l2_multi_orbit', float('nan')):.2e} (bar <1e-6)",
        ),
        (
            "Phase 5 - E5 IDF ablation reported",
            ("idf_uplift" in e5),
            f"idf_uplift {e5.get('idf_uplift', float('nan')):+.4f} "
            f"(report-only; sign varies by data)",
        ),
        (
            "Phase 5 - E7 reservoir param efficiency",
            (e7.get("n_trainable_params_reservoir", 1)
             < 0.1 * e7.get("n_trainable_params_endtoend", 0)),
            f"reservoir params {e7.get('n_trainable_params_reservoir', 0):.0f} "
            f"vs end-to-end {e7.get('n_trainable_params_endtoend', 0):.0f} "
            f"(bar <10%)",
        ),
        (
            "Phase 5 - E7 reservoir accuracy parity (xfail @ natural noise)",
            (e7.get("accuracy_reservoir", 0)
             >= 0.9 * e7.get("accuracy_endtoend", 1)),
            f"reservoir {e7.get('accuracy_reservoir', 0):.3f} "
            f"vs end-to-end {e7.get('accuracy_endtoend', 0):.3f} "
            f"(bar 0.9x; expected FAIL: 2-D random projection bottleneck)",
        ),
        (
            "Phase 5 - E8 transfer reported",
            ("transfer_gap" in e8),
            f"transfer_gap {e8.get('transfer_gap', float('nan')):+.4f} "
            f"(report-only; no bar)",
        ),
    ]
    for name, ok, detail in claims:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:<55s} -- {detail}")


if __name__ == "__main__":
    main()
