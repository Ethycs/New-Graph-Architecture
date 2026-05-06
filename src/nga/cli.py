"""CLI runner for the NGA research pipeline.

Entry point: ``python run.py`` delegates here via ``nga.cli.main``.

Responsibilities (Phase 0):
  - Parse the 11-flag argument surface with argparse.
  - Derive ``run_id`` and create the output directory (guarded against
    accidental overwrites).
  - Seed all RNGs via ``nga.drivers.seeding.set_seed``.
  - Load and merge config; write ``config_snapshot.yaml``.
  - Load ablation file; resolve the requested tuple; write
    ``ablation_snapshot.yaml``.
  - Load the Graph FSM spec; snapshot it to ``graph_fsm.yaml``.
  - Enforce the cross-field invariant: if ``group_quotient_enabled`` is True
    the config must supply a ``group_spec_path``.
  - Touch ``metrics.jsonl``, ``results.jsonl``, and ``scores.jsonl`` so they
    exist on disk (byte-empty for noop runs).
  - Return exit code 0 on success; 1 on any caught, expected error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

from nga.drivers import ablation_flags as ablation_flags_mod
from nga.drivers import config as config_mod
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod
from nga.drivers import seeding

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

EXPERIMENT_CHOICES: list[str] = [f"E{i}" for i in range(10)]
ABLATION_PATTERN = re.compile(r"^A\d+$")


def derive_run_id(experiment: str, ablation: str, seed: int) -> str:
    """Return the canonical run identifier string.

    Format: ``<experiment>_<ablation>_seed<seed>``, e.g. ``E0_A0_seed42``.
    """
    return f"{experiment}_{ablation}_seed{seed}"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Exposed at module level so test code can introspect the parser without
    invoking ``main``.
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "NGA pipeline runner. Loads config + graph FSM + ablation tuple, "
            "creates the run directory, and dispatches training/evaluation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- required flags -----------------------------------------------
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=EXPERIMENT_CHOICES,
        help="Experiment identifier (E0 through E9).",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        help="Ablation variant identifier, e.g. A0. Must match ^A\\d+$.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the run config YAML (must exist).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Integer seed for all RNGs. Overrides Config.seed.",
    )

    # ---- optional flags -----------------------------------------------
    parser.add_argument(
        "--ablation-file",
        type=Path,
        default=Path("configs/ablations.yaml"),
        dest="ablation_file",
        help="Path to the ablations YAML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. Derived as runs/<run_id> when omitted. "
            "If the derived directory already exists and is non-empty, "
            "the run aborts with exit code 1."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cpu' or 'cuda:0'. Overrides Config.device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Mini-batch size. Overrides Config.batch_size.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        dest="num_epochs",
        help="Number of training epochs. Overrides Config.num_epochs.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=False,
        dest="skip_train",
        help="Skip the training phase (noop).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        dest="skip_eval",
        help="Skip the evaluation phase (noop).",
    )
    parser.add_argument(
        "--source-runs",
        type=str,
        nargs="+",
        default=None,
        dest="source_runs",
        help=(
            "Upstream run_ids whose results.jsonl files E4 should aggregate. "
            "Defaults to ['E0_A0_seed42', 'E1_A0_seed42'] when --experiment E4."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        dest="runs_root",
        help="Root directory under which run_ids resolve. Default: ./runs",
    )

    return parser


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_ablation_flag(ablation: str) -> None:
    """Raise ValueError if ablation does not match ^A\\d+$."""
    if not ABLATION_PATTERN.match(ablation):
        raise ValueError(
            f"--ablation must match ^A\\d+$; got '{ablation}'"
        )


def _resolve_output_dir(
    output_arg: Optional[Path],
    run_id: str,
    output_was_explicit: bool,
) -> Path:
    """Return the resolved output directory path.

    - If ``--output`` was given explicitly, use it with ``exist_ok=True``.
    - Otherwise, derive ``runs/<run_id>/`` and fail if it already exists and
      is non-empty (exit code 1 is signalled by raising ValueError).
    """
    if output_was_explicit and output_arg is not None:
        output_dir = output_arg
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("runs") / run_id
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(
                f"Run directory '{output_dir}' already exists and is non-empty. "
                "Use --output to specify an explicit destination, or remove the "
                "existing directory before re-running."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _apply_cli_overrides(
    cfg: config_mod.Config,
    args: argparse.Namespace,
) -> config_mod.Config:
    """Return a new Config with CLI flag values merged in.

    CLI flags win over config-file values when supplied.
    """
    overrides: dict[str, object] = cfg.model_dump()
    if args.device is not None:
        overrides["device"] = args.device
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    # --seed is always required, so always override Config.seed.
    overrides["seed"] = args.seed
    return config_mod.Config.model_validate(overrides)


def _write_ablation_snapshot(
    tuple_obj: ablation_flags_mod.AblationTuple,
    output_dir: Path,
) -> None:
    """Dump just the resolved AblationTuple to ablation_snapshot.yaml.

    The snapshot contains a flat YAML mapping of the tuple's fields (via
    model_dump) rather than the full ablations file.
    """
    data = tuple_obj.model_dump()
    snapshot_path = output_dir / "ablation_snapshot.yaml"
    with snapshot_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


def _touch_empty_jsonl_files(output_dir: Path) -> None:
    """Create byte-empty JSONL sink files in the run directory.

    For noop runs (--skip-train --skip-eval) these files are touched so they
    exist on disk at 0 bytes. For live runs the same files will be opened by
    the arch/exp writers later; touching here guarantees their presence
    regardless of pipeline outcome.
    """
    for name in ("metrics.jsonl", "results.jsonl", "scores.jsonl"):
        (output_dir / name).touch()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Parse arguments, set up the run environment, and launch the pipeline.

    Returns 0 on success; 1 on any caught, expected error.  Unexpected
    exceptions propagate to the caller (and ultimately to the OS).

    Steps:
      1. Parse argv; validate --ablation pattern.
      2. Derive run_id and resolve/create the output directory.
      3. Seed all RNGs.
      4. Load config; apply CLI overrides; write config_snapshot.yaml.
      5. Load ablation file; resolve the requested tuple; write
         ablation_snapshot.yaml.
      6. Load Graph FSM spec; write graph_fsm.yaml.
      7. Enforce group_quotient cross-field invariant.
      8. Touch metric/result/score JSONL sinks.
      9. Return 0.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Step 1 - Validate --ablation pattern
    # ------------------------------------------------------------------
    try:
        _validate_ablation_flag(args.ablation)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 2 - Derive run_id and resolve output directory
    # ------------------------------------------------------------------
    run_id = derive_run_id(args.experiment, args.ablation, args.seed)
    output_was_explicit = args.output is not None

    try:
        output_dir = _resolve_output_dir(args.output, run_id, output_was_explicit)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 3 - Seed RNGs
    # ------------------------------------------------------------------
    seeding.set_seed(args.seed)

    # ------------------------------------------------------------------
    # Step 4 - Load config, apply CLI overrides, snapshot
    # ------------------------------------------------------------------
    try:
        raw_config = config_mod.load(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: config file not found - {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        msg = str(exc)
        if "schema_version" in msg:
            print(f"ERROR: config schema_version mismatch - {msg}", file=sys.stderr)
        else:
            print(f"ERROR: invalid config - {msg}", file=sys.stderr)
        return 1

    try:
        merged_config = _apply_cli_overrides(raw_config, args)
    except Exception as exc:
        print(f"ERROR: failed to apply CLI overrides to config - {exc}", file=sys.stderr)
        return 1

    config_mod.dump(merged_config, output_dir / "config_snapshot.yaml")

    # ------------------------------------------------------------------
    # Step 5 - Load ablation file, resolve tuple, snapshot
    # ------------------------------------------------------------------
    try:
        ablation_file = ablation_flags_mod.load(args.ablation_file)
    except FileNotFoundError as exc:
        print(f"ERROR: ablation file not found - {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        msg = str(exc)
        if "schema_version" in msg:
            print(
                f"ERROR: ablation file schema_version mismatch - {msg}",
                file=sys.stderr,
            )
        else:
            print(f"ERROR: invalid ablation file - {msg}", file=sys.stderr)
        return 1

    try:
        ablation_tuple = ablation_flags_mod.get(ablation_file, args.ablation)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _write_ablation_snapshot(ablation_tuple, output_dir)

    # ------------------------------------------------------------------
    # Step 6 - Load Graph FSM spec, snapshot
    # ------------------------------------------------------------------
    try:
        fsm_spec = graph_fsm_spec_mod.load(merged_config.graph_fsm_path)
    except FileNotFoundError as exc:
        print(f"ERROR: graph FSM file not found - {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        msg = str(exc)
        if "schema_version" in msg:
            print(
                f"ERROR: graph FSM schema_version mismatch - {msg}",
                file=sys.stderr,
            )
        else:
            print(f"ERROR: invalid graph FSM spec - {msg}", file=sys.stderr)
        return 1

    graph_fsm_spec_mod.dump(fsm_spec, output_dir / "graph_fsm.yaml")

    # ------------------------------------------------------------------
    # Step 7 - Cross-field invariant: group_quotient_enabled => group_spec_path
    # ------------------------------------------------------------------
    if ablation_tuple.group_quotient_enabled and merged_config.group_spec_path is None:
        print(
            "ERROR: ablation has group_quotient_enabled=True but config.group_spec_path "
            "is not set. Provide a group_spec_path in the config or select an ablation "
            "with group_quotient_enabled=False.",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Step 8 - Dispatch or noop
    # ------------------------------------------------------------------

    # Phase 0 noop: touch empty JSONL sinks and return immediately.
    if args.skip_train and args.skip_eval:
        _touch_empty_jsonl_files(output_dir)
        print(f"run_id: {run_id}  output: {output_dir}")
        return 0

    # Phase 1+: create the JSONL sink files so writers can append to them.
    _touch_empty_jsonl_files(output_dir)

    # Dispatch to experiment runner.
    if args.experiment == "E0":
        from nga.arch.graph_fsm import GraphFSM
        from nga.exp.e0_mnist import run_e0

        fsm = GraphFSM(fsm_spec)
        result = run_e0(
            config=merged_config,
            ablation=ablation_tuple,
            fsm=fsm,
            run_id=run_id,
            output_dir=output_dir,
            seed=args.seed,
        )
        print(
            f"E0 result: accuracy={result.accuracy:.4f} "
            f"low_margin_accuracy={result.low_margin_accuracy:.4f} "
            f"confusion_graph_density={result.confusion_graph_density:.4f}"
        )
        print(f"run_id: {run_id}  output: {output_dir}")
        return 0

    if args.experiment == "E1":
        from nga.arch.graph_fsm import GraphFSM
        from nga.exp.e1_synthetic_babyai import run_e1

        fsm = GraphFSM(fsm_spec)
        result = run_e1(
            config=merged_config,
            ablation=ablation_tuple,
            fsm=fsm,
            run_id=run_id,
            output_dir=output_dir,
            seed=args.seed,
        )
        print(
            f"E1 result: accuracy={result.accuracy:.4f} "
            f"no_mask={result.accuracy_no_mask:.4f} "
            f"uplift={result.mask_accuracy_uplift:+.4f} "
            f"illegal_rate={result.illegal_transition_rate:.4f}"
        )
        print(f"run_id: {run_id}  output: {output_dir}")
        return 0

    if args.experiment == "E2":
        from nga.arch.graph_fsm import GraphFSM
        from nga.exp.e2_real_babyai import run_e2

        fsm = GraphFSM(fsm_spec)
        result = run_e2(
            config=merged_config, ablation=ablation_tuple, fsm=fsm,
            run_id=run_id, output_dir=output_dir, seed=args.seed,
        )
        print(
            f"E2 result: accuracy={result.accuracy:.4f} "
            f"no_mask={result.accuracy_no_mask:.4f} "
            f"uplift={result.mask_accuracy_uplift:+.4f} "
            f"illegal_rate={result.illegal_transition_rate:.4f} "
            f"n_test={result.n_test}"
        )
        print(f"run_id: {run_id}  output: {output_dir}")
        return 0

    if args.experiment == "E3":
        from nga.arch.graph_fsm import GraphFSM
        from nga.exp.e3_hyperbolic_vs_euclidean import run_e3

        fsm = GraphFSM(fsm_spec)
        result = run_e3(
            config=merged_config,
            ablation=ablation_tuple,
            fsm=fsm,
            run_id=run_id,
            output_dir=output_dir,
            seed=args.seed,
        )
        print(
            f"E3 result: hyperbolic_d8={result.hyperbolic_d8_acc:.4f} "
            f"euclidean_d16={result.euclidean_d16_acc:.4f} "
            f"uplift={result.hyperbolic_uplift:+.4f} "
            f"cells={len(result.cells)}"
        )
        print(f"run_id: {run_id}  output: {output_dir}")
        return 0

    if args.experiment == "E4":
        from nga.exp.e4_singularity_auroc import run_e4

        sources = args.source_runs or ["E0_A0_seed42", "E1_A0_seed42"]
        result = run_e4(
            runs_root=args.runs_root,
            source_run_ids=sources,
            run_id=run_id,
            output_dir=output_dir,
            seed=args.seed,
        )
        print(
            f"E4 result: margin_auroc={result.margin_auroc_value:.4f} "
            f"sigma_auroc={result.sigma_auroc_value:.4f} "
            f"sigma_uplift={result.sigma_uplift:+.4f} "
            f"n_samples={result.n_samples} sources={result.sources}"
        )
        print(f"run_id: {run_id}  output: {output_dir}")
        return 0

    if args.experiment == "E5":
        from nga.arch.graph_fsm import GraphFSM
        from nga.exp.e5_idf_ablation import run_e5

        fsm = GraphFSM(fsm_spec)
        result = run_e5(
            config=merged_config,
            ablation=ablation_tuple,
            fsm=fsm,
            run_id=run_id,
            output_dir=output_dir,
            seed=args.seed,
        )
        print(
            f"E5 result: accuracy_idf={result.accuracy_idf:.4f} "
            f"accuracy_no_idf={result.accuracy_no_idf:.4f} "
            f"idf_uplift={result.idf_uplift:+.4f} "
            f"n_samples={result.n_samples} n_features={result.n_features}"
        )
        print(f"run_id: {run_id}  output: {output_dir}")
        return 0

    if args.experiment == "E6":
        from nga.arch.graph_fsm import GraphFSM
        from nga.exp.e6_group_quotient_attention import run_e6

        fsm = GraphFSM(fsm_spec)
        result = run_e6(
            config=merged_config, ablation=ablation_tuple, fsm=fsm,
            run_id=run_id, output_dir=output_dir, seed=args.seed,
        )
        print(f"E6 result: standard_flops={result.standard_flops:,} "
              f"orbit_pair_flops={result.orbit_pair_flops:,} "
              f"reduction={result.flops_reduction_ratio:+.4f} "
              f"parity_l2={result.output_parity_l2:.2e}")
        print(f"run_id: {run_id}  output: {output_dir}")
        return 0

    if args.experiment == "E7":
        from nga.arch.graph_fsm import GraphFSM
        from nga.exp.e7_reservoir_vs_endtoend import run_e7

        fsm = GraphFSM(fsm_spec)
        result = run_e7(
            config=merged_config, ablation=ablation_tuple, fsm=fsm,
            run_id=run_id, output_dir=output_dir, seed=args.seed,
        )
        print(
            f"E7 result: acc_endtoend={result.accuracy_endtoend:.4f} "
            f"acc_reservoir={result.accuracy_reservoir:.4f} "
            f"acc_ratio={result.accuracy_ratio:.4f} "
            f"params_endtoend={result.n_trainable_params_endtoend:,} "
            f"params_reservoir={result.n_trainable_params_reservoir:,} "
            f"param_ratio={result.param_ratio:.4f}"
        )
        print(f"run_id: {run_id}  output: {output_dir}")
        return 0

    if args.experiment == "E8":
        from nga.arch.graph_fsm import GraphFSM
        from nga.exp.e8_transfer import run_e8

        fsm = GraphFSM(fsm_spec)
        result = run_e8(
            config=merged_config,
            ablation=ablation_tuple,
            fsm=fsm,
            run_id=run_id,
            output_dir=output_dir,
            seed=args.seed,
        )
        print(
            f"E8 result: accuracy_train={result.accuracy_train:.4f} "
            f"accuracy_test={result.accuracy_test:.4f} "
            f"transfer_gap={result.transfer_gap:+.4f} "
            f"n_train={result.n_train} n_test={result.n_test}"
        )
        print(f"run_id: {run_id}  output: {output_dir}")
        return 0

    # Other experiments will land in later phases.
    print(
        f"experiment {args.experiment!r} not yet implemented for Phase 1; "
        f"pass --skip-train --skip-eval for a noop check.",
        file=sys.stderr,
    )
    return 2


__all__ = ["build_parser", "derive_run_id", "main"]
