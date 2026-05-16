"""Smoke-test: run E30 on python_expr with a tiny config to validate the pipeline.

Should print regime-graph headline numbers in well under a minute on GPU.
"""
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(Path.home() / "models" / "hf")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "src"))

from nga.arch.graph_fsm import GraphFSM  # noqa: E402
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod  # noqa: E402
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH  # noqa: E402
from nga.exp.e30_pcg_extractor_pretrained import run_e30  # noqa: E402

dispatch = GRAMMAR_DISPATCH["python_expr"]
fsm = GraphFSM(graph_fsm_spec_mod.load(REPO_ROOT / dispatch["fsm_yaml_path"]))

out = REPO_ROOT / "runs" / "E30_smoke_python_expr"
if out.exists():
    shutil.rmtree(out)

result = run_e30(
    fsm=fsm,
    run_id="E30_smoke_python_expr",
    output_dir=out,
    seed=42,
    grammar="python_expr",
    n_programs=20,
    n_projection_train_epochs=10,
)

print()
print(f"grammar: {result.grammar}")
print(f"model: {result.model_id}, harvest_layer={result.harvest_layer}")
print(f"V_ground_truth: {result.V_ground_truth}")
print(f"n_total_steps: {result.n_total_steps}")
print(f"n_argmax_cells: {result.n_argmax_cells}")
print(f"n_regimes_after_merge: {result.n_regimes_after_merge}")
print(f"mean_purity_against_current_state: {result.mean_purity_against_current_state:.4f}")
print(f"mean_failure_rate: {result.mean_failure_rate:.4f}")
print(f"projection_next_state_acc: {result.projection_next_state_accuracy:.4f}")
print(f"aligned_hamming_at_target_V: {result.aligned_hamming_at_target_V}")
print(f"n_programs_truncated_to_context: {result.n_programs_truncated_to_context}")
print(
    f"timings: phase1={result.phase_1_wall_clock_seconds:.2f}s "
    f"phase2={result.phase_2_wall_clock_seconds:.2f}s "
    f"phase3={result.phase_3_wall_clock_seconds:.2f}s "
    f"phase4={result.phase_4_wall_clock_seconds:.2f}s "
    f"total={result.total_wall_clock_seconds:.2f}s"
)
