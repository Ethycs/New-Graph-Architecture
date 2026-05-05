# Test Fixtures and Pytest Layout

**Status:** decided
**Cluster:** setup

## What

Pytest layout, shared fixtures, golden-file strategy, ablation parametrization, property tests, determinism check, CI tiers, test data, and mocking policy. This note is the contract every Phase 0 and Phase 1 test file conforms to. It is written before any code so that the first commit under `tests/` has a place to land.

## Directory layout

```
tests/
  conftest.py                       # session-wide fixtures (see Fixtures below)
  unit/
    test_config.py                  # one file per driver, mirrors docs/drivers/*
    test_graph_fsm_spec.py
    test_typed_score_record.py
    test_ablation_flags.py
    test_metrics_jsonl.py
    test_results_jsonl.py
    test_cli_runner.py
    test_legality_mask.py           # arch-atom unit tests, named test_<atom>.py
    test_margin_uncertainty.py
    test_singularity_detector.py
  integration/
    test_phase0_noop.py             # Phase 0 acceptance test
    test_cli_writes_run_dir.py      # CLI + drivers wired together, no model
    test_metric_collector_to_evidence.py
  e2e/
    test_e0_mnist.py                # Phase 1 acceptance test (E0 end-to-end)
    test_e1_synthetic_babyai.py     # Phase 2 acceptance test
    test_ablation_matrix_e0.py      # parametrized A0..A9 sweep on E0
  property/
    test_typed_score_record_props.py
    test_graph_fsm_spec_props.py
    test_metrics_jsonl_props.py
  fixtures/
    configs/
      mnist_minimal.yaml            # E0 + A0 minimal valid config
      babyai_synthetic_minimal.yaml # E1 + A0
    graphs/
      mnist.fsm.yaml                # 10-vertex digit graph
      tiny_3state.fsm.yaml          # 3-vertex toy FSM for unit tests
    ablations/
      ablations.yaml                # A0..A9 (mirror of configs/ablations.yaml)
    records/
      one_typed_score_record.json   # canonical TypedScoreRecord example
      malformed_score_missing_dist.json
  golden/
    e0/
      A0_seed42/
        metrics.jsonl
        results.jsonl
        scores.jsonl
        ablation_snapshot.yaml
      A1_seed42/
        metrics.jsonl
        ...
    e1/
      A0_seed42/
        ...
```

Naming rules: `test_<driver>.py` for the seven drivers; `test_<atom>.py` for arch atoms (one-to-one with `docs/arch/<atom>.md`); `test_e<N>.py` for experiments E0..E9; `test_<thing>_props.py` for property tests; integration files describe the wiring under test.

## Phase 0 acceptance test (canonical)

Translates the build-order acceptance line into one pytest function. Lives in `tests/integration/test_phase0_noop.py`.

```python
def test_phase0_noop_writes_empty_streams(
    tmp_run_dir, minimal_config, ablation_yaml_path, fake_graph_fsm_path,
    seeded_rng, monkeypatch,
):
    """
    `python run.py --experiment E0 --ablation A0 --config <minimal>
                  --seed 42` must:
      1. load and validate the config (schema_version 1.0)
      2. validate the graph FSM spec (vertex_count == len(vertices))
      3. resolve the A0 ablation tuple (all flags True)
      4. write four files into runs/E0_A0_seed42/:
         - config_snapshot.yaml
         - ablation_snapshot.yaml
         - graph_fsm.yaml
         - metrics.jsonl, results.jsonl, scores.jsonl  (all empty)
    """
    from run import main
    rc = main([
        "--experiment", "E0", "--ablation", "A0",
        "--config",     str(minimal_config),
        "--seed",       "42",
        "--output",     str(tmp_run_dir),
        "--skip-train", "--skip-eval",
    ])
    assert rc == 0
    for name in ("config_snapshot.yaml", "ablation_snapshot.yaml",
                 "graph_fsm.yaml"):
        assert (tmp_run_dir / name).exists(), f"missing {name}"
    for name in ("metrics.jsonl", "results.jsonl", "scores.jsonl"):
        path = tmp_run_dir / name
        assert path.exists(), f"missing {name}"
        assert path.read_bytes() == b"", f"{name} should be empty for noop"

    snap = yaml.safe_load((tmp_run_dir / "ablation_snapshot.yaml").read_text())
    assert snap["graph_mask_enabled"] is True
    assert snap["typed_scores_enabled"] is True
```

This test is the gate for Phase 0 closing. It must pass before any arch atom lands.

## Phase 1 acceptance test (canonical)

Lives in `tests/e2e/test_e0_mnist.py`. Uses golden-file comparison for `metrics.jsonl`.

```python
@pytest.mark.slow
def test_e0_mnist_end_to_end_matches_golden(
    tmp_run_dir, mnist_config, seeded_rng, golden_dir, update_goldens,
):
    """
    Phase 1 acceptance: E0 runs end-to-end on sklearn digits and writes
    accuracy + low_margin_accuracy + confusion_graph_density into
    metrics.jsonl, plus per-sample singular_flag into results.jsonl.
    """
    from run import main
    rc = main([
        "--experiment", "E0", "--ablation", "A0",
        "--config",     str(mnist_config),
        "--seed",       "42",
        "--output",     str(tmp_run_dir),
    ])
    assert rc == 0

    metrics = read_jsonl(tmp_run_dir / "metrics.jsonl")
    names = {r["metric_name"] for r in metrics}
    assert {"accuracy", "low_margin_accuracy",
            "confusion_graph_density"} <= names

    results = read_jsonl(tmp_run_dir / "results.jsonl")
    assert all("singular_flag" in r for r in results)
    acc = next(r["value"] for r in metrics if r["metric_name"] == "accuracy")
    assert acc >= 0.90

    compare_golden(
        tmp_run_dir / "metrics.jsonl",
        golden_dir / "e0" / "A0_seed42" / "metrics.jsonl",
        update=update_goldens,
        ignore_fields={"timestamp"},
    )
```

`compare_golden` (helper in `tests/conftest.py`) does a line-by-line JSON diff after stripping `timestamp`, and either fails with a unified diff or rewrites the golden when `--update-goldens` is set.

## Driver contract tests

One file per driver. Each contract test asserts the driver's wire format is round-trippable, version-checked, and rejects malformed input.

| Driver | Test file | Asserts |
|---|---|---|
| Config (YAML) | `tests/unit/test_config.py` | Pydantic load of every fixture config; round-trip dump-then-load equals input; missing `schema_version` raises; unsupported major raises; `group_spec_path` must be set iff `group_quotient_enabled`; CLI override merge preserves type. |
| Graph FSM Spec | `tests/unit/test_graph_fsm_spec.py` | `vertex_count == len(vertices)`; `edge_count == len(edges)`; every edge endpoint references a real vertex id; `coordinates.dimension == config.embedding_dim`; `node_embeddings` length matches; `build_legality_matrix` shape `(V, V)` with correct binary entries. |
| Typed Score Record | `tests/unit/test_typed_score_record.py` | Round-trip JSON-line write-then-read equals input; `0 <= confidence <= 1`; `len(dist) == vertex_count`; `len(z_H) == embedding_dim`; `softmax_distribution` sums to 1.0 within 1e-6; missing required field raises; unknown major version raises. |
| Ablation Flag Set | `tests/unit/test_ablation_flags.py` | All ten ablations load; A0 has every flag True; A1..A3, A5..A9 differ from A0 on exactly one flag; A4 differs in routing only (computed flag tracked); unknown ablation id raises; flag schema rejects extra keys. |
| metrics.jsonl | `tests/unit/test_metrics_jsonl.py` | Append-only writer never truncates; reader streams; round-trip equality; `metric_name` warns (does not raise) on unknown name; `split` rejects unknown enum; `schema_version` mismatch raises. |
| results.jsonl | `tests/unit/test_results_jsonl.py` | Round-trip; `y_true` and `y_hat` validated against vertex ids; `-1 <= margin <= 1`; optional fields absent on read default to None; `transition_legal` consistent with FSM edges. |
| CLI Runner | `tests/unit/test_cli_runner.py` | `argparse` accepts every flag in the table; `run_id` derived as `<exp>_<abl>_seed<seed>`; output dir auto-created; refuses to overwrite non-empty existing dir unless `--output` differs; exit code 1 on schema-version mismatch; exit code 0 on `--skip-train --skip-eval`. |

Each contract test uses its driver's fixture (`fake_typed_score_record`, `fake_graph_fsm`, etc.) plus golden round-trip data under `tests/fixtures/records/`.

## Fixtures (conftest.py)

`tests/conftest.py` contains the shared fixtures. Numbered with scope and dependencies.

1. `tmp_run_dir` (function). Creates `tmp_path / "run"`, yields `pathlib.Path`, pytest cleans up. No deps.
2. `seeded_rng` (function). Sets seeds across `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all` from env var `TEST_SEED` (default 42); returns the int. No deps.
3. `golden_dir` (session). Returns `Path("tests/golden")`. No deps.
4. `update_goldens` (session). Reads env var `UPDATE_GOLDENS` (set by the `--update-goldens` CLI flag via `pytest_addoption`); returns bool. No deps.
5. `fake_graph_fsm` (session). Loads `tests/fixtures/graphs/tiny_3state.fsm.yaml` and returns a parsed `GraphFSMSpec` (3 vertices, 2 edges, 8-dim coords). Depends on the fsm-loader being importable.
6. `fake_graph_fsm_path` (session). Returns the path to `tiny_3state.fsm.yaml` for tests that need a path string rather than a parsed object. No deps.
7. `mnist_fsm` (session). Loads `tests/fixtures/graphs/mnist.fsm.yaml` (10 digit vertices). Used by E0 tests.
8. `minimal_config` (session). Loads `tests/fixtures/configs/mnist_minimal.yaml` and returns a parsed `ConfigSchema`. Depends on `mnist_fsm` for path consistency.
9. `mnist_config` (session). Same as `minimal_config` but as a Path, for CLI-level tests that re-load it.
10. `ablation_yaml_path` (session). Path to `tests/fixtures/ablations/ablations.yaml`. No deps.
11. `fake_typed_score_record` (function). Returns one valid `TypedScoreRecord` with `dist` of length 3 (matches `fake_graph_fsm`), `z_H` of length 8. Depends on `fake_graph_fsm` for length consistency.
12. `recorded_metrics_jsonl` (function). Yields a writer that appends to `tmp_run_dir / "metrics.jsonl"` and a reader that returns the list of records on demand; lets tests assert on what was written. Depends on `tmp_run_dir`.
13. `recorded_results_jsonl` (function). Same pattern for `results.jsonl`. Depends on `tmp_run_dir`.
14. `mock_classifier` (function). A deterministic `sklearn.linear_model.LogisticRegression` pre-fit on a 64-sample digits subset cached at session start; used by E0 unit tests that want the classifier without paying the fit cost. Depends on `mnist_data`.
15. `mnist_data` (session). Returns `sklearn.datasets.load_digits()` once per session; `(X, y)` tuple. No deps.

Dependency graph: `minimal_config -> mnist_fsm`; `fake_typed_score_record -> fake_graph_fsm`; `recorded_metrics_jsonl -> tmp_run_dir`; `mock_classifier -> mnist_data`.

## Golden files

Decision: golden files live in git under `tests/golden/`. Justification: they are JSONL-text and YAML, compress well in git, are diffable in PRs, and the alternative (regenerate from a hash on every run) breaks the moment any non-determinism slips in (timestamp, dict-iteration order, float printing) and gives no signal about what changed. The cost is roughly 50-200 KB per golden run; for E0-E1 that is well under 5 MB total.

Location: `tests/golden/<exp_id>/<ablation>_seed<seed>/{metrics,results,scores}.jsonl` plus `ablation_snapshot.yaml`.

Regeneration: pass `--update-goldens` to pytest. Wired via `pytest_addoption` in `tests/conftest.py`:

```python
def pytest_addoption(parser):
    parser.addoption("--update-goldens", action="store_true", default=False,
                     help="Rewrite golden files instead of comparing.")

@pytest.fixture(scope="session")
def update_goldens(request):
    return request.config.getoption("--update-goldens")
```

Diff presentation on failure: `compare_golden` reads both files, parses each line as JSON, strips the ignore set (default `{"timestamp"}`), and on mismatch raises `AssertionError` with `difflib.unified_diff` output truncated to 200 lines.

Time-varying fields (`timestamp`) are always stripped before comparison. Float fields are compared with `pytest.approx(rel=1e-6)`; if a metric value drifts beyond that threshold the test fails and a human investigates rather than blindly regenerating.

## Ablation parametrization

A0..A9 are tuples; one parametrized test sweeps all ten on E0. Lives in `tests/e2e/test_ablation_matrix_e0.py`.

```python
ABLATIONS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]

@pytest.mark.slow
@pytest.mark.parametrize("ablation", ABLATIONS, ids=ABLATIONS)
def test_e0_runs_under_each_ablation(
    ablation, tmp_run_dir, mnist_config, seeded_rng, golden_dir, update_goldens,
):
    from run import main
    rc = main([
        "--experiment", "E0", "--ablation", ablation,
        "--config",     str(mnist_config),
        "--seed",       "42",
        "--output",     str(tmp_run_dir),
    ])
    assert rc == 0, f"E0 with {ablation} returned {rc}"
    metrics = read_jsonl(tmp_run_dir / "metrics.jsonl")
    assert any(r["metric_name"] == "accuracy" for r in metrics)
    compare_golden(
        tmp_run_dir / "metrics.jsonl",
        golden_dir / "e0" / f"{ablation}_seed42" / "metrics.jsonl",
        update=update_goldens,
        ignore_fields={"timestamp"},
    )
```

For ablations that depend on Phase 3+ components (A6 hyperbolic, A7 group quotient), tests are skipped with `pytest.skip` until the relevant phase lands; the skip message names the gating phase.

## Property tests

Use `hypothesis` for the four contracts where parse-emit symmetry is load-bearing.

1. `test_typed_score_record_roundtrip` (`tests/property/test_typed_score_record_props.py`). Generate records, serialize, parse back, assert equality.

   ```python
   @given(
       sample_id=st.text(min_size=1, max_size=32, alphabet=ALPHANUM),
       label=st.sampled_from(THREE_STATE_LABELS),
       confidence=st.floats(0.0, 1.0, allow_nan=False),
       margin=st.floats(-1.0, 1.0, allow_nan=False),
       dist=st.lists(st.floats(0.0, 10.0, allow_nan=False),
                     min_size=3, max_size=3),
       seed=st.integers(0, 2**31 - 1),
       step=st.integers(0, 1_000_000),
   )
   def test_typed_score_record_roundtrip(...):
       rec = TypedScoreRecord(schema_version="1.0", ...)
       parsed = TypedScoreRecord.from_json(rec.to_json())
       assert parsed == rec
   ```

2. `test_graph_fsm_spec_roundtrip` (`test_graph_fsm_spec_props.py`). Generate FSMs with 2-8 vertices, edges over those vertex ids, embedding dim in `{4, 8, 16}`. Serialize to YAML, load, assert equal. Catches off-by-one in `vertex_count`/`edge_count` validation.

3. `test_legality_matrix_idempotent`. For any generated FSM, `build_legality_matrix(load(dump(spec)))` equals `build_legality_matrix(spec)`.

4. `test_metrics_jsonl_append_read_invariant` (`test_metrics_jsonl_props.py`). Generate a list of `MetricsRecord`s, append each to a fresh file, read back; assert `read(write(records)) == records` and that file is exactly N lines.

5. `test_ablation_flag_one_off_invariant` (`tests/unit/test_ablation_flags.py`, parametrized). For each ablation in `{A1, A2, A3, A5, A6, A7, A8, A9}`, exactly one boolean flag differs from A0. A4 is the documented exception (computed-but-ignored).

Hypothesis settings: `deadline=None` for slow round-trips; `max_examples=200` for unit-level, `max_examples=50` for the FSM generator (more expensive).

## Determinism test

Single test in `tests/integration/test_determinism.py`. Critical for the reproducibility claim.

```python
@pytest.mark.slow
def test_e0_byte_identical_under_same_seed(tmp_path, mnist_config):
    from run import main
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        rc = main([
            "--experiment", "E0", "--ablation", "A0",
            "--config", str(mnist_config), "--seed", "42",
            "--output", str(out),
        ])
        assert rc == 0

    def strip_ts(path):
        return b"\n".join(
            json.dumps({k: v for k, v in json.loads(line).items()
                        if k != "timestamp"}, sort_keys=True).encode()
            for line in path.read_bytes().splitlines() if line
        )
    for name in ("metrics.jsonl", "results.jsonl", "scores.jsonl"):
        assert strip_ts(out_a / name) == strip_ts(out_b / name), name
```

If this fails, every other reproducibility-dependent claim is suspect; the test gates Phase 1 closure.

## CI tiers

| Tier | When | Tests included | Wall-clock budget |
|---|---|---|---|
| Smoke | every commit / pre-push hook | `tests/unit/` and `tests/property/` (excluding `@pytest.mark.slow`) | < 30 s |
| PR | on pull request | Smoke + `tests/integration/` + `tests/e2e/test_e0_mnist.py` (single-ablation) | < 5 min |
| Nightly | nightly cron on main | PR tier + `tests/e2e/test_ablation_matrix_e0.py` (all 10) + `tests/e2e/test_e1_synthetic_babyai.py` + determinism + golden-file diff report | < 30 min |

`@pytest.mark.slow` marks anything that loads a model or runs E0+ end-to-end. The smoke tier runs `pytest -m "not slow"`. Nightly runs `pytest` with no marker filter.

## Test data

Source: `sklearn.datasets.load_digits()`. 1797 samples, 8x8 = 64 features each, ten classes. Loaded once per pytest session via the `mnist_data` fixture.

Size: ~ 100 KB in memory; nothing on disk. No vendored CSV. No on-the-fly generator (digits are already a fixed sklearn artifact, so reproducibility is free).

Cache policy: sklearn caches at `~/scikit_learn_data/`. CI workers should pre-warm this in a setup step (`python -c "from sklearn.datasets import load_digits; load_digits()"`) so the first test run does not hit the network. Test code never downloads at test time.

For E1 (synthetic BabyAI), the fixture is generated procedurally inside `tests/fixtures/` from a fixed seed and committed as `tests/fixtures/babyai_synthetic_traces.jsonl` (~ 200 KB). Re-generation script is `tests/fixtures/_regen_babyai_synthetic.py`, run manually on schema bumps only.

## Mocking policy

Default: do not mock. The codebase aims for end-to-end reproducibility, and mocking removes the very behavior the integration tests are designed to verify.

Per-phase rules:

- Phase 0 (drivers). No mocking. Drivers are pure data; tests use real loader and writer code on real (small) fixtures.
- Phase 1 (E0 MNIST). No mocking. sklearn fits in milliseconds; real `LogisticRegression` is faster than wiring a fake. The `mock_classifier` fixture is a pre-fit real classifier, not a mock.
- Phase 2 (singularity). No mocking of the singularity detector itself. Allowed: a stub `confusion-graph` for unit tests of `singularity-detector` in isolation, but the integration test must use the real one.
- Phase 3 (hyperbolic). Allowed and recommended: when testing downstream consumers (`graph-prototype-vectors`, `hyperbolic-distance-loss`) in isolation, replace `hyperbolic-embedding` with a fixture that returns pre-computed coordinates from a saved tensor at `tests/fixtures/hyperbolic_coords.pt`. Reason: a real hyperbolic embedding fit takes minutes and dominates test time. Phase 3 e2e tests still use the real embedding.
- Phase 4 (group quotient). Same pattern: stubbed orbit table for unit tests, real one for e2e.
- Phase 5 (energy, reservoir). Frozen-encoder backbone is loaded from a checkpoint fixture; never freshly trained inside a test.

Networked services (none planned for the research codebase) are always mocked when they appear.

## What's NOT decided here

- Language and framework versions: see [language-framework.md](language-framework.md).
- Repo directory layout (where `run.py`, `arch/`, `exp/`, `configs/` live): deferred.
- CI runner choice: deferred (probably GitHub Actions; lock when language-framework decision lands).
- Coverage tool and threshold: deferred until after Phase 1 lands and a baseline coverage number exists.

## Links
- [README](../README.md)
- [build-order](../build-order.md)
- [drivers/_index.md](../drivers/_index.md)
