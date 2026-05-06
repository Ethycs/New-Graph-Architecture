"""Shared pytest fixtures and helpers. See docs/setup/test-fixtures.md §Fixtures.

Phase 0 scope: Phase 1+ fixtures (mock_classifier, mnist_data, recorded_*_jsonl,
fake_typed_score_record) deferred to when E0 lands.
"""
from __future__ import annotations

import difflib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Make src/ importable for tests (matches pyproject.toml [tool.pytest.ini_options].pythonpath).
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nga.drivers import (
    ablation_flags,
    config,
    graph_fsm_spec,
    seeding,
    typed_score_record,
)


# ---------------------------------------------------------------------------
# pytest_addoption hook
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Rewrite golden files instead of comparing.",
    )


# ---------------------------------------------------------------------------
# Helper functions (exported for use by tests)
# ---------------------------------------------------------------------------

def read_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return list of dicts. Empty file => []."""
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def compare_golden(
    actual: Path,
    expected: Path,
    update: bool,
    ignore_fields: set[str],
) -> None:
    """Line-by-line JSON diff after stripping ignore_fields.

    Time-varying field 'timestamp' is always added to ignore_fields automatically.

    On mismatch:
    - if update is True, rewrite expected with actual content.
    - else raise AssertionError with unified diff truncated to 200 lines.
    """
    ignore = set(ignore_fields) | {"timestamp"}

    def _strip(records: list[dict[str, Any]]) -> list[str]:
        """Return normalised JSON lines with ignore_fields removed."""
        result = []
        for record in records:
            cleaned = {k: v for k, v in record.items() if k not in ignore}
            result.append(json.dumps(cleaned, sort_keys=True))
        return result

    actual_records = read_jsonl_lines(actual)
    actual_lines = _strip(actual_records)

    if update:
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_text(actual.read_text(encoding="utf-8"), encoding="utf-8")
        return

    if not expected.exists():
        raise AssertionError(
            f"Golden file does not exist: {expected}\n"
            "Run with --update-goldens to create it."
        )

    expected_records = read_jsonl_lines(expected)
    expected_lines = _strip(expected_records)

    if actual_lines == expected_lines:
        return

    diff = list(
        difflib.unified_diff(
            expected_lines,
            actual_lines,
            fromfile=str(expected),
            tofile=str(actual),
            lineterm="",
        )
    )
    truncated = diff[:200]
    if len(diff) > 200:
        truncated.append(f"... ({len(diff) - 200} more lines truncated)")

    raise AssertionError(
        "Golden file mismatch. Run with --update-goldens to regenerate.\n"
        + "\n".join(truncated)
    )


# ---------------------------------------------------------------------------
# Fixtures - Phase 0
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def tmp_run_dir(tmp_path: Path) -> Path:
    """Fresh path for a run output directory.

    Returns tmp_path / "run" without pre-creating it - the CLI will mkdir.
    pytest cleans up tmp_path automatically after the test.
    """
    return tmp_path / "run"


@pytest.fixture(scope="function")
def seeded_rng() -> int:
    """Set deterministic seeds from TEST_SEED env var (default 42). Returns the int."""
    seed = int(os.environ.get("TEST_SEED", "42"))
    return seeding.set_seed(seed)


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Path to the golden files directory."""
    return REPO_ROOT / "tests" / "golden"


@pytest.fixture(scope="session")
def update_goldens(request: pytest.FixtureRequest) -> bool:
    """True when --update-goldens was passed on the pytest command line."""
    return bool(request.config.getoption("--update-goldens"))


@pytest.fixture(scope="session")
def fake_graph_fsm_path() -> Path:
    """Path to the tiny 3-state FSM fixture YAML."""
    return REPO_ROOT / "tests" / "fixtures" / "graphs" / "tiny_3state.fsm.yaml"


@pytest.fixture(scope="session")
def fake_graph_fsm(fake_graph_fsm_path: Path) -> graph_fsm_spec.GraphFSMSpec:
    """Parsed GraphFSMSpec for the tiny 3-state FSM (3 vertices, 2 edges, 8-dim coords)."""
    return graph_fsm_spec.load(fake_graph_fsm_path)


@pytest.fixture(scope="session")
def mnist_fsm_path() -> Path:
    """Path to the 10-vertex MNIST digit FSM fixture YAML."""
    return REPO_ROOT / "tests" / "fixtures" / "graphs" / "mnist.fsm.yaml"


@pytest.fixture(scope="session")
def mnist_fsm(mnist_fsm_path: Path) -> graph_fsm_spec.GraphFSMSpec:
    """Parsed GraphFSMSpec for the MNIST digit graph (10 vertices, dimension 16)."""
    return graph_fsm_spec.load(mnist_fsm_path)


@pytest.fixture(scope="session")
def minimal_config_path() -> Path:
    """Path to the minimal MNIST config YAML."""
    return REPO_ROOT / "tests" / "fixtures" / "configs" / "mnist_minimal.yaml"


@pytest.fixture(scope="session")
def minimal_config(
    minimal_config_path: Path,
    mnist_fsm: graph_fsm_spec.GraphFSMSpec,  # noqa: ARG001 - ensures FSM loads first
) -> config.Config:
    """Parsed Config for the minimal MNIST run configuration."""
    return config.load(minimal_config_path)


@pytest.fixture(scope="session")
def mnist_config(minimal_config_path: Path) -> Path:
    """Path to the minimal MNIST config YAML (alias for CLI-level tests)."""
    return minimal_config_path


@pytest.fixture(scope="session")
def ablation_yaml_path() -> Path:
    """Path to the ablations fixture YAML."""
    return REPO_ROOT / "tests" / "fixtures" / "ablations" / "ablations.yaml"


@pytest.fixture(scope="session")
def ablations(ablation_yaml_path: Path) -> ablation_flags.AblationFile:
    """Parsed AblationFile with the full A0-A9 ablation matrix."""
    return ablation_flags.load(ablation_yaml_path)


# ---------------------------------------------------------------------------
# TODO: Phase 1 fixtures (mock_classifier, mnist_data, recorded_metrics_jsonl,
#       recorded_results_jsonl, fake_typed_score_record) - add when E0 lands.
# ---------------------------------------------------------------------------
