"""Unit tests for the diagnostic dataset generator + 3-state FSM
(Phase 19).

The diagnostic dataset is the FIRST real-world (non-synthetic) input
in the suite: it linearises the Kaggle "Disease Prediction from
Symptoms" CSV into per-patient parse traces. The FSM has 3 states
(OBSERVING_FEW / OBSERVING_ENOUGH / DIAGNOSED) and the walker resolves
the count-conditional FEW -> {FEW, ENOUGH} ambiguity at runtime by
the running symptom count.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nga.arch.graph_fsm import GraphFSM
from nga.exp.dataset_diagnostic import (
    ALPHABET_SIZE,
    FEATURE_DIM,
    PROMOTION_THRESHOLD,
    generate_diagnostic_dataset,
    linearise_patient,
    load_diagnostic_csv,
    train_test_split_by_patient,
    walk_fsm_diagnostic,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FSM_PATH = REPO_ROOT / "tests/fixtures/graphs/diagnostic.fsm.yaml"
TRAIN_CSV = REPO_ROOT / "tests/fixtures/data/diagnostic_training.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diagnostic_fsm() -> GraphFSM:
    return GraphFSM.from_yaml(FSM_PATH)


@pytest.fixture(scope="module")
def small_dataset(diagnostic_fsm: GraphFSM):
    return generate_diagnostic_dataset(
        fsm=diagnostic_fsm,
        csv_path=TRAIN_CSV,
        n_patients=40,
        seed=42,
        illegal_temptation_fraction=0.10,
    )


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def test_diagnostic_csv_loads():
    """The Kaggle CSV parses to 4920 patients, 132 symptom columns,
    and 41 distinct prognoses."""
    syms, indicators, diagnoses = load_diagnostic_csv(TRAIN_CSV)
    assert len(syms) == 132, f"expected 132 symptom cols; got {len(syms)}"
    assert len(indicators) == 4920, (
        f"expected 4920 patients; got {len(indicators)}"
    )
    assert len(diagnoses) == 4920
    assert len(set(diagnoses)) == 41, (
        f"expected 41 distinct prognoses; got {len(set(diagnoses))}"
    )
    # All entries are 0/1.
    for row in indicators[:50]:
        assert all(v in (0, 1) for v in row)
    # Symptom names disambiguated to be unique.
    assert len(set(syms)) == 132


# ---------------------------------------------------------------------------
# Linearisation
# ---------------------------------------------------------------------------


def test_linearise_per_patient_emits_correct_length():
    """sequence length == count(TRUE symptoms) + 1 diagnosis."""
    syms, indicators, diagnoses = load_diagnostic_csv(TRAIN_CSV)
    sym_id_for = {n: i for i, n in enumerate(syms)}
    diag_id_for = {n: 132 + i for i, n in enumerate(sorted(set(diagnoses)))}
    for row, diag in list(zip(indicators, diagnoses))[:10]:
        seq = linearise_patient(row, diag, sym_id_for, diag_id_for)
        n_true = sum(row)
        assert len(seq) == n_true + 1
        assert seq[-1] == diag


def test_linearise_canonical_order():
    """Same indicator vector + diagnosis must produce the same sequence
    every call (deterministic, by column index)."""
    syms, indicators, diagnoses = load_diagnostic_csv(TRAIN_CSV)
    sym_id_for = {n: i for i, n in enumerate(syms)}
    diag_id_for = {n: 132 + i for i, n in enumerate(sorted(set(diagnoses)))}
    row, diag = indicators[123], diagnoses[123]
    a = linearise_patient(row, diag, sym_id_for, diag_id_for)
    b = linearise_patient(row, diag, sym_id_for, diag_id_for)
    assert a == b
    # Tokens preceding the diagnosis are sorted by column index, i.e.
    # the order they appear in the symptom_names list.
    sym_only = a[:-1]
    name_to_idx = {n: i for i, n in enumerate(syms)}
    indices = [name_to_idx[t] for t in sym_only]
    assert indices == sorted(indices), (
        f"linearised symptoms must be in column-index order; got {indices}"
    )


# ---------------------------------------------------------------------------
# FSM walker
# ---------------------------------------------------------------------------


def test_walk_fsm_diagnostic_legal_per_step(diagnostic_fsm: GraphFSM):
    """Every emitted (prev, observed, next) triple is a legal FSM
    transition under the 3-state diagnostic FSM."""
    syms, indicators, diagnoses = load_diagnostic_csv(TRAIN_CSV)
    sym_id_for = {n: i for i, n in enumerate(syms)}
    diag_id_for = {n: 132 + i for i, n in enumerate(sorted(set(diagnoses)))}
    diagnosis_set = set(diag_id_for.keys())
    for k in range(8):
        row, diag = indicators[k * 7], diagnoses[k * 7]
        if sum(row) < PROMOTION_THRESHOLD:
            continue
        seq = linearise_patient(row, diag, sym_id_for, diag_id_for)
        triples = walk_fsm_diagnostic(
            seq, diagnostic_fsm, diagnosis_set=diagnosis_set
        )
        for prev, _tok, nxt in triples:
            assert diagnostic_fsm.is_legal_transition(prev, nxt), (
                f"illegal transition {prev!r} -> {nxt!r}"
            )
        # Last triple lands in DIAGNOSED.
        assert triples[-1][2] == "DIAGNOSED"


# ---------------------------------------------------------------------------
# Materialised arrays
# ---------------------------------------------------------------------------


def test_diagnostic_dataset_features_one_hot(small_dataset):
    """Each row of X is zero-padded one-hot of the observed token over
    the 173-entry alphabet (padded to 176 dims)."""
    ds = small_dataset
    assert ds.X.shape[1] == FEATURE_DIM == 176
    # Each row sums to exactly 1.0 (one-hot).
    assert np.allclose(ds.X.sum(axis=1), 1.0)
    # Active dimension is in [0, 173).
    nz_cols = np.argmax(ds.X, axis=1)
    assert int(nz_cols.max()) < ALPHABET_SIZE
    assert int(nz_cols.min()) >= 0


def test_diagnostic_adversarial_marked(small_dataset):
    """Some samples must be flagged adversarial (the temptation
    fraction is 10% in the small_dataset fixture)."""
    ds = small_dataset
    assert ds.is_adversarial.dtype == bool
    n_adv = int(ds.is_adversarial.sum())
    assert n_adv > 0, "expected at least one adversarial sample at p=0.10"
    # The total fraction is bounded above by the temptation fraction +
    # noise. With 10% target the realised fraction should be in (0, 0.5).
    frac = n_adv / max(len(ds.samples), 1)
    assert 0.0 < frac < 0.5


# ---------------------------------------------------------------------------
# Determinism + split
# ---------------------------------------------------------------------------


def test_diagnostic_deterministic_under_seed(diagnostic_fsm: GraphFSM):
    """Same seed + same CSV -> bit-identical numpy arrays."""
    a = generate_diagnostic_dataset(
        fsm=diagnostic_fsm,
        csv_path=TRAIN_CSV,
        n_patients=30,
        seed=7,
        illegal_temptation_fraction=0.05,
    )
    b = generate_diagnostic_dataset(
        fsm=diagnostic_fsm,
        csv_path=TRAIN_CSV,
        n_patients=30,
        seed=7,
        illegal_temptation_fraction=0.05,
    )
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y_next, b.y_next)
    assert np.array_equal(a.is_adversarial, b.is_adversarial)
    assert np.array_equal(a.is_diagnosis_step, b.is_diagnosis_step)


def test_diagnostic_train_test_split_disjoint_by_patient(small_dataset):
    """train and test partitions hold MUTUALLY EXCLUSIVE patient_ids
    (so a single patient never appears on both sides of the split)."""
    train_ds, test_ds = train_test_split_by_patient(
        small_dataset, seed=42, test_fraction=0.25
    )
    train_pids = {s.patient_id for s in train_ds.samples}
    test_pids = {s.patient_id for s in test_ds.samples}
    assert train_pids.isdisjoint(test_pids)
    assert len(train_pids) + len(test_pids) == small_dataset.n_patients
    assert len(train_ds.samples) + len(test_ds.samples) == len(
        small_dataset.samples
    )
