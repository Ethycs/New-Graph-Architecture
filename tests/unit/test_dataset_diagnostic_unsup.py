"""Unit tests for the unsupervised diagnostic dataset (Phase 19B).

The diagnostic CSV has 4920 patients x 132 binary symptom indicators
(plus a held-out prognosis column). The unsupervised loader returns the
binary matrix and keeps the disease labels separately for evaluation
only -- they must NOT be touched in training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nga.exp.dataset_diagnostic_unsup import (
    dataset_unsup_summary,
    embed_symptoms_in_poincare,
    load_diagnostic_csv_as_vectors,
    markov_trajectory_for_patient,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = REPO_ROOT / "tests/fixtures/data/diagnostic_training.csv"


def test_load_csv_as_vectors_shape():
    """X is (4920, 132), labels are length 4920, 41 distinct diseases."""
    X, labels, sym_names, disease_names = load_diagnostic_csv_as_vectors(
        TRAIN_CSV
    )
    assert X.shape == (4920, 132), f"unexpected X shape {X.shape}"
    assert len(labels) == 4920, f"unexpected #labels {len(labels)}"
    assert len(sym_names) == 132
    assert len(disease_names) == 41
    # Binary entries.
    assert np.all((X == 0) | (X == 1))


def test_embed_symptoms_in_poincare_in_ball():
    """Every embedding has L2 norm strictly less than 1."""
    X, _labels, _sym_names, _disease_names = load_diagnostic_csv_as_vectors(
        TRAIN_CSV
    )
    # Use a small subset for speed.
    Xs = X[:100]
    Z = embed_symptoms_in_poincare(Xs, dim=8, seed=0)
    norms = np.linalg.norm(Z, axis=1)
    assert Z.shape == (100, 8)
    assert np.all(norms < 1.0), (
        f"max norm {norms.max()} should be strictly < 1 (Poincare ball)"
    )


def test_embed_deterministic_under_seed():
    """Same seed -> identical embeddings."""
    X, _l, _s, _d = load_diagnostic_csv_as_vectors(TRAIN_CSV)
    Xs = X[:50]
    Z1 = embed_symptoms_in_poincare(Xs, dim=8, seed=123)
    Z2 = embed_symptoms_in_poincare(Xs, dim=8, seed=123)
    assert np.allclose(Z1, Z2)
    # Different seed -> different embedding (with high probability).
    Z3 = embed_symptoms_in_poincare(Xs, dim=8, seed=999)
    assert not np.allclose(Z1, Z3)


def test_markov_trajectory_random_order():
    """Same patient + different seeds -> different orderings (typical case)."""
    X, _l, sym_names, _d = load_diagnostic_csv_as_vectors(TRAIN_CSV)
    sym_id_for = {n: i for i, n in enumerate(sym_names)}
    # Pick a patient with at least 5 TRUE symptoms so a permutation
    # difference is likely.
    sym_counts = X.sum(axis=1)
    pid = int(np.argmax(sym_counts))
    assert int(sym_counts[pid]) >= 5

    t1 = markov_trajectory_for_patient(X[pid], sym_id_for, seed=1)
    t2 = markov_trajectory_for_patient(X[pid], sym_id_for, seed=99)
    assert t1 != t2, (
        f"two different seeds produced identical orderings for patient {pid}"
    )


def test_markov_trajectory_same_set():
    """Same patient + any seed -> same SET of tokens."""
    X, _l, sym_names, _d = load_diagnostic_csv_as_vectors(TRAIN_CSV)
    sym_id_for = {n: i for i, n in enumerate(sym_names)}
    pid = 0
    t1 = sorted(markov_trajectory_for_patient(X[pid], sym_id_for, seed=1))
    t2 = sorted(markov_trajectory_for_patient(X[pid], sym_id_for, seed=99))
    assert t1 == t2, f"set of tokens disagreed between seeds: {t1} vs {t2}"
    # Length matches the patient's TRUE symptom count.
    assert len(t1) == int(X[pid].sum())


def test_summary_stats():
    """mean_symptoms_per_patient on this CSV is between 5 and 15."""
    X, labels, _sym_names, _disease_names = load_diagnostic_csv_as_vectors(
        TRAIN_CSV
    )
    s = dataset_unsup_summary(X, labels)
    assert s["n_patients"] == 4920
    assert s["n_symptoms"] == 132
    assert s["n_diseases"] == 41
    assert 5.0 <= s["mean_symptoms_per_patient"] <= 15.0, (
        f"mean_symptoms_per_patient = {s['mean_symptoms_per_patient']} "
        f"out of expected band [5, 15]"
    )
