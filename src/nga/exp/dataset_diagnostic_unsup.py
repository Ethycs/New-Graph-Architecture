"""Diagnostic dataset (Phase 19B) -- *unsupervised* per-patient symptom
matrix and Markov-randomised symptom-trajectory generator.

This is the FSM-free / label-free reformulation of the Kaggle
"Disease Prediction from Symptoms" CSV. Where ``dataset_diagnostic.py``
produces a tokenised parse trace under a 3-state FSM, this module
delivers:

  * ``X``: a binary patient x symptom matrix of shape (n_patients, 132).
  * ``disease_labels``: the ground-truth diagnosis per patient -- HELD
    OUT for evaluation only; never consumed by the architecture during
    training.
  * Hyperbolic embeddings into the Poincare ball via random projection.
  * A Markov-randomised trajectory generator: for any patient, return
    the patient's TRUE symptoms in a random order (deterministic given
    a seed). Different seeds give different orders; the underlying SET
    of symptoms is invariant.

The point: feed only the embeddings to a Riemannian k-means clustering
routine and ask whether the architecture's latent-clustering machinery
recovers the medical taxonomy from symptom co-occurrence alone.

This module deliberately does NOT depend on ``GraphFSM``; the discovered
cluster lattice IS the implicit FSM in 19B.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nga.arch.hyperbolic_embedding import embed_euclidean_to_poincare
from nga.exp.dataset_diagnostic import load_diagnostic_csv

__all__ = [
    "load_diagnostic_csv_as_vectors",
    "embed_symptoms_in_poincare",
    "markov_trajectory_for_patient",
    "dataset_unsup_summary",
]


# ---------------------------------------------------------------------------
# CSV -> matrix loader
# ---------------------------------------------------------------------------


def load_diagnostic_csv_as_vectors(
    path: str | Path,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Load the diagnostic CSV into an unsupervised matrix layout.

    Reuses ``dataset_diagnostic.load_diagnostic_csv`` for the parsing
    invariants (header sanity, integer 0/1 values, duplicate-column
    disambiguation, deterministic prognosis ordering).

    Returns
    -------
    X:
        Float array of shape ``(n_patients, 132)`` with binary symptom
        indicators.
    disease_labels:
        Per-patient ground-truth diagnosis strings (length n_patients).
        These are kept exclusively for *evaluation*; the unsupervised
        pipeline must never consume them.
    symptom_names:
        The 132 symptom column names in canonical order.
    disease_names:
        The 41 distinct prognoses sorted alphabetically.
    """
    symptom_names, indicators, diagnoses = load_diagnostic_csv(path)
    X = np.asarray(indicators, dtype=np.float64)
    disease_labels = list(diagnoses)
    disease_names = sorted(set(disease_labels))
    return X, disease_labels, list(symptom_names), disease_names


# ---------------------------------------------------------------------------
# Hyperbolic embedding
# ---------------------------------------------------------------------------


def embed_symptoms_in_poincare(
    X: np.ndarray,
    *,
    dim: int = 16,
    seed: int = 42,
) -> np.ndarray:
    """Project binary symptom vectors into the Poincare ball.

    The two-step recipe is:

      1. Random projection: a fixed Gaussian matrix ``R`` of shape
         ``(132, dim)`` (deterministic under ``seed``) maps the binary
         132-dim symptom vector to a ``dim``-dim Euclidean point.
         Entries scaled by ``1 / sqrt(132)`` to keep typical norms O(1).
      2. ``embed_euclidean_to_poincare`` (tanh squashing) maps each
         Euclidean point into the open unit ball.

    Parameters
    ----------
    X:
        Float array of shape ``(n_patients, 132)`` -- binary symptom
        matrix.
    dim:
        Output Poincare-ball dimension. Must be >= 1.
    seed:
        RNG seed for the random-projection matrix.

    Returns
    -------
    np.ndarray
        Poincare-ball points of shape ``(n_patients, dim)``. Every row
        has Euclidean L2 norm strictly less than 1.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_patients, n_symptoms); got {X.shape}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1; got {dim}")
    n_features = X.shape[1]
    rng = np.random.default_rng(int(seed))
    R = rng.standard_normal((n_features, int(dim))) / np.sqrt(float(n_features))
    Z = X @ R                                          # (n_patients, dim)
    return embed_euclidean_to_poincare(Z, scale=0.5)


# ---------------------------------------------------------------------------
# Markov-randomised symptom trajectory
# ---------------------------------------------------------------------------


def markov_trajectory_for_patient(
    symptom_vec: np.ndarray,
    symptom_id_for: dict[str, int],
    *,
    seed: int,
) -> list[str]:
    """Return one random ordering of a patient's TRUE symptom tokens.

    Given the patient's binary indicator vector (length 132) and the
    ``symptom_id_for`` map (name -> column index), pull out the names of
    the TRUE symptoms and shuffle them under a deterministic RNG keyed
    by ``seed``.

    Same patient + same seed -> same order.
    Same patient + different seed -> different order (with high prob).
    Same patient + any seed -> same SET of tokens (just permuted).

    Returns
    -------
    list[str]
        Symptom names in randomised order. Length == count of TRUE
        entries in ``symptom_vec``.
    """
    vec = np.asarray(symptom_vec).reshape(-1)
    if vec.shape[0] != len(symptom_id_for):
        raise ValueError(
            f"symptom_vec length {vec.shape[0]} disagrees with "
            f"len(symptom_id_for)={len(symptom_id_for)}"
        )
    # Sort symptom names by their column index so traversal is canonical.
    names_in_order = [
        name for name, _idx in sorted(symptom_id_for.items(), key=lambda kv: kv[1])
    ]
    true_names = [n for n, v in zip(names_in_order, vec) if int(v) == 1]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(true_names)  # type: ignore[arg-type]
    return list(true_names)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def dataset_unsup_summary(
    X: np.ndarray,
    disease_labels: list[str],
) -> dict:
    """Return summary statistics suitable for aggregate.py reporting.

    Parameters
    ----------
    X:
        Binary symptom matrix of shape ``(n_patients, n_symptoms)``.
    disease_labels:
        Ground-truth disease strings, length n_patients.

    Returns
    -------
    dict
        Keys: ``n_patients``, ``n_symptoms``, ``n_diseases``,
        ``mean_symptoms_per_patient``, ``min_symptoms_per_patient``,
        ``max_symptoms_per_patient``,
        ``mean_patients_per_disease``.
    """
    X = np.asarray(X)
    n_patients, n_symptoms = X.shape if X.ndim == 2 else (X.shape[0], 0)
    if len(disease_labels) != n_patients:
        raise ValueError(
            f"disease_labels length {len(disease_labels)} disagrees with "
            f"n_patients={n_patients}"
        )
    counts = X.sum(axis=1) if X.ndim == 2 else np.zeros(n_patients)
    n_diseases = len(set(disease_labels))
    mean_per_disease = (
        float(n_patients) / float(n_diseases) if n_diseases > 0 else 0.0
    )
    return {
        "n_patients": int(n_patients),
        "n_symptoms": int(n_symptoms),
        "n_diseases": int(n_diseases),
        "mean_symptoms_per_patient": float(np.mean(counts)) if n_patients else 0.0,
        "min_symptoms_per_patient": int(np.min(counts)) if n_patients else 0,
        "max_symptoms_per_patient": int(np.max(counts)) if n_patients else 0,
        "mean_patients_per_disease": float(mean_per_disease),
    }
