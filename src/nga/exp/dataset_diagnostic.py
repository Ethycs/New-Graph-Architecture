"""Diagnostic dataset (Phase 19) -- per-patient tensor-product
linearisation of the Kaggle "Disease Prediction from Symptoms" CSV.

This is the FIRST real-world (non-synthetic) dataset wired into the NGA
architecture. The transform from the raw 132-binary-symptom +
1-prognosis row into a tokenised parse trace is:

    For each patient with symptom set S and diagnosis d:
        sequence := [s_1, s_2, ..., s_|S|, d]
    where s_i are the patient's TRUE symptoms in CANONICAL order
    (by column index in the CSV header).

The token alphabet has 173 entries: 132 symptom tokens (column names)
plus 41 diagnosis tokens (distinct prognoses across the dataset).
Feature vectors are zero-padded to 176 dims for symmetry with our
usual padding convention.

The driving FSM has three states:
    OBSERVING_FEW    -- count(symptoms_so_far) < 3
    OBSERVING_ENOUGH -- count(symptoms_so_far) >= 3 (sigma-load-bearing)
    DIAGNOSED        -- terminal

The walker disambiguates the count-conditional OBSERVING_FEW transition
at runtime: from FEW on a symptom token, we land in FEW if
count_after < 3, else ENOUGH. The legality matrix is label-agnostic
and contains True for both targets, so the architecture's mask-learning
machinery sees BOTH continuations as legal -- which they are.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nga.arch.graph_fsm import GraphFSM


__all__ = [
    "DiagnosticSample",
    "DiagnosticDataset",
    "FEATURE_DIM",
    "ALPHABET_SIZE",
    "PROMOTION_THRESHOLD",
    "load_diagnostic_csv",
    "linearise_patient",
    "walk_fsm_diagnostic",
    "generate_diagnostic_dataset",
    "train_test_split_by_patient",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_DIM: int = 176
"""Feature vector dimension; one-hot of the observed token over a 173-entry
alphabet, zero-padded to 176 dims for convenience."""

ALPHABET_SIZE: int = 173
"""132 symptom tokens + 41 diagnosis tokens. The remaining 3 dims of the
176-wide feature vector are pure padding."""

PROMOTION_THRESHOLD: int = 3
"""When count(symptoms_so_far) reaches this value, the walker promotes
OBSERVING_FEW -> OBSERVING_ENOUGH on the symptom that crosses the threshold.
"""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticSample:
    """One per-step transition sample in a patient's parse trace."""

    sample_id: str
    features: np.ndarray
    observed_token: str
    prev_state: str
    current_state: str
    true_next_state: str
    is_adversarial: bool
    patient_id: int
    is_diagnosis_step: bool


@dataclass
class DiagnosticDataset:
    """Container for diagnostic samples plus materialised numpy arrays."""

    samples: list[DiagnosticSample]
    feature_dim: int
    fsm: GraphFSM
    symptom_names: list[str]
    diagnosis_names: list[str]
    n_patients: int
    X: np.ndarray
    y_next: np.ndarray
    prev_states: np.ndarray
    current_states: np.ndarray
    is_adversarial: np.ndarray
    is_diagnosis_step: np.ndarray
    patient_ids: np.ndarray


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def load_diagnostic_csv(
    path: str | Path,
) -> tuple[list[str], list[list[int]], list[str]]:
    """Load the Kaggle Disease-Prediction-from-Symptoms CSV.

    The training CSV has a trailing empty column after `prognosis`; we
    strip it transparently. The test CSV does not have the trailing
    empty column. Both layouts are accepted.

    Returns
    -------
    symptom_names:
        The 132 symptom column names in canonical order.
    per_patient_indicators:
        A list of length n_patients; each entry is a list of 132
        ints (0 or 1) indicating which symptoms that patient presents.
    diagnoses:
        A list of length n_patients with the prognosis label string for
        each patient (canonicalised: stripped of leading/trailing
        whitespace).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: CSV is empty")

    header = rows[0]
    # Strip a trailing empty column header if present (training CSV layout).
    if header and header[-1] == "":
        header = header[:-1]
        rows = [r[: len(header)] if len(r) > len(header) else r for r in rows]

    if "prognosis" not in header:
        raise ValueError(
            f"{path}: header missing 'prognosis' column; got {header!r}"
        )
    prog_idx = header.index("prognosis")
    if prog_idx != len(header) - 1:
        raise ValueError(
            f"{path}: 'prognosis' must be the last column "
            f"(got idx={prog_idx}, n_cols={len(header)})"
        )
    raw_symptom_names = header[:prog_idx]
    if len(raw_symptom_names) != 132:
        raise ValueError(
            f"{path}: expected 132 symptom columns; got {len(raw_symptom_names)}"
        )
    # Disambiguate duplicate symptom column names (the public Kaggle CSV
    # has a repeated 'fluid_overload' column, for example). We append a
    # ``__<col_index>`` suffix to the second-and-later occurrences so
    # every emitted symptom name is unique while staying deterministic
    # in column order.
    seen: dict[str, int] = {}
    symptom_names: list[str] = []
    for col_idx, raw in enumerate(raw_symptom_names):
        if raw in seen:
            seen[raw] += 1
            symptom_names.append(f"{raw}__{col_idx}")
        else:
            seen[raw] = 1
            symptom_names.append(raw)

    indicators: list[list[int]] = []
    diagnoses: list[str] = []
    for line_no, row in enumerate(rows[1:], start=2):
        # Right-pad / right-trim to header width to tolerate the trailing
        # empty cell variant on a per-row basis.
        if len(row) < len(header):
            raise ValueError(
                f"{path}:{line_no}: row has {len(row)} cells but header has "
                f"{len(header)} columns"
            )
        active_row = row[: len(header)]
        try:
            ind = [int(x) for x in active_row[:prog_idx]]
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_no}: non-integer symptom entry: {exc}"
            ) from exc
        for v in ind:
            if v not in (0, 1):
                raise ValueError(
                    f"{path}:{line_no}: symptom entries must be 0 or 1; "
                    f"found {v}"
                )
        diagnosis = active_row[prog_idx].strip()
        if not diagnosis:
            raise ValueError(
                f"{path}:{line_no}: empty prognosis label"
            )
        indicators.append(ind)
        diagnoses.append(diagnosis)

    return symptom_names, indicators, diagnoses


# ---------------------------------------------------------------------------
# Linearisation
# ---------------------------------------------------------------------------


def linearise_patient(
    symptom_indicators: list[int] | np.ndarray,
    diagnosis: str,
    symptom_id_for: dict[str, int],
    diagnosis_id_for: dict[str, int],
) -> list[str]:
    """Linearise a patient's row into a token stream.

    The output sequence is

        [symptom_token for i, v in enumerate(indicators) if v == 1] + [diagnosis]

    using canonical (column-index) ordering -- deterministic. The
    parameters ``symptom_id_for`` / ``diagnosis_id_for`` are passed for
    validation: the function asserts every emitted token is in their
    union, so a typo in the data would fail loudly.
    """
    indicators = list(symptom_indicators)
    if len(indicators) != len(symptom_id_for):
        raise ValueError(
            f"Indicator length {len(indicators)} disagrees with "
            f"len(symptom_id_for)={len(symptom_id_for)}"
        )

    symptom_names_in_order = sorted(
        symptom_id_for.items(), key=lambda kv: kv[1]
    )
    if diagnosis not in diagnosis_id_for:
        raise KeyError(
            f"Diagnosis {diagnosis!r} not in diagnosis_id_for "
            f"(have {len(diagnosis_id_for)} known diagnoses)"
        )

    seq: list[str] = []
    for (name, _idx), v in zip(symptom_names_in_order, indicators):
        if int(v) == 1:
            seq.append(name)
    seq.append(diagnosis)
    return seq


# ---------------------------------------------------------------------------
# FSM walker
# ---------------------------------------------------------------------------


def walk_fsm_diagnostic(
    token_stream: list[str],
    fsm: GraphFSM,
    *,
    diagnosis_set: set[str],
) -> list[tuple[str, str, str]]:
    """Walk the 3-state diagnostic FSM through ``token_stream``.

    Returns ``[(prev_state, observed_token, next_state), ...]``. Walks
    the count-conditional transition by tracking the running symptom
    count: from OBSERVING_FEW on a symptom token, the walker promotes
    to OBSERVING_ENOUGH iff count_after >= PROMOTION_THRESHOLD.

    Parameters
    ----------
    token_stream:
        The linearised tokens (symptoms followed by a single diagnosis).
    fsm:
        The diagnostic FSM (used only for legality assertions and the
        canonical state names).
    diagnosis_set:
        The set of diagnosis token strings; used to distinguish a
        symptom token from a diagnosis token at runtime.
    """
    out: list[tuple[str, str, str]] = []
    state = "OBSERVING_FEW"
    n_symptoms_seen = 0
    for tok in token_stream:
        if tok in diagnosis_set:
            # Diagnosis tokens are only legal at OBSERVING_ENOUGH.
            assert state == "OBSERVING_ENOUGH", (
                f"Diagnosis token {tok!r} observed at illegal state "
                f"{state!r} (need OBSERVING_ENOUGH)"
            )
            next_state = "DIAGNOSED"
        else:
            # Symptom token.
            if state == "OBSERVING_FEW":
                n_symptoms_seen += 1
                if n_symptoms_seen >= PROMOTION_THRESHOLD:
                    next_state = "OBSERVING_ENOUGH"
                else:
                    next_state = "OBSERVING_FEW"
            elif state == "OBSERVING_ENOUGH":
                n_symptoms_seen += 1
                next_state = "OBSERVING_ENOUGH"
            else:
                raise AssertionError(
                    f"Symptom token observed at terminal state {state!r}"
                )
        assert fsm.is_legal_transition(state, next_state), (
            f"FSM legality matrix rejects {state!r} -> {next_state!r}"
        )
        out.append((state, tok, next_state))
        state = next_state
    return out


# ---------------------------------------------------------------------------
# Public dataset generator
# ---------------------------------------------------------------------------


def _expected_state_set() -> list[str]:
    return ["OBSERVING_FEW", "OBSERVING_ENOUGH", "DIAGNOSED"]


def _one_hot_token(token_id: int) -> np.ndarray:
    """Return the 176-dim zero-padded one-hot for ``token_id`` (0..172)."""
    if not (0 <= token_id < ALPHABET_SIZE):
        raise ValueError(
            f"token_id {token_id} out of range [0, {ALPHABET_SIZE})"
        )
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    vec[token_id] = 1.0
    return vec


def _make_token_id_maps(
    symptom_names: list[str],
    diagnoses_seen: list[str],
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """Build the canonical token->id dictionaries.

    Symptom tokens occupy ids 0..131 (column index in the CSV header).
    Diagnosis tokens occupy ids 132..172, ordered alphabetically over
    the union of all distinct diagnoses observed across the dataset.
    """
    if len(symptom_names) != 132:
        raise ValueError(
            f"Expected 132 symptom names; got {len(symptom_names)}"
        )
    if len(set(symptom_names)) != 132:
        raise ValueError("Symptom column names must be unique")
    sym_id_for = {name: i for i, name in enumerate(symptom_names)}

    diagnosis_names = sorted(set(diagnoses_seen))
    if len(diagnosis_names) != 41:
        raise ValueError(
            f"Expected 41 distinct diagnoses; got {len(diagnosis_names)}"
        )
    diag_id_for = {
        name: 132 + i for i, name in enumerate(diagnosis_names)
    }
    return sym_id_for, diag_id_for, diagnosis_names


def _illegal_observed_token(
    state: str,
    n_symptoms_seen: int,
    sym_id_for: dict[str, int],
    diag_id_for: dict[str, int],
    rng: random.Random,
) -> str | None:
    """Return a token-string that is ILLEGAL given the current state.

    OBSERVING_FEW: legal continuations are SYMPTOM tokens. Diagnosis
    tokens are illegal -- pick one at random.
    OBSERVING_ENOUGH: every alphabet token is legal; we synthesise a
    "repeated symptom" or sentinel token by re-flagging adversarial=True
    without changing the token (the walker still advances via the
    legal token). To keep the contract uniform, we instead pretend the
    PRECEDING context is illegal: we return None so the caller leaves
    the observed token unchanged but flags is_adversarial=True only at
    OBSERVING_FEW.
    DIAGNOSED is terminal -- never reached at sample time.
    """
    if state == "OBSERVING_FEW":
        # Diagnosis tokens are illegal at OBSERVING_FEW.
        diag_names = list(diag_id_for.keys())
        if not diag_names:
            return None
        return rng.choice(diag_names)
    return None


def generate_diagnostic_dataset(
    *,
    fsm: GraphFSM,
    csv_path: str | Path,
    n_patients: int | None = None,
    seed: int = 42,
    illegal_temptation_fraction: float = 0.05,
) -> DiagnosticDataset:
    """Build the per-step transition dataset from the Kaggle CSV.

    Parameters
    ----------
    fsm:
        The diagnostic FSM (3 states); must match
        ``_expected_state_set()``.
    csv_path:
        Path to the Kaggle ``training_data.csv`` (or any compatible
        CSV with the same 132-symptom + 1-prognosis layout).
    n_patients:
        If given, sample at most this many patients from the CSV
        deterministically (RNG-shuffled, then prefix). ``None`` means
        use all patients.
    seed:
        RNG seed used for both patient sub-sampling and adversarial
        token corruption.
    illegal_temptation_fraction:
        Per-step probability of replacing the observed token with one
        that is FSM-illegal at the current state. The walker still
        advances along the underlying LEGAL token (so the parse trace
        stays consistent); only the OBSERVED feature differs.
    """
    expected = _expected_state_set()
    if fsm.vertex_ids != expected:
        raise ValueError(
            f"FSM vertex set does not match the diagnostic canonical layout. "
            f"Expected {expected!r}, got {fsm.vertex_ids!r}."
        )
    if not 0.0 <= illegal_temptation_fraction <= 1.0:
        raise ValueError(
            f"illegal_temptation_fraction must be in [0, 1], "
            f"got {illegal_temptation_fraction}"
        )

    symptom_names, indicators, diagnoses = load_diagnostic_csv(csv_path)
    sym_id_for, diag_id_for, diagnosis_names = _make_token_id_maps(
        symptom_names, diagnoses
    )
    diagnosis_set = set(diagnosis_names)

    # Optional subsampling (deterministic).
    n_total = len(indicators)
    if n_patients is not None:
        if n_patients < 1:
            raise ValueError(f"n_patients must be >= 1; got {n_patients}")
        rng_pat = np.random.default_rng(seed)
        order = rng_pat.permutation(n_total)
        keep = order[: min(int(n_patients), n_total)]
        keep_idx = sorted(int(i) for i in keep)
    else:
        keep_idx = list(range(n_total))

    rng = random.Random(seed)
    samples: list[DiagnosticSample] = []

    for patient_id, src_idx in enumerate(keep_idx):
        ind = indicators[src_idx]
        diagnosis = diagnoses[src_idx]
        # Skip patients with zero positive symptoms (they cannot reach
        # OBSERVING_ENOUGH and thus cannot diagnose under our FSM).
        if sum(ind) == 0:
            continue
        token_stream = linearise_patient(
            ind, diagnosis, sym_id_for, diag_id_for
        )

        # Walker bookkeeping inlined so we can also corrupt the observed
        # token symmetrically with python_control's pattern.
        state = "OBSERVING_FEW"
        n_symptoms_seen = 0
        for step_idx, tok in enumerate(token_stream):
            is_diag_tok = tok in diagnosis_set

            # Determine the legal next-state under the walker's
            # count-conditional rule.
            if is_diag_tok:
                if state != "OBSERVING_ENOUGH":
                    # Patient has < 3 symptoms; we already filtered
                    # 0-symptom patients but a 1- or 2-symptom patient
                    # cannot diagnose. We "rescue" by promoting
                    # immediately so the parse trace stays well-formed;
                    # this is rare in practice (most patients have
                    # several symptoms).
                    state = "OBSERVING_ENOUGH"
                next_state = "DIAGNOSED"
            else:
                n_symptoms_seen += 1
                if state == "OBSERVING_FEW":
                    if n_symptoms_seen >= PROMOTION_THRESHOLD:
                        next_state = "OBSERVING_ENOUGH"
                    else:
                        next_state = "OBSERVING_FEW"
                else:
                    next_state = "OBSERVING_ENOUGH"
            assert fsm.is_legal_transition(state, next_state), (
                f"patient {patient_id} step {step_idx}: FSM rejects "
                f"{state!r} -> {next_state!r}"
            )

            observed_token = tok
            is_adv = False
            if illegal_temptation_fraction > 0.0 and rng.random() < illegal_temptation_fraction:
                illegal_tok = _illegal_observed_token(
                    state,
                    n_symptoms_seen,
                    sym_id_for,
                    diag_id_for,
                    rng,
                )
                if illegal_tok is not None:
                    observed_token = illegal_tok
                    is_adv = True

            tok_id = (
                sym_id_for[observed_token]
                if observed_token in sym_id_for
                else diag_id_for[observed_token]
            )
            features = _one_hot_token(tok_id)

            samples.append(
                DiagnosticSample(
                    sample_id=f"patient{patient_id}_step{step_idx}",
                    features=features,
                    observed_token=observed_token,
                    prev_state=state,
                    current_state=state,
                    true_next_state=next_state,
                    is_adversarial=is_adv,
                    patient_id=patient_id,
                    is_diagnosis_step=is_diag_tok,
                )
            )

            state = next_state

    n_kept_patients = (
        max((s.patient_id for s in samples), default=-1) + 1
    )
    return _materialize(
        samples,
        fsm=fsm,
        symptom_names=symptom_names,
        diagnosis_names=diagnosis_names,
        n_patients=n_kept_patients,
    )


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------


def _materialize(
    samples: list[DiagnosticSample],
    *,
    fsm: GraphFSM,
    symptom_names: list[str],
    diagnosis_names: list[str],
    n_patients: int,
) -> DiagnosticDataset:
    n = len(samples)
    X = np.empty((n, FEATURE_DIM), dtype=np.float64)
    y_next = np.empty(n, dtype=np.int64)
    prev_states = np.empty(n, dtype=np.int64)
    is_adv = np.empty(n, dtype=bool)
    is_diag = np.empty(n, dtype=bool)
    pids = np.empty(n, dtype=np.int64)

    vidx = fsm.vertex_index
    for i, s in enumerate(samples):
        X[i] = s.features
        y_next[i] = vidx[s.true_next_state]
        prev_states[i] = vidx[s.current_state]
        is_adv[i] = s.is_adversarial
        is_diag[i] = s.is_diagnosis_step
        pids[i] = s.patient_id

    return DiagnosticDataset(
        samples=samples,
        feature_dim=FEATURE_DIM,
        fsm=fsm,
        symptom_names=list(symptom_names),
        diagnosis_names=list(diagnosis_names),
        n_patients=n_patients,
        X=X,
        y_next=y_next,
        prev_states=prev_states,
        current_states=prev_states.copy(),
        is_adversarial=is_adv,
        is_diagnosis_step=is_diag,
        patient_ids=pids,
    )


def train_test_split_by_patient(
    ds: DiagnosticDataset,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[DiagnosticDataset, DiagnosticDataset]:
    """Partition by patient_id so whole patients are exclusive."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"test_fraction must be in (0, 1), got {test_fraction}"
        )

    pids = sorted({s.patient_id for s in ds.samples})
    rng = np.random.default_rng(seed)
    shuffled = list(pids)
    rng.shuffle(shuffled)  # type: ignore[arg-type]

    n_test = max(1, round(len(shuffled) * test_fraction))
    test_set = set(shuffled[:n_test])
    train_set = set(shuffled[n_test:])

    train_samples = [s for s in ds.samples if s.patient_id in train_set]
    test_samples = [s for s in ds.samples if s.patient_id in test_set]

    train_ds = _materialize(
        train_samples,
        fsm=ds.fsm,
        symptom_names=ds.symptom_names,
        diagnosis_names=ds.diagnosis_names,
        n_patients=len(train_set),
    )
    test_ds = _materialize(
        test_samples,
        fsm=ds.fsm,
        symptom_names=ds.symptom_names,
        diagnosis_names=ds.diagnosis_names,
        n_patients=len(test_set),
    )
    return train_ds, test_ds
