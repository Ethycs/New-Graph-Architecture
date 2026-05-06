"""Typed Field Pipeline arch atom.

Front of the inference pipeline: maps a precomputed feature vector through a
fitted classifier to produce a probability distribution over states and an
argmax predicted label.

Phase 1 scope: thin wrapper around any callable with a `predict_proba` API
(e.g. sklearn LogisticRegression). Grammar-object construction is deferred to
Phase 5 (grammar-compiler). Hyperbolic embedding `z_H` is None until Phase 3.

See docs/arch/typed-field-pipeline.md for the full spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nga.drivers.ablation_flags import AblationTuple

__all__ = ["TypedFieldOutput", "TypedFieldPipeline"]


@dataclass
class TypedFieldOutput:
    """Output of one TypedFieldPipeline inference step.

    Fields
    ------
    predicted_state:
        vertex_id corresponding to argmax of the distribution.
    stratum_label:
        Phase 1 - same as predicted_state (lambda_t = q_t). Empty string when
        typed_scores_enabled is False (signals "untyped" to downstream).
    distribution:
        Shape (V,), non-negative, sums to 1.0.
    confidence:
        max(distribution) - the probability of the top prediction.
    embedding:
        Penultimate-layer dense embedding when available; None for sklearn
        classifiers and until Phase 3 wires the hyperbolic encoder.
    """

    predicted_state: str
    stratum_label: str
    distribution: np.ndarray
    confidence: float
    embedding: np.ndarray | None


class TypedFieldPipeline:
    """Observation -> grammar objects -> typed feature vector.

    Phase 1 scope: takes a precomputed feature vector and a fitted classifier;
    produces a softmax distribution and an argmax label. Grammar-object
    construction is deferred to Phase 5 grammar-compiler. Embedding `z_H` is
    None until Phase 3 hyperbolic-embedding lands.

    Parameters
    ----------
    classifier:
        Any object with a `.predict_proba(X) -> ndarray of shape (n, V)`
        method. For E0 MNIST this is an sklearn LogisticRegression; later
        phases replace it with a torch module.
    vertex_ids:
        Length-V list of vertex id strings. Column i of predict_proba maps to
        vertex_ids[i]. Must match the distribution dimension exactly.
    ablation:
        AblationTuple controlling which components are active. When
        typed_scores_enabled is False the stratum_label is set to "" to signal
        "untyped" mode to downstream consumers.
    """

    def __init__(
        self,
        classifier: object,
        vertex_ids: list[str],
        ablation: AblationTuple,
    ) -> None:
        if not vertex_ids:
            raise ValueError("vertex_ids must be non-empty")
        self._classifier = classifier
        self._vertex_ids: list[str] = list(vertex_ids)
        self._ablation = ablation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_one(self, x: np.ndarray) -> TypedFieldOutput:
        """Run inference on a single feature vector.

        Parameters
        ----------
        x:
            1-D feature vector of shape (F,).

        Returns
        -------
        TypedFieldOutput
        """
        X = x[np.newaxis, :]  # shape (1, F)
        return self.predict_batch(X)[0]

    def predict_batch(self, X: np.ndarray) -> list[TypedFieldOutput]:
        """Run inference on a batch of feature vectors.

        Parameters
        ----------
        X:
            2-D array of shape (n, F).

        Returns
        -------
        List of n TypedFieldOutput objects, one per row of X.
        """
        proba: np.ndarray = self._classifier.predict_proba(X)  # type: ignore[union-attr]
        n, v = proba.shape

        if v != len(self._vertex_ids):
            raise ValueError(
                f"predict_proba returned {v} columns but vertex_ids has "
                f"{len(self._vertex_ids)} entries"
            )

        outputs: list[TypedFieldOutput] = []
        for i in range(n):
            dist = proba[i]  # shape (V,)
            argmax_idx: int = int(np.argmax(dist))
            predicted_state = self._vertex_ids[argmax_idx]
            confidence = float(dist[argmax_idx])

            if self._ablation.typed_scores_enabled:
                stratum_label = predicted_state
            else:
                stratum_label = ""

            outputs.append(
                TypedFieldOutput(
                    predicted_state=predicted_state,
                    stratum_label=stratum_label,
                    distribution=dist,
                    confidence=confidence,
                    embedding=None,  # None until Phase 3
                )
            )

        return outputs
