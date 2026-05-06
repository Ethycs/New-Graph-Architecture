"""Typed Readout Layer arch atom.

Implements the per-type readout heads h_t: R^d -> Y_t described in
docs/arch/typed-readout-layer.md. Each ``type_id`` (e.g. one of
``SingularityType`` tags or a per-stratum identifier) owns an independent
sklearn ``LogisticRegression`` instance. The readout heads are the SOLE
trainable component on top of the frozen encoder backbone.

Per-type isolation contract
---------------------------
1. **Parameter independence.** Heads do not share weights. Fitting head A
   never modifies head B's coefficients. The collection of heads is
   ``dict[type_id, LogisticRegression]``; each entry is constructed with the
   same hyperparameters but is otherwise an independent estimator.
2. **Strict dispatch.** Predict / decision_function / predict_proba dispatch
   on ``type_id``. An unknown ``type_id`` raises ``KeyError`` (it was not
   pre-allocated at construction time). Calling a predict-style method on a
   head that has not yet been fitted raises ``RuntimeError`` so the caller
   cannot silently consume garbage outputs from an unfitted estimator.
3. **Idempotent re-fit.** ``fit`` may be called repeatedly on the same
   ``type_id``. Each call replaces the head's weights with a fresh fit on
   the supplied (X, y) -- this matches sklearn's ``fit`` semantics. The
   readout layer IS trainable; the backbone is not.
4. **Param accounting.** ``n_trainable_params`` counts only fitted heads
   (an unfitted head has no ``coef_`` / ``intercept_`` yet, so it
   contributes zero). The per-type breakdown sums exactly to the total.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

__all__ = ["TypedReadoutLayer"]

TypeId = str | int


class TypedReadoutLayer:
    """Per-type sklearn LogisticRegression heads with strict dispatch.

    The layer pre-allocates one ``LogisticRegression`` per ``type_id`` at
    construction time. Heads share construction kwargs (``logreg_kwargs``)
    but are otherwise fully independent estimators -- fitting one does not
    affect the others. See module docstring for the full isolation contract.

    Parameters
    ----------
    type_ids:
        Iterable of distinct identifiers, one per head. Strings or ints are
        both supported (e.g. ``SingularityType`` member values, integer
        stratum indices). Order is preserved for the ``n_trainable_params_*``
        accessors. Duplicates raise ``ValueError``.
    n_classes:
        Declared number of label classes per head. Currently advisory --
        sklearn infers ``classes_`` from ``y`` at fit time -- but recorded
        for downstream consumers and validated to be ``>= 2``.
    **logreg_kwargs:
        Forwarded verbatim to each ``LogisticRegression(...)`` constructor.
    """

    def __init__(
        self,
        type_ids: list[TypeId],
        n_classes: int,
        **logreg_kwargs: Any,
    ) -> None:
        if not type_ids:
            raise ValueError("type_ids must be non-empty")
        if len(set(type_ids)) != len(type_ids):
            raise ValueError(f"type_ids must be unique; got {type_ids!r}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2; got {n_classes}")

        self._type_ids: list[TypeId] = list(type_ids)
        self._n_classes: int = int(n_classes)
        self._logreg_kwargs: dict[str, Any] = dict(logreg_kwargs)

        # One independent estimator per type_id. Heads are unfitted until
        # ``fit(type_id, ...)`` is called.
        self._heads: dict[TypeId, LogisticRegression] = {
            t: LogisticRegression(**self._logreg_kwargs) for t in self._type_ids
        }
        # Track fit status explicitly: sklearn raises NotFittedError when
        # we ask, but we want a cheap, exception-free predicate.
        self._fitted: dict[TypeId, bool] = {t: False for t in self._type_ids}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_head(self, type_id: TypeId) -> LogisticRegression:
        """Return the head for ``type_id`` or raise ``KeyError``."""
        if type_id not in self._heads:
            raise KeyError(
                f"unknown type_id {type_id!r}; "
                f"known type_ids: {list(self._heads.keys())!r}"
            )
        return self._heads[type_id]

    def _get_fitted_head(self, type_id: TypeId) -> LogisticRegression:
        """Return a fitted head or raise (KeyError unknown / RuntimeError unfit)."""
        head = self._get_head(type_id)
        if not self._fitted[type_id]:
            raise RuntimeError(
                f"readout head for type_id {type_id!r} is not fitted; "
                f"call .fit({type_id!r}, X, y) first"
            )
        return head

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def type_ids(self) -> list[TypeId]:
        """Return the list of registered type_ids in construction order."""
        return list(self._type_ids)

    @property
    def n_classes(self) -> int:
        """Return the declared number of classes per head."""
        return self._n_classes

    def is_fitted(self, type_id: TypeId) -> bool:
        """Return True iff the head for ``type_id`` has been fitted.

        Raises ``KeyError`` for unknown ``type_id`` (consistent with
        predict-style methods).
        """
        # Validate type_id first so callers cannot mask typos with False.
        self._get_head(type_id)
        return self._fitted[type_id]

    def fit(
        self, type_id: TypeId, X: np.ndarray, y: np.ndarray
    ) -> "TypedReadoutLayer":
        """Fit (or re-fit) the head for ``type_id``.

        Idempotent re-fit is allowed -- each call replaces the head's
        coefficients with a fresh sklearn fit on (X, y). Other heads are
        untouched (per-type isolation).
        """
        head = self._get_head(type_id)
        head.fit(X, y)
        self._fitted[type_id] = True
        return self

    def predict(self, type_id: TypeId, X: np.ndarray) -> np.ndarray:
        """Return predicted labels for X under the head for ``type_id``."""
        head = self._get_fitted_head(type_id)
        return np.asarray(head.predict(X))

    def predict_proba(self, type_id: TypeId, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape ``(n_samples, n_classes_seen)``."""
        head = self._get_fitted_head(type_id)
        return np.asarray(head.predict_proba(X))

    def decision_function(self, type_id: TypeId, X: np.ndarray) -> np.ndarray:
        """Return raw margin / logits, matching sklearn convention.

        Shape is ``(n_samples,)`` for binary heads and
        ``(n_samples, n_classes_seen)`` for multiclass heads.
        """
        head = self._get_fitted_head(type_id)
        return np.asarray(head.decision_function(X))

    def n_trainable_params(self) -> int:
        """Return total trainable parameter count across all fitted heads.

        Unfitted heads contribute zero (no ``coef_`` / ``intercept_`` exist
        yet). For each fitted head we count
        ``coef_.size + intercept_.size``.
        """
        return sum(self.n_trainable_params_per_type().values())

    def n_trainable_params_per_type(self) -> dict[TypeId, int]:
        """Per-type trainable parameter count.

        Sums to ``n_trainable_params()`` exactly. Unfitted heads map to 0.
        """
        out: dict[TypeId, int] = {}
        for t in self._type_ids:
            if self._fitted[t]:
                head = self._heads[t]
                coef = np.asarray(head.coef_)
                intercept = np.asarray(head.intercept_)
                out[t] = int(coef.size) + int(intercept.size)
            else:
                out[t] = 0
        return out
