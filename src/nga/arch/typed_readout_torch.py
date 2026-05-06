"""Torch-backed Typed Readout Layer arch atom.

A sibling of :mod:`nga.arch.typed_readout_layer` that swaps the sklearn
``LogisticRegression`` substrate for a small ``torch.nn.Sequential`` MLP
per ``type_id`` (Linear -> ReLU -> Linear, logits output). The public
contract mirrors the sklearn sibling (``fit``, ``predict``,
``predict_proba``, ``decision_function``, ``n_trainable_params``,
``n_trainable_params_per_type``, ``is_fitted``) so a runner can opt into
the torch substrate via a constructor flag without touching downstream
code.

Per-type isolation contract
---------------------------
Identical to the sklearn sibling: heads are independent ``nn.Sequential``
instances, fitting head A never touches head B, predict-style methods
dispatch on ``type_id`` and raise ``KeyError`` for unknown ids /
``RuntimeError`` for unfitted heads.

Lazy import
-----------
``torch`` is imported at module load time but failures are tolerated;
the class methods raise a clear :class:`ImportError` with install
instructions if torch is unavailable. This keeps the arch package
import-safe in environments that do not have torch installed.
"""

from __future__ import annotations

import numpy as np

try:  # Lazy / optional torch dependency.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only on torch-less envs.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

__all__ = ["TypedReadoutTorch"]

TypeId = str | int

_TORCH_MISSING_MSG = (
    "torch is required for TypedReadoutTorch; install via "
    "`pixi add pytorch` (or `pip install torch`)."
)


class TypedReadoutTorch:
    """Per-type torch MLP heads with strict dispatch.

    The layer pre-allocates one ``nn.Sequential(Linear, ReLU, Linear)`` per
    ``type_id`` at construction time. Heads share construction
    hyperparameters (``input_dim``, ``hidden_dim``, ``n_classes``) but are
    otherwise fully independent modules -- fitting one does not touch the
    others. The final ``Linear`` produces raw logits (shape
    ``(n_samples, n_classes)``); softmax is applied only inside
    ``predict_proba``.

    Parameters
    ----------
    type_ids:
        Iterable of distinct identifiers, one per head. Strings or ints
        are both supported. Order is preserved for the
        ``n_trainable_params_per_type`` accessor. Duplicates raise
        ``ValueError``.
    n_classes:
        Number of label classes per head. Must be ``>= 2``. The output
        layer of every head has this many units; ``decision_function``
        returns logits of shape ``(n_samples, n_classes)``.
    input_dim:
        Input feature dimension for every head.
    hidden_dim:
        Width of the hidden layer (default 32).
    lr:
        Adam learning rate used by ``fit`` (default 1e-3).
    n_epochs:
        Number of full-batch Adam epochs run by each ``fit`` call
        (default 50).
    seed:
        Seed driving ``torch.manual_seed`` at construction (controls head
        initialisation) and at the start of every ``fit`` call (controls
        Adam's stochastic behaviour, here deterministic since we use full
        batches but kept defensively for parity with sklearn's
        ``random_state``).
    """

    def __init__(
        self,
        type_ids: list[TypeId],
        n_classes: int,
        input_dim: int,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        n_epochs: int = 50,
        seed: int = 42,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError(_TORCH_MISSING_MSG)
        if not type_ids:
            raise ValueError("type_ids must be non-empty")
        if len(set(type_ids)) != len(type_ids):
            raise ValueError(f"type_ids must be unique; got {type_ids!r}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2; got {n_classes}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {n_epochs}")

        self._type_ids: list[TypeId] = list(type_ids)
        self._n_classes: int = int(n_classes)
        self._input_dim: int = int(input_dim)
        self._hidden_dim: int = int(hidden_dim)
        self._lr: float = float(lr)
        self._n_epochs: int = int(n_epochs)
        self._seed: int = int(seed)

        # Deterministic init: seed BEFORE building modules so each head's
        # Linear layers draw reproducible Kaiming-uniform weights. We seed
        # once globally; the per-type heads then consume successive draws
        # from the (now seeded) generator. Two ``TypedReadoutTorch``
        # instances with the same ``(type_ids, ..., seed)`` produce
        # identical initial weights for every head.
        torch.manual_seed(self._seed)
        self._heads: dict[TypeId, nn.Sequential] = {
            t: self._build_head() for t in self._type_ids
        }
        self._fitted: dict[TypeId, bool] = {t: False for t in self._type_ids}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_head(self) -> "nn.Sequential":
        assert nn is not None  # for type-checkers; guarded in __init__.
        return nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._n_classes),
        )

    def _get_head(self, type_id: TypeId) -> "nn.Sequential":
        if type_id not in self._heads:
            raise KeyError(
                f"unknown type_id {type_id!r}; "
                f"known type_ids: {list(self._heads.keys())!r}"
            )
        return self._heads[type_id]

    def _get_fitted_head(self, type_id: TypeId) -> "nn.Sequential":
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
        return list(self._type_ids)

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def is_fitted(self, type_id: TypeId) -> bool:
        """Return True iff the head for ``type_id`` has been fitted.

        Raises ``KeyError`` for unknown ``type_id`` (consistent with
        predict-style methods).
        """
        self._get_head(type_id)
        return self._fitted[type_id]

    def fit(
        self, type_id: TypeId, X: np.ndarray, y: np.ndarray
    ) -> "TypedReadoutTorch":
        """Fit (or re-fit) the head for ``type_id`` via Adam + cross-entropy.

        Idempotent re-fit: each call rebuilds the head from scratch and
        re-trains for ``n_epochs`` epochs on (X, y). Other heads are
        untouched (per-type isolation). Re-seeding ``torch.manual_seed``
        per fit makes the result reproducible given a fixed
        ``(seed, X, y)``.
        """
        if torch is None or nn is None:
            raise ImportError(_TORCH_MISSING_MSG)
        # Validates type_id presence:
        self._get_head(type_id)

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D (n, input_dim); got shape {X_arr.shape}")
        if X_arr.shape[1] != self._input_dim:
            raise ValueError(
                f"X has {X_arr.shape[1]} features but head expects {self._input_dim}"
            )
        y_arr = np.asarray(y, dtype=np.int64)
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y_arr.shape}")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"len(X)={X_arr.shape[0]} != len(y)={y_arr.shape[0]}"
            )

        # Re-seed and rebuild the head so re-fits are deterministic and
        # do not carry hidden state from a prior fit.
        torch.manual_seed(self._seed)
        head = self._build_head()
        self._heads[type_id] = head
        head.train()

        x_t = torch.from_numpy(X_arr)
        y_t = torch.from_numpy(y_arr)
        optimizer = torch.optim.Adam(head.parameters(), lr=self._lr)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self._n_epochs):
            optimizer.zero_grad()
            logits = head(x_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()

        head.eval()
        self._fitted[type_id] = True
        return self

    def decision_function(self, type_id: TypeId, X: np.ndarray) -> np.ndarray:
        """Return raw logits, shape ``(n_samples, n_classes)``."""
        if torch is None:
            raise ImportError(_TORCH_MISSING_MSG)
        head = self._get_fitted_head(type_id)
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[1] != self._input_dim:
            raise ValueError(
                f"X must be 2-D with {self._input_dim} features; got {X_arr.shape}"
            )
        with torch.no_grad():
            x_t = torch.from_numpy(X_arr)
            logits = head(x_t).detach().cpu().numpy()
        return np.asarray(logits, dtype=float)

    def predict_proba(self, type_id: TypeId, X: np.ndarray) -> np.ndarray:
        """Return softmax class probabilities, shape ``(n_samples, n_classes)``."""
        if torch is None:
            raise ImportError(_TORCH_MISSING_MSG)
        head = self._get_fitted_head(type_id)
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[1] != self._input_dim:
            raise ValueError(
                f"X must be 2-D with {self._input_dim} features; got {X_arr.shape}"
            )
        with torch.no_grad():
            x_t = torch.from_numpy(X_arr)
            probs = torch.softmax(head(x_t), dim=1).detach().cpu().numpy()
        return np.asarray(probs, dtype=float)

    def predict(self, type_id: TypeId, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices, shape ``(n_samples,)``."""
        logits = self.decision_function(type_id, X)
        return np.asarray(np.argmax(logits, axis=1), dtype=np.int64)

    def n_trainable_params(self) -> int:
        """Total trainable parameter count across all heads (fitted or not).

        Unlike the sklearn sibling -- whose ``coef_`` only exists after
        fit -- a torch head's parameter tensors are allocated at
        construction time and are always trainable. To preserve the
        sklearn sibling's "0 before any fit, positive after" semantics we
        only count parameters of heads with ``_fitted[t] == True``.
        """
        return sum(self.n_trainable_params_per_type().values())

    def n_trainable_params_per_type(self) -> dict[TypeId, int]:
        """Per-type trainable parameter count.

        Sums to ``n_trainable_params()`` exactly. Unfitted heads map to 0
        (see :meth:`n_trainable_params` for rationale).
        """
        out: dict[TypeId, int] = {}
        for t in self._type_ids:
            if self._fitted[t]:
                head = self._heads[t]
                out[t] = int(
                    sum(p.numel() for p in head.parameters() if p.requires_grad)
                )
            else:
                out[t] = 0
        return out
