"""Frozen Encoder Backbone arch atom.

Reservoir-style encoder h_theta(x) -> R^d that is **immutable during training**.
Only downstream readout layers update; the parameters of this module (theta)
are fit exactly once and then sealed.

Substrate
---------
Phase 1 has no torch dependency, so the "pre-trained encoder" of the spec is
realised as the simplest non-trivial fixed nonlinearity available in the
sklearn substrate:

    h(x) = StandardScaler(x) @ W

where ``StandardScaler`` is fit once on the calibration batch (its mean/var
are then frozen) and ``W in R^{input_dim x output_dim}`` is a Gaussian random
projection drawn once with a fixed seed (variance 1/output_dim, the
Johnson-Lindenstrauss-style normalisation).

Both pieces are "weights" in the reservoir-computing sense: they are part of
theta, drawn from a fixed distribution, and never updated. Swapping in a real
LLM checkpoint later is a substrate change, not an interface change - the
public API (``fit``, ``encode``, ``is_frozen``, ``update``) is what the rest
of the system depends on.

Frozen contract
---------------
1. ``fit(X)`` may be called **at most once**. Subsequent calls raise
   ``RuntimeError("frozen encoder already fit")``.
2. ``update(...)`` always raises ``RuntimeError("frozen encoder is
   immutable")``. It exists so that an upstream optimiser that walks a list
   of modules and dispatches ``.update(...)`` cannot silently mutate this
   atom - the failure is loud.
3. ``encode(X)`` is pure: same input, same output, no internal state change.

See ``docs/arch/frozen-encoder-backbone.md`` for the full spec.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

__all__ = ["FrozenEncoderBackbone"]


class FrozenEncoderBackbone:
    """Fixed-parameter encoder that maps R^{input_dim} -> R^{output_dim}.

    Parameters
    ----------
    output_dim:
        Embedding dimensionality d. Must be a positive integer.
    seed:
        Seed for the random projection matrix W. Two backbones constructed
        with the same ``seed`` and fit on identical data produce identical
        embeddings (this is what makes "frozen" reproducible).
    """

    def __init__(self, output_dim: int = 32, seed: int = 42) -> None:
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        self.output_dim: int = int(output_dim)
        self._seed: int = int(seed)

        # Frozen state, populated exactly once by ``fit``.
        self._scaler: StandardScaler | None = None
        self._W: np.ndarray | None = None
        self._input_dim: int | None = None
        self._is_frozen: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "FrozenEncoderBackbone":
        """Fit the scaler and draw the projection matrix. Idempotent-by-error.

        Parameters
        ----------
        X:
            Calibration batch of shape (n, input_dim). Used solely to fit the
            ``StandardScaler``; rows are not stored.

        Returns
        -------
        self, for chaining.

        Raises
        ------
        RuntimeError
            If ``fit`` has already been called on this instance.
        """
        if self._is_frozen:
            raise RuntimeError("frozen encoder already fit")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, input_dim); got shape {X.shape}")

        input_dim = int(X.shape[1])
        scaler = StandardScaler()
        scaler.fit(X)

        rng = np.random.default_rng(self._seed)
        # Variance 1/output_dim keeps the embedding scale roughly invariant
        # to ``output_dim`` (Johnson-Lindenstrauss normalisation).
        scale = 1.0 / np.sqrt(self.output_dim)
        W = rng.normal(loc=0.0, scale=scale, size=(input_dim, self.output_dim))

        self._scaler = scaler
        self._W = W
        self._input_dim = input_dim
        self._is_frozen = True
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode a batch into R^{output_dim}. Pure; raises if not yet fit.

        Parameters
        ----------
        X:
            Array of shape (n, input_dim).

        Returns
        -------
        np.ndarray
            Embeddings of shape (n, output_dim).
        """
        if not self._is_frozen or self._scaler is None or self._W is None:
            raise RuntimeError("frozen encoder must be fit before encode")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, input_dim); got shape {X.shape}")
        if X.shape[1] != self._input_dim:
            raise ValueError(
                f"X has {X.shape[1]} features but encoder was fit on "
                f"{self._input_dim}"
            )

        scaled = self._scaler.transform(X)
        return np.asarray(scaled @ self._W, dtype=float)

    def update(self, *args: object, **kwargs: object) -> None:
        """Always raises - the encoder is immutable by construction."""
        raise RuntimeError("frozen encoder is immutable")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        """True once ``fit`` has been called; False before."""
        return self._is_frozen

    @property
    def n_parameters(self) -> int:
        """Total number of stored (non-trainable) weights.

        Counts the projection matrix entries plus the scaler's mean and scale
        vectors. Reported for efficiency accounting only - none of these
        weights participate in gradient updates.
        """
        if not self._is_frozen or self._W is None or self._input_dim is None:
            return 0
        # W: input_dim * output_dim ; scaler stores mean_ and scale_ each of
        # length input_dim.
        return int(self._W.size) + 2 * int(self._input_dim)
