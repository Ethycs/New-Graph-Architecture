"""Torch-backed Frozen Encoder Backbone arch atom.

A sibling of :mod:`nga.arch.frozen_encoder_backbone` that swaps the
sklearn substrate for a small ``torch.nn.Sequential`` MLP. The encoder
is randomly initialised (deterministic under ``seed``) and **all
parameters have ``requires_grad=False``** -- the public contract
(``fit``, ``encode``, ``update``, ``is_frozen``, ``n_parameters``) is
identical to the sklearn sibling so a runner can opt into the torch
substrate via a constructor flag without touching downstream code.

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

__all__ = ["FrozenEncoderTorch"]

_TORCH_MISSING_MSG = (
    "torch is required for FrozenEncoderTorch; install via "
    "`pixi add pytorch` (or `pip install torch`)."
)


class FrozenEncoderTorch:
    """Fixed-parameter torch MLP encoder ``R^{input_dim} -> R^{output_dim}``.

    The encoder is a two-layer ``nn.Sequential`` with a ReLU
    nonlinearity. All parameters are initialised at construction time
    under the supplied ``seed`` and immediately frozen
    (``requires_grad=False``); ``fit`` is therefore a no-op flag flip
    that exists solely to mirror the sklearn sibling's lifecycle.

    Parameters
    ----------
    input_dim:
        Input feature dimension.
    output_dim:
        Output embedding dimension.
    hidden_dim:
        Width of the hidden layer.
    seed:
        Seed driving ``torch.manual_seed`` for parameter initialisation.
        Two encoders constructed with the same ``(input_dim, output_dim,
        hidden_dim, seed)`` tuple produce identical encodings.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError(_TORCH_MISSING_MSG)
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.input_dim: int = int(input_dim)
        self.output_dim: int = int(output_dim)
        self.hidden_dim: int = int(hidden_dim)
        self._seed: int = int(seed)
        self._is_fitted: bool = False

        # Deterministic init: seed before constructing the modules so the
        # Linear layers' Kaiming-uniform draws are reproducible.
        torch.manual_seed(self._seed)
        self._net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        # Freeze ALL parameters. This is enforced in eval mode too via
        # ``self._net.eval()`` so dropout / batchnorm (none here, but
        # defensively) cannot mutate state at encode time.
        for p in self._net.parameters():
            p.requires_grad_(False)
        self._net.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "FrozenEncoderTorch":
        """No-op fit: the torch encoder is randomly initialised and frozen.

        Mirrors the sklearn sibling's lifecycle so callers can use either
        interchangeably. Sets ``is_fitted=True`` and returns ``self``.
        """
        if torch is None:
            raise ImportError(_TORCH_MISSING_MSG)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, input_dim); got shape {X.shape}")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {X.shape[1]} features but encoder expects {self.input_dim}"
            )
        self._is_fitted = True
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode a batch into ``R^{output_dim}`` via the frozen MLP.

        Pure: same input -> same output, no parameter updates.
        """
        if torch is None:
            raise ImportError(_TORCH_MISSING_MSG)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, input_dim); got shape {X.shape}")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {X.shape[1]} features but encoder expects {self.input_dim}"
            )

        # ``no_grad`` keeps the autograd graph empty even though the
        # parameters are frozen -- belt-and-braces for hot encode loops.
        with torch.no_grad():
            x_t = torch.from_numpy(X)
            h_t = self._net(x_t)
            h = h_t.detach().cpu().numpy()
        return np.asarray(h, dtype=float)

    def update(self, *args: object, **kwargs: object) -> None:
        """Always raises -- the torch encoder is immutable by construction."""
        raise RuntimeError("frozen encoder is immutable")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        """True from construction onward -- params are frozen at ``__init__``."""
        return True

    @property
    def is_fitted(self) -> bool:
        """True once ``fit`` has been called (lifecycle parity with sklearn sibling)."""
        return self._is_fitted

    @property
    def n_parameters(self) -> int:
        """Total stored (non-trainable) parameter count of the underlying MLP."""
        if torch is None:
            return 0
        return int(sum(p.numel() for p in self._net.parameters()))
