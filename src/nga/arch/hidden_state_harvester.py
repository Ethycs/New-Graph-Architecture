"""Hidden-state harvester: collect a network's per-step activations.

Phase 20 / graph-extraction Wave A atom. This module gives the rest of the
extraction pipeline a *substrate-agnostic* way to obtain the stream
``{h_t}`` of hidden states a trained model emits on a corpus, so the
clustering + Phase A pipeline that already works on input vectors can be
re-pointed at the model's internal representations.

The contract
============

Given:

* a corpus ``X`` (numpy array of shape ``(N, ...)`` or an iterable of
  per-sample inputs), and
* a callable ``encode_fn`` mapping one input to a hidden-state vector
  (shape ``(d,)``) -- or, for sequence models, a ``(T, d)`` stack,

the harvester returns a contiguous ``np.ndarray`` of shape
``(N_total_steps, d)`` containing every harvested hidden state, plus a
parallel ``sample_index`` array of shape ``(N_total_steps,)`` recording
which sample each step came from. The pair ``(hidden_states,
sample_index)`` is the input the graph-extraction runner needs:
clustering acts on rows; transition counting groups consecutive rows
with the same ``sample_index``.

Substrate adapters
==================

The harvester accepts three substrate flavours; each is a thin adapter
around the underlying model:

1. **Plain callable** -- the user passes any callable ``x -> h``; the
   harvester just iterates the corpus and stacks the outputs. Use this
   for sklearn pipelines (``decision_function``) or a hand-written
   embedding function.
2. **Torch ``nn.Module`` with hook** -- the user passes ``(module,
   layer_name)``; the harvester registers a forward hook on the named
   submodule, runs the corpus through ``module(...)`` in inference mode,
   captures the layer's output, and unregisters. The output is detached
   to numpy and consumed by step 1.
3. **Iterable of per-sample sequences** -- the user passes an iterable
   yielding ``(T_i, d)`` arrays; the harvester concatenates them in
   order, recording ``sample_index = i`` for every row of sample ``i``.

Determinism
===========

The hidden states are returned in corpus order, with stable row ordering
across calls under fixed inputs. ``torch`` adapter sets ``model.eval()``
and uses ``torch.no_grad()`` so dropout / batch-norm running stats do
not perturb the harvest. The harvester never trains the model.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

try:  # Lazy / optional torch dependency.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only on torch-less envs.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

__all__ = ["HiddenStateHarvester", "HarvestResult"]


_TORCH_MISSING_MSG = (
    "torch is required for the torch-adapter; install via "
    "`pixi add pytorch` (or `pip install torch`). The plain-callable "
    "adapter does NOT require torch."
)


class HarvestResult:
    """Container holding the harvest's two parallel arrays.

    Attributes
    ----------
    hidden_states:
        Float array of shape ``(N_total_steps, hidden_dim)``. One row per
        emitted hidden state, in corpus order.
    sample_index:
        Integer array of shape ``(N_total_steps,)`` whose entries name
        which input sample each row came from (0-indexed).
    """

    __slots__ = ("hidden_states", "sample_index")

    def __init__(
        self,
        hidden_states: np.ndarray,
        sample_index: np.ndarray,
    ) -> None:
        if hidden_states.ndim != 2:
            raise ValueError(
                f"hidden_states must be 2D (N_total_steps, hidden_dim), "
                f"got shape {hidden_states.shape}"
            )
        if sample_index.ndim != 1:
            raise ValueError(
                f"sample_index must be 1D (N_total_steps,), got shape "
                f"{sample_index.shape}"
            )
        if hidden_states.shape[0] != sample_index.shape[0]:
            raise ValueError(
                f"hidden_states.shape[0] ({hidden_states.shape[0]}) must "
                f"equal sample_index.shape[0] ({sample_index.shape[0]})"
            )
        self.hidden_states = hidden_states
        self.sample_index = sample_index

    @property
    def n_steps(self) -> int:
        """Total number of harvested hidden states (rows)."""
        return int(self.hidden_states.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of distinct samples represented in the harvest."""
        return int(self.sample_index.max()) + 1 if self.n_steps else 0

    @property
    def hidden_dim(self) -> int:
        """Width of each hidden-state row."""
        return int(self.hidden_states.shape[1]) if self.n_steps else 0


class HiddenStateHarvester:
    """Substrate-agnostic adapter that yields a ``(rows, sample_index)``
    harvest from any trained system with an accessible hidden-state stream.

    Three constructor flavours; pick the one matching the substrate.

    1. **Plain callable** -- ``HiddenStateHarvester.from_callable(fn)``.
       ``fn(x)`` must return a 1-D array (one hidden state per call) or
       a 2-D array (one row per step for that sample).
    2. **Torch hook** -- ``HiddenStateHarvester.from_torch_module(module,
       layer_name)``. Registers a forward hook on ``module.<layer_name>``;
       the hook captures the layer's output for each forward pass. The
       module must have a callable interface ``module(x) -> y``.
    3. **Pre-harvested sequences** -- ``HiddenStateHarvester.from_sequences(
       sequences)``. ``sequences`` is an iterable of ``(T_i, d)`` arrays;
       useful when the user has already extracted hidden states by some
       other means.

    Calling ``harvest(corpus)`` returns a :class:`HarvestResult`.
    """

    def __init__(
        self,
        encode_fn: Callable[[Any], np.ndarray],
        per_sample_emits_sequence: bool = False,
    ) -> None:
        self._encode_fn = encode_fn
        self._per_sample_emits_sequence = per_sample_emits_sequence
        self._cleanup_fn: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_callable(
        cls,
        fn: Callable[[Any], np.ndarray],
        per_sample_emits_sequence: bool = False,
    ) -> HiddenStateHarvester:
        """Wrap any callable ``x -> h``.

        Parameters
        ----------
        fn:
            Callable. Should return either a 1-D vector (``(d,)``, one row
            per sample) or a 2-D matrix (``(T, d)``, one row per step in
            the sample). Set ``per_sample_emits_sequence=True`` for the
            2-D case.
        """
        return cls(encode_fn=fn, per_sample_emits_sequence=per_sample_emits_sequence)

    @classmethod
    def from_torch_module(
        cls,
        module: Any,
        layer_name: str | None = None,
        per_sample_emits_sequence: bool = False,
    ) -> HiddenStateHarvester:
        """Register a forward hook on the named submodule and harvest its
        output on each forward pass.

        If ``layer_name`` is ``None``, the module's own output is used
        (equivalent to wrapping the module itself).

        Parameters
        ----------
        module:
            A torch ``nn.Module``-like object callable on the inputs.
        layer_name:
            Dotted path to the submodule whose output should be captured
            (e.g., ``"encoder.layers.2"``). If ``None``, captures the
            module's top-level output.
        per_sample_emits_sequence:
            Set to ``True`` when each forward pass emits a ``(T, d)``
            stack (sequence models).
        """
        if torch is None:  # pragma: no cover - import-time guard
            raise ImportError(_TORCH_MISSING_MSG)

        captured: dict[str, Any] = {"out": None}
        handle = None

        if layer_name is not None:
            target = module
            for piece in layer_name.split("."):
                target = getattr(target, piece)

            def _hook(_mod: Any, _inp: Any, out: Any) -> None:
                captured["out"] = out

            handle = target.register_forward_hook(_hook)

        def _encode(x: Any) -> np.ndarray:
            module.eval()
            with torch.no_grad():
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
                y = module(x)
                out = captured["out"] if layer_name is not None else y
                if isinstance(out, tuple):
                    out = out[0]
                arr = out.detach().cpu().numpy()
                # If the model batched a single sample, drop the leading
                # batch dim so the caller sees (T, d) or (d,).
                if arr.ndim >= 2 and arr.shape[0] == 1 and not per_sample_emits_sequence:
                    arr = arr[0]
                elif arr.ndim == 3 and arr.shape[0] == 1 and per_sample_emits_sequence:
                    arr = arr[0]
                return arr

        inst = cls(
            encode_fn=_encode,
            per_sample_emits_sequence=per_sample_emits_sequence,
        )
        inst._cleanup_fn = handle.remove if handle is not None else None
        return inst

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[np.ndarray],
    ) -> HiddenStateHarvester:
        """Wrap a pre-computed iterable of per-sample sequences.

        Each yielded array must be 2-D of shape ``(T_i, d)``. The harvester
        does not call into any model; ``harvest`` consumes the iterable and
        stacks it.
        """
        seq_iter = iter(sequences)

        def _encode(_x: Any) -> np.ndarray:
            return np.asarray(next(seq_iter), dtype=np.float64)

        return cls(encode_fn=_encode, per_sample_emits_sequence=True)

    # ------------------------------------------------------------------
    # Harvest
    # ------------------------------------------------------------------

    def harvest(self, corpus: Iterable[Any]) -> HarvestResult:
        """Run the encoder over the corpus and stack the hidden states.

        Parameters
        ----------
        corpus:
            Iterable of per-sample inputs. Each input is passed unchanged
            to the encode function.

        Returns
        -------
        HarvestResult with parallel ``hidden_states`` and ``sample_index``
        arrays.
        """
        rows: list[np.ndarray] = []
        indices: list[np.ndarray] = []

        for i, x in enumerate(corpus):
            h = np.asarray(self._encode_fn(x), dtype=np.float64)
            if self._per_sample_emits_sequence:
                if h.ndim == 1:
                    # The user said sequences but the encoder emitted a vector;
                    # treat it as a length-1 sequence.
                    h = h[np.newaxis, :]
                if h.ndim != 2:
                    raise ValueError(
                        f"per_sample_emits_sequence=True but encoder "
                        f"emitted shape {h.shape} for sample {i} (need 2-D)"
                    )
                rows.append(h)
                indices.append(np.full(h.shape[0], i, dtype=np.int64))
            else:
                if h.ndim == 2 and h.shape[0] == 1:
                    h = h[0]
                if h.ndim != 1:
                    raise ValueError(
                        f"per_sample_emits_sequence=False but encoder "
                        f"emitted shape {h.shape} for sample {i} "
                        f"(need 1-D or (1, d))"
                    )
                rows.append(h[np.newaxis, :])
                indices.append(np.array([i], dtype=np.int64))

        if not rows:
            return HarvestResult(
                hidden_states=np.zeros((0, 0), dtype=np.float64),
                sample_index=np.zeros((0,), dtype=np.int64),
            )

        hidden_states = np.concatenate(rows, axis=0)
        sample_index = np.concatenate(indices, axis=0)
        return HarvestResult(hidden_states=hidden_states, sample_index=sample_index)

    def close(self) -> None:
        """Release any registered torch hook (no-op for non-torch adapters)."""
        if self._cleanup_fn is not None:
            self._cleanup_fn()
            self._cleanup_fn = None

    def __enter__(self) -> HiddenStateHarvester:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
